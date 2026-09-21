use axum::body::Body;
use axum::http::{Request, StatusCode};
use bytes::Bytes;
use http_body_util::BodyExt;
use mastercurve_bench::testkit::{spawn_app, AppHandle};
use tower::ServiceExt;

async fn call(app: &AppHandle, method: &str, path: &str, json: Option<&str>) -> (StatusCode, serde_json::Value) {
    let mut builder = Request::builder().method(method).uri(path);
    let body = match json {
        Some(j) => {
            builder = builder.header("content-type", "application/json");
            Body::from(j.to_string())
        }
        None => Body::empty(),
    };
    let resp = app.router.clone().oneshot(builder.body(body).unwrap()).await.unwrap();
    let status = resp.status();
    let bytes: Bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let v = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    (status, v)
}

fn fit_body(model: &str, damage: bool, extra_targets: bool) -> String {
    let mut v = serde_json::json!({
        "model": model,
        "criterion": "rupture",
        "damage_model": damage,
        "shrink": 1.0,
        "min_temp_c": 480.0, "max_temp_c": 625.0,
        "min_stress_mpa": 35.0, "max_stress_mpa": 210.0,
    });
    if extra_targets {
        v["targets"] = serde_json::json!([
          {"temp_value":550,"temp_unit":"C","stress_mpa":100,"heat":"A"},
          {"temp_value":625,"temp_unit":"C","stress_mpa":50,"heat":"A"},
          {"temp_value":650,"temp_unit":"C","stress_mpa":30,"heat":"A"}
        ]);
    }
    v.to_string()
}

#[tokio::test]
async fn full_acceptance_flow() {
    let app = spawn_app().await;

    let (s, health) = call(&app, "GET", "/api/health", None).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(health["ok"], true);

    // 首页包含中文标题
    let resp = app
        .router
        .clone()
        .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let html = resp.into_body().collect().await.unwrap().to_bytes();
    let html = String::from_utf8(html.to_vec()).unwrap();
    assert!(html.contains("持时破裂谱"));

    // 初始为空（每个测试用独立临时库）
    let (_, v) = call(&app, "GET", "/api/specimens", None).await;
    assert_eq!(v["specimens"].as_array().unwrap().len(), 0);

    let (s, v) = call(&app, "POST", "/api/fixtures/load", None).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(v["imported"], 24);
    // 重放：再次导入应全部跳过，不产生重复
    let (_, v2) = call(&app, "POST", "/api/fixtures/load", None).await;
    assert_eq!(v2["skipped_duplicate"], 24);

    // 温度单位缺失必须拒收
    let (_, v) = call(
        &app, "POST", "/api/specimens",
        Some(r#"{"specimens":[{"specimen_id":"BAD","heat":"A","temp_value":550,"stress_mpa":90,"time_h":10,"outcome":"ruptured","criterion":"rupture"}]}"#),
    ).await;
    assert!(v["rejected"][0].as_str().unwrap().contains("缺少温度单位"));
    // 启用损伤模型：21 个 rupture 样本（含 3 变载）
    let (s, on) = call(&app, "POST", "/api/fit", Some(&fit_body("LM", true, true))).await;
    assert_eq!(s, StatusCode::OK, "{}", on);
    let ron = &on["result"];
    assert_eq!(ron["n_included"], 21);
    assert_eq!(ron["n_variable"], 3);
    let incl_on: std::collections::BTreeSet<String> = ron["inclusion"]
        .as_array().unwrap().iter()
        .filter(|r| r["included"] == true)
        .map(|r| r["specimen_id"].as_str().unwrap().to_string()).collect();
    assert!(incl_on.contains("V01") && incl_on.contains("V02") && incl_on.contains("V03"));
    // 参数恢复（噪声较大，宽容差）
    let params: std::collections::HashMap<String, f64> = ron["params"]
        .as_array().unwrap().iter()
        .map(|p| (p["name"].as_str().unwrap().to_string(), p["value"].as_f64().unwrap()))
        .collect();
    assert!((params["C_LM"] - 20.0).abs() < 2.0);
    assert!((params["a1"] - (-13.0)).abs() < 3.0);

    // 禁用损伤模型：变载样本必须排除，集合为 18
    let (s, off) = call(&app, "POST", "/api/fit", Some(&fit_body("LM", false, true))).await;
    assert_eq!(s, StatusCode::OK, "{}", off);
    let roff = &off["result"];
    assert_eq!(roff["n_included"], 18);
    assert_eq!(roff["n_variable"], 0);
    let excl: Vec<String> = roff["inclusion"].as_array().unwrap().iter()
        .filter(|r| r["included"] == false)
        .map(|r| r["specimen_id"].as_str().unwrap().to_string()).collect();
    for v in ["V01", "V02", "V03"] { assert!(excl.contains(&v.to_string()), "{}", v); }
    // 第二准则样本在两种情况下都被排除
    assert!(excl.iter().any(|x| x == "S01"));

    // 右删失条目：绝不出现断裂式残差
    for r in ron["censoring"]["entries"].as_array().unwrap() {
        assert!(r["survival_probability_at_stop"].as_f64().unwrap() <= 1.0);
    }
    assert_eq!(ron["censoring"]["n_censored"], 5);

    // 外推三目标：包络内 / 允许外推 / 被拒绝
    let preds = ron["predictions"].as_array().unwrap();
    assert_eq!(preds[0]["within_envelope"], true);
    assert_eq!(preds[0]["allowed"], true);
    assert_eq!(preds[0]["risk"], "low");
    assert_eq!(preds[1]["within_envelope"], false);
    assert_eq!(preds[1]["allowed"], true);
    assert_eq!(preds[1]["distance_temp_c"], 25.0);
    assert!(preds[1]["pi_low_h"].as_f64().unwrap() < preds[1]["median_h"].as_f64().unwrap());
    assert_eq!(preds[2]["allowed"], false);
    assert!(preds[2]["blocked_by"].as_str().unwrap().contains("625"));
    let ba = preds[2]["boundary_active"].as_array().unwrap();
    assert!(ba.iter().any(|b| b.as_str().unwrap().contains("温度上限")));

    // CX 自定义模型也能拟合同一数据
    let (s, cx) = call(&app, "POST", "/api/fit", Some(&fit_body("CX", true, false))).await;
    assert_eq!(s, StatusCode::OK, "{}", cx);
    assert_eq!(cx["result"]["n_included"], 21);

    // 比较：同集合 LM-on vs CX-on 可以比较；LM-on vs LM-off 必须被拒
    let id_on = on["run_id"].as_i64().unwrap();
    let id_off = off["run_id"].as_i64().unwrap();
    let id_cx = cx["run_id"].as_i64().unwrap();
    let (s, cmp) = call(
        &app, "POST", "/api/compare",
        Some(&format!(r#"{{"run_a":{id_on},"run_b":{id_cx}}}"#)),
    ).await;
    assert_eq!(s, StatusCode::OK, "{}", cmp);
    assert_eq!(cmp["same_samples"], 21);

    let (s, badcmp) = call(
        &app, "POST", "/api/compare",
        Some(&format!(r#"{{"run_a":{id_on},"run_b":{id_off}}}"#)),
    ).await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert!(badcmp["error"].as_str().unwrap().contains("纳入样本集合不同"));

    // 显式 K 合法（在纯净 fixture 拟合断言之后再加入，避免污染样本计数）
    let (sk, _) = call(
        &app, "POST", "/api/specimens",
        Some(r#"{"specimens":[{"specimen_id":"K1","heat":"A","temp_value":823.15,"temp_unit":"K","stress_mpa":90,"time_h":10,"outcome":"ruptured","criterion":"rupture"}]}"#),
    ).await;
    assert_eq!(sk, StatusCode::OK);

    // 准则混合拟合：strain_1pct 与 rupture 绝不混合——该准则只有 3 条，且与 rupture 分开
    let (s, sc) = call(
        &app, "POST", "/api/fit",
        Some(r#"{"model":"LM","criterion":"strain_1pct","damage_model":false}"#),
    ).await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert!(sc["error"].as_str().unwrap().contains("至少需要 4"));

    // 导出包含 runs 与 specimens
    let (s, exp) = call(&app, "GET", "/api/runs/export", None).await;
    assert_eq!(s, StatusCode::OK);
    assert!(exp["runs"].as_array().unwrap().len() >= 3);
    assert!(exp["specimens"].as_array().unwrap().len() >= 25);

    // 清空 -> 复核为空 -> 重新导入 fixture 成功
    let (s, _) = call(&app, "POST", "/api/specimens/clear", None).await;
    assert_eq!(s, StatusCode::OK);
    let (_, v) = call(&app, "GET", "/api/specimens", None).await;
    assert_eq!(v["specimens"].as_array().unwrap().len(), 0);
    let (_, reload) = call(&app, "POST", "/api/fixtures/load", None).await;
    assert_eq!(reload["imported"], 24);

    // 超范围外推边界显式配置：允许到 700°C 后 650 点不再拒绝（但风险仍高）
    let body = r#"{"model":"LM","criterion":"rupture","damage_model":true,
       "min_temp_c":400,"max_temp_c":700,"min_stress_mpa":20,"max_stress_mpa":220,
       "targets":[{"temp_value":650,"temp_unit":"C","stress_mpa":30,"heat":"A"}]}"#;
    let (s, wide) = call(&app, "POST", "/api/fit", Some(body)).await;
    assert_eq!(s, StatusCode::OK, "{}", wide);
    let p = &wide["result"]["predictions"][0];
    assert_eq!(p["allowed"], true);
    assert_eq!(p["risk"], "high");
    assert!(p["distance_temp_c"].as_f64().unwrap() > 0.0);
    assert!(p["distance_stress_mpa"].as_f64().unwrap() > 0.0);
    // 仍在放宽后的允许范围内，故无硬边界告警
    assert!(p["boundary_active"].as_array().unwrap().is_empty());
    // 但必须同时给出区间与距离
    assert!(p["pi_low_h"].as_f64().unwrap() > 0.0);
    assert!(p["pi_high_h"].as_f64().unwrap() > p["pi_low_h"].as_f64().unwrap());

    // 更极端的外推点必须触发杠杆度边界活跃
    let body2 = r#"{"model":"LM","criterion":"rupture","damage_model":true,
       "min_temp_c":300,"max_temp_c":760,"min_stress_mpa":10,"max_stress_mpa":260,
       "targets":[{"temp_value":700,"temp_unit":"C","stress_mpa":20,"heat":"A"}]}"#;
    let (s, far) = call(&app, "POST", "/api/fit", Some(body2)).await;
    assert_eq!(s, StatusCode::OK, "{}", far);
    let pf = &far["result"]["predictions"][0];
    assert_eq!(pf["allowed"], true);
    assert_eq!(pf["risk"], "high");
    assert!(pf["boundary_active"].as_array().unwrap().iter()
        .any(|b| b.as_str().unwrap().contains("杠杆度")), "{}", pf);
}
