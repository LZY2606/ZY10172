//! 黑盒验收：启动真实服务（独立临时 db），通过 HTTP 验证数据口径、
//! 删失处理、损伤模型开关导致的样本集合差异、外推审计与重放闭环。

use std::net::TcpListener;
use std::process::{Child, Command};
use std::time::Duration;

fn wait_ready(port: u16, child: &mut Child) {
    let url = format!("http://127.0.0.1:{port}/api/samples");
    for i in 0..120 {
        if let Ok(Some(status)) = child.try_wait() {
            panic!("服务子进程提前退出: {status:?}");
        }
        if let Ok(r) = reqwest::blocking::get(&url) {
            if r.status().as_u16() == 200 {
                return;
            }
        }
        std::thread::sleep(Duration::from_millis(100));
        let _ = i;
    }
    let _ = child.kill();
    panic!("服务未在预期时间内启动: {url}");
}

struct Server {
    child: Child,
    base: String,
    db: String,
}
impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_file(&self.db);
        for suffix in ["-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{db}{suffix}", db = self.db));
        }
    }
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
}

fn spawn() -> Server {
    let port = free_port();
    let db = format!("test-{port}.db");
    let child = Command::new(env!("CARGO_BIN_EXE_rupture-spectrum"))
        .args(["--listen", &format!("127.0.0.1:{port}"), "--db", &db])
        .spawn()
        .expect("启动服务二进制失败");
    let mut s = Server { child, base: format!("http://127.0.0.1:{port}"), db };
    wait_ready(port, &mut s.child);
    s
}

fn post(base: &str, path: &str, body: serde_json::Value) -> (u16, serde_json::Value) {
    let r = reqwest::blocking::Client::new()
        .post(format!("{base}{path}"))
        .json(&body)
        .send()
        .unwrap();
    let status = r.status().as_u16();
    let text = r.text().unwrap();
    let v = serde_json::from_str(&text).unwrap_or_else(|_| serde_json::json!({"_text": text}));
    (status, v)
}

fn get(base: &str, path: &str) -> (u16, serde_json::Value) {
    let r = reqwest::blocking::get(format!("{base}{path}")).unwrap();
    let status = r.status().as_u16();
    let text = r.text().unwrap();
    let v = serde_json::from_str(&text).unwrap_or_else(|_| serde_json::json!({"_text": text}));
    (status, v)
}

fn rupture_fit(damage: bool) -> serde_json::Value {
    serde_json::json!({
        "model": "larson_miller", "criterion": "rupture",
        "damage_model": damage, "reference_heat": "HA",
        "allow_temp_min_c": 400, "allow_temp_max_c": 700,
        "allow_stress_min_mpa": 40, "allow_stress_max_mpa": 320
    })
}

#[test]
fn index_title_and_fixture() {
    let s = spawn();
    let text = reqwest::blocking::get(format!("{}/", s.base)).unwrap().text().unwrap();
    assert!(text.contains("持时破裂谱"), "首页必须包含标题“持时破裂谱”");
    let (_, v) = get(&s.base, "/api/samples");
    let n = v["samples"].as_array().unwrap().len();
    assert_eq!(n, 27, "固定 fixture 含 27 个样本");
    let kinds: std::collections::HashSet<String> = v["samples"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x["load_kind"].as_str().unwrap().to_string())
        .collect();
    assert!(kinds.contains("variable") && kinds.contains("constant"));
}

#[test]
fn damage_toggle_changes_inclusion_set() {
    let s = spawn();
    let (_, off) = post(&s.base, "/api/fits", rupture_fit(false));
    let (_, on) = post(&s.base, "/api/fits", rupture_fit(true));
    let inc_off: std::collections::HashSet<String> = serde_json::from_value(off["included"].clone()).unwrap();
    let inc_on: std::collections::HashSet<String> = serde_json::from_value(on["included"].clone()).unwrap();
    assert!(!inc_off.contains("V-001") && !inc_off.contains("V-002"), "禁用损伤模型：变载样本必须排除");
    assert!(inc_on.contains("V-001") && inc_on.contains("V-002"), "启用损伤模型：变载样本必须纳入");
    // 应变准则样本在两种情况下都不得进入 rupture 拟合
    for id in ["S-001", "S-004"] {
        assert!(!inc_off.contains(id) && !inc_on.contains(id));
    }
    // 排除原因可追溯
    let reasons: Vec<String> = on["excluded"].as_array().unwrap().iter().map(|e| e["id"].as_str().unwrap().into()).collect();
    assert!(reasons.iter().any(|x| x == "S-001"));
    assert!(off["result"]["converged"].as_bool().unwrap());
    assert!(on["result"]["converged"].as_bool().unwrap());
}

#[test]
fn right_censored_is_a_lower_bound_not_an_exact_failure() {
    let s = spawn();
    let (_, fit) = post(&s.base, "/api/fits", rupture_fit(false));
    let residuals = fit["result"]["residuals"].as_array().unwrap();
    let find = |id: &str| residuals.iter().find(|r| r["sample_id"] == id).unwrap().clone();
    // 已断裂样本：有点残差
    let fractured = find("R-001");
    assert_eq!(fractured["obs"], "exact");
    assert!(fractured["residual_dex"].as_f64().is_some());
    // 右删失样本：无点残差（不能当作在截止时刻断裂），但有删失 NLL 贡献
    let censored = find("R-015");
    assert_eq!(censored["obs"], "censored");
    assert!(censored["residual_dex"].is_null());
    assert!(censored["nll_contribution"].as_f64().unwrap().is_finite());
}

#[test]
fn failure_criteria_are_never_mixed() {
    let s = spawn();
    let body = serde_json::json!({
        "model": "larson_miller", "criterion": "strain2pct",
        "damage_model": false, "reference_heat": "HA",
        "allow_temp_min_c": 400, "allow_temp_max_c": 700,
        "allow_stress_min_mpa": 40, "allow_stress_max_mpa": 320
    });
    let (_, fit) = post(&s.base, "/api/fits", body);
    let inc: Vec<String> = fit["included"].as_array().unwrap().iter().map(|x| x.as_str().unwrap().into()).collect();
    assert_eq!(inc.len(), 6);
    assert!(inc.iter().all(|x| x.starts_with("S-")));
}

#[test]
fn temperature_unit_is_required_and_validated() {
    let s = spawn();
    let missing = serde_json::json!({"samples":[{
        "id":"BAD1","heat":"HA","criterion":"rupture","load_kind":"constant",
        "temp_value":550,"stress_mpa":150,"time_h":1000.0,"status":"fractured"}]});
    let (_, v) = post(&s.base, "/api/import", missing);
    assert!(!v["ok"].as_bool().unwrap());
    assert!(v["errors"][0].as_str().unwrap().contains("温度单位"));

    // 开尔文换算正确：823.15 K == 550 °C
    let kelvin = serde_json::json!({"samples":[{
        "id":"K1","heat":"HA","criterion":"rupture","load_kind":"constant",
        "temp_value":823.15,"temp_unit":"K","stress_mpa":150,"time_h":1000.0,"status":"fractured"}]});
    let (_, v) = post(&s.base, "/api/import", kelvin);
    assert!(v["ok"].as_bool().unwrap());
    let (_, samples) = get(&s.base, "/api/samples");
    let k1 = samples["samples"].as_array().unwrap().iter().find(|x| x["id"]=="K1").unwrap();
    assert!((k1["temp_c"].as_f64().unwrap() - 550.0).abs() < 1e-6);
}

#[test]
fn extrapolation_shows_distance_interval_and_boundary_and_can_be_rejected() {
    let s = spawn();
    let (_, fit) = post(&s.base, "/api/fits", rupture_fit(false));
    let fit_id = fit["id"].as_i64().unwrap();

    // 1) 包络内
    let (_, inside) = post(&s.base, "/api/predict", serde_json::json!({
        "fit_id": fit_id, "temp_value": 550, "temp_unit": "C", "stress_mpa": 150 }));
    let p = &inside["prediction"];
    assert_eq!(p["risk"], "interpolated");
    assert!(p["pi_low_h"].as_f64().unwrap() < p["median_h"].as_f64().unwrap());
    assert!(p["median_h"].as_f64().unwrap() < p["pi_high_h"].as_f64().unwrap());

    // 2) 包络外、允许范围内：距离>0、区间、边界活跃、杠杆度
    let (_, outp) = post(&s.base, "/api/predict", serde_json::json!({
        "fit_id": fit_id, "temp_value": 680, "temp_unit": "C", "stress_mpa": 200 }));
    let p = &outp["prediction"];
    assert!(["low", "medium", "high"].contains(&p["risk"].as_str().unwrap()));
    assert!(p["normalized_distance"].as_f64().unwrap() > 0.0);
    assert!(p["extrapolation_leverage"].as_f64().unwrap() >= 0.0);
    assert!(p["boundary_active"].as_array().unwrap().iter().any(|b| b=="temp_max"));
    assert!(p["pi_low_h"].as_f64().is_some());

    // 3) 超出允许外推范围：明确拒绝，不返回曲线延长
    let (code, rej) = post(&s.base, "/api/predict", serde_json::json!({
        "fit_id": fit_id, "temp_value": 900, "temp_unit": "C", "stress_mpa": 200 }));
    assert_eq!(code, 200);
    assert!(rej["rejected"].as_bool().unwrap());
    assert_eq!(rej["prediction"]["risk"], "rejected");
}

#[test]
fn compare_same_criterion_and_block_cross_criterion() {
    let s = spawn();
    let (_, a) = post(&s.base, "/api/fits", rupture_fit(false));
    let (_, b) = post(&s.base, "/api/fits", rupture_fit(true));
    let (_, cmp) = get(&s.base, &format!("/api/compare?a={}&b={}", a["id"], b["id"]));
    assert_eq!(cmp["criterion"], "rupture");
    assert!(cmp["common_samples"].as_i64().unwrap() >= 19);
    assert!(cmp["only_in_b"].as_array().unwrap().iter().any(|x| x=="V-001"));

    let (_, strain) = post(&s.base, "/api/fits", serde_json::json!({
        "model":"larson_miller","criterion":"strain2pct","damage_model":false,"reference_heat":"HA",
        "allow_temp_min_c":400,"allow_temp_max_c":700,"allow_stress_min_mpa":40,"allow_stress_max_mpa":320}));
    let (code, _) = get(&s.base, &format!("/api/compare?a={}&b={}", a["id"], strain["id"]));
    assert_eq!(code, 400);
}

#[test]
fn export_wipe_reimport_and_replay_reproduces_fits() {
    let s = spawn();
    let (_, _fit) = post(&s.base, "/api/fits", rupture_fit(true));
    let (_, bundle) = get(&s.base, "/api/export");
    assert!(bundle["samples_raw"].as_array().unwrap().len() >= 27);
    assert!(!bundle["fit_specs"].as_array().unwrap().is_empty());

    let (_, wiped) = post(&s.base, "/api/reset", serde_json::json!({}));
    assert!(wiped["ok"].as_bool().unwrap());
    let (_, empty) = get(&s.base, "/api/samples");
    assert_eq!(empty["samples"].as_array().unwrap().len(), 0);

    // 空库重放（清空 → 重导入 → 逐条重拟合）
    let (code, rep) = post(&s.base, "/api/replay", serde_json::json!({"bundle": bundle}));
    assert_eq!(code, 200);
    assert!(rep["samples"].as_i64().unwrap() >= 27);
    assert!(rep["refits"].as_array().unwrap().iter().any(|f|
        f["criterion"] == "rupture" && f["included"].as_array().unwrap().iter().any(|x| x=="V-001")));
}

#[test]
fn custom_model_respects_nonpositive_quadratic_constraints() {
    let s = spawn();
    let body = serde_json::json!({
        "model": "custom", "criterion": "rupture", "damage_model": false,
        "reference_heat": "HA",
        "allow_temp_min_c": 400, "allow_temp_max_c": 700,
        "allow_stress_min_mpa": 40, "allow_stress_max_mpa": 320
    });
    let (_, fit) = post(&s.base, "/api/fits", body);
    assert!(fit["result"]["converged"].as_bool().unwrap());
    let get_p = |name: &str| fit["result"]["params"].as_array().unwrap().iter()
        .find(|p| p["name"]==name).unwrap()["value"].as_f64().unwrap();
    assert!(get_p("b2") <= 0.0, "应力二次项必须受限为非正");
    assert!(get_p("b4") <= 0.0, "温度二次项必须受限为非正");
}
