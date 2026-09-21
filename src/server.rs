use crate::db::Db;
use crate::ingest::{Sample, SegmentIn};
use crate::stats::{self, envelope, select_observations, FitConfig, FixedHeat};
use crate::models::ModelKind;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{Html, IntoResponse, Json},
    routing::{get, post},
    Router,
};
use serde::Deserialize;
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tower_http::services::ServeDir;

#[derive(Clone)]
pub struct AppState {
    pub db: Arc<Db>,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/api/samples", get(list_samples).post(import_samples))
        .route("/api/import", post(import_samples))
        .route("/api/fits", get(list_fits).post(run_fit))
        .route("/api/fits/:id", get(get_fit))
        .route("/api/predict", post(predict))
        .route("/api/compare", get(compare))
        .route("/api/export", get(export_bundle))
        .route("/api/replay", post(replay))
        .route("/api/reset", post(reset))
        .route("/api/audit", get(audit))
        .nest_service("/static", ServeDir::new("static"))
        .with_state(state)
}

async fn index() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}

type AppErr = (StatusCode, String);
fn err(e: impl std::fmt::Display) -> AppErr {
    (StatusCode::BAD_REQUEST, e.to_string())
}

async fn list_samples(State(st): State<AppState>) -> impl IntoResponse {
    let conn = st.db.0.lock().unwrap();
    let samples = crate::db::list_samples(&conn).unwrap_or_default();
    Json(json!({ "samples": samples }))
}

async fn list_fits(State(st): State<AppState>) -> impl IntoResponse {
    let conn = st.db.0.lock().unwrap();
    Json(json!({ "fits": crate::db::list_fits(&conn).unwrap_or_default() }))
}

async fn get_fit(State(st): State<AppState>, Path(id): Path<i64>) -> Result<Json<serde_json::Value>, AppErr> {
    let conn = st.db.0.lock().unwrap();
    let v = crate::db::get_fit(&conn, id).map_err(err)?
        .ok_or((StatusCode::NOT_FOUND, format!("fit {id} 不存在")))?;
    Ok(Json(v))
}

async fn audit(State(st): State<AppState>) -> impl IntoResponse {
    let conn = st.db.0.lock().unwrap();
    let rows = crate::db::list_audit(&conn).unwrap_or_default()
        .into_iter()
        .map(|(id, action, at)| json!({"id": id, "action": action, "at": at}))
        .collect::<Vec<_>>();
    Json(json!({ "audit": rows }))
}

#[derive(Deserialize)]
struct ImportBody {
    samples: Vec<Sample>,
}

async fn import_samples(
    State(st): State<AppState>,
    Json(body): Json<ImportBody>,
) -> Result<Json<serde_json::Value>, AppErr> {
    let mut seen = HashSet::new();
    let mut valid = Vec::new();
    let mut errors = Vec::new();
    for s in &body.samples {
        match crate::ingest::validate(s, &mut seen) {
            Ok(v) => valid.push(v),
            Err(e) => errors.push(e),
        }
    }
    if !errors.is_empty() {
        return Ok(Json(json!({ "ok": false, "imported": 0, "errors": errors })));
    }
    {
        let mut conn = st.db.0.lock().unwrap();
        let tx = conn.transaction().map_err(err)?;
        for v in &valid {
            crate::db::insert_sample(&tx, v).map_err(err)?;
        }
        crate::db::audit(&tx, "import", &format!("导入 {} 个样本", valid.len()));
        tx.commit().map_err(err)?;
    }
    Ok(Json(json!({ "ok": true, "imported": valid.len(), "errors": errors })))
}

#[derive(Deserialize)]
struct FitBody {
    model: String,
    criterion: String,
    damage_model: bool,
    reference_heat: String,
    #[serde(default)]
    fixed_heats: Vec<FixedHeatIn>,
    #[serde(default = "default_tau")]
    shrink_tau: f64,
    allow_temp_min_c: f64,
    allow_temp_max_c: f64,
    allow_stress_min_mpa: f64,
    allow_stress_max_mpa: f64,
}
fn default_tau() -> f64 {
    0.3
}
#[derive(Deserialize)]
struct FixedHeatIn {
    heat: String,
    bias: f64,
}

fn perform_fit(st: &AppState, body: &FitBody) -> Result<serde_json::Value, AppErr> {
    let kind = ModelKind::parse(&body.model)
        .ok_or_else(|| err(format!("未知模型 '{}'（接受 larson_miller | custom）", body.model)))?;
    if body.allow_temp_min_c >= body.allow_temp_max_c
        || body.allow_stress_min_mpa >= body.allow_stress_max_mpa
    {
        return Err(err("允许外推范围上下限顺序错误"));
    }
    let conn = st.db.0.lock().unwrap();
    let samples = crate::db::list_samples(&conn).map_err(err)?;
    drop(conn);
    let (obs, excluded) = select_observations(&samples, &body.criterion, body.damage_model);
    if obs.is_empty() {
        return Err(err("该准则下没有可纳入样本（变载样本需启用累积损伤模型）"));
    }
    let mut heats: Vec<String> = obs.iter().map(|o| o.heat.clone()).collect();
    heats.sort();
    heats.dedup();
    if !heats.iter().any(|h| h == &body.reference_heat) {
        return Err(err(format!(
            "参考炉次 {} 不在纳入样本炉次 {:?} 中",
            body.reference_heat, heats
        )));
    }
    let cfg = FitConfig {
        kind,
        observations: obs,
        heats: heats.clone(),
        reference_heat: body.reference_heat.clone(),
        fixed_heats: body
            .fixed_heats
            .iter()
            .map(|f| FixedHeat { heat: f.heat.clone(), bias: f.bias })
            .collect(),
        shrink_tau: if body.shrink_tau > 0.0 { body.shrink_tau } else { 0.3 },
        allow_temp_k: (body.allow_temp_min_c + 273.15, body.allow_temp_max_c + 273.15),
        allow_stress: (body.allow_stress_min_mpa, body.allow_stress_max_mpa),
    };
    let included = cfg.observations.iter().map(|o| o.sample_id.clone()).collect::<Vec<_>>();
    let env = envelope(&cfg.observations);
    let out = stats::fit(cfg).ok_or_else(|| err("拟合失败：样本不足或优化未收敛"))?;

    let config_json = serde_json::to_string(&json!({
        "model": body.model,
        "criterion": body.criterion,
        "damage_model": body.damage_model,
        "reference_heat": body.reference_heat,
        "fixed_heats": body.fixed_heats.iter().map(|f| json!({"heat": f.heat, "bias": f.bias})).collect::<Vec<_>>(),
        "shrink_tau": body.shrink_tau,
        "allow_temp_min_c": body.allow_temp_min_c,
        "allow_temp_max_c": body.allow_temp_max_c,
        "allow_stress_min_mpa": body.allow_stress_min_mpa,
        "allow_stress_max_mpa": body.allow_stress_max_mpa,
    })).map_err(err)?;
    let result_json = serde_json::to_string(&out.result).map_err(err)?;
    let env_json = serde_json::to_string(&env).map_err(err)?;
    let excluded_json = serde_json::to_string(
        &excluded.iter().map(|(id, why)| json!({"id": id, "reason": why})).collect::<Vec<_>>(),
    )
    .map_err(err)?;

    let conn = st.db.0.lock().unwrap();
    let id = crate::db::insert_fit(
        &conn, &body.model, &body.criterion, body.damage_model,
        &config_json, &serde_json::to_string(&included).unwrap(),
        &excluded_json, &result_json, &env_json,
    )
    .map_err(err)?;
    crate::db::audit(
        &conn,
        "fit",
        &format!("model={} criterion={} damage_model={} included={}", body.model, body.criterion, body.damage_model, included.len()),
    );
    Ok(json!({
        "id": id,
        "included": included,
        "excluded": excluded.iter().map(|(i, w)| json!({"id": i, "reason": w})).collect::<Vec<_>>(),
        "envelope": env,
        "result": out.result,
    }))
}

async fn run_fit(State(st): State<AppState>, Json(body): Json<FitBody>) -> Result<Json<serde_json::Value>, AppErr> {
    Ok(Json(perform_fit(&st, &body)?))
}

fn reconstruct(st: &AppState, fit_id: i64) -> Result<(stats::FitOutput, stats::Envelope, serde_json::Value, FitBody), AppErr> {
    let conn = st.db.0.lock().unwrap();
    let rec = crate::db::get_fit(&conn, fit_id)
        .map_err(err)?
        .ok_or((StatusCode::NOT_FOUND, format!("fit {fit_id} 不存在")))?;
    let result: crate::stats::FitResult =
        serde_json::from_value(rec["result"].clone()).map_err(|e| err(e.to_string()))?;
    let env: stats::Envelope =
        serde_json::from_value(rec["envelope"].clone()).map_err(|e| err(e.to_string()))?;
    let cfg_v = rec["config"].clone();
    let cfg: FitBody = serde_json::from_value(cfg_v).map_err(|e| err(e.to_string()))?;
    let scale: crate::models::Scale =
        serde_json::from_value(serde_json::to_value(&result.scale).unwrap())
            .map_err(|e| err(e.to_string()))?;
    let temp_max_k = result.internal.temp_max_k;
    let fo = stats::FitOutput {
        result,
        ws_scale: scale,
        raw: rec["result"]["internal"]["raw"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect(),
        cov: serde_json::from_value(rec["result"]["internal"]["cov_raw"].clone())
            .map_err(|e| err(e.to_string()))?,
        temp_max_k,
        heat_index: vec![],
        n_obs: rec["included"].as_array().map(|a| a.len()).unwrap_or(0),
    };
    Ok((fo, env, rec, cfg))
}

#[derive(Deserialize)]
struct PredictBody {
    fit_id: i64,
    temp_value: f64,
    temp_unit: String,
    stress_mpa: f64,
    #[serde(default)]
    heat: Option<String>,
    #[serde(default = "default_pi")]
    pi_z: f64,
}
fn default_pi() -> f64 {
    1.96
}

async fn predict(
    State(st): State<AppState>,
    Json(body): Json<PredictBody>,
) -> Result<Json<serde_json::Value>, AppErr> {
    let temp_c = crate::ingest::to_celsius(body.temp_value, Some(&body.temp_unit), "预测点").map_err(err)?;
    let temp_k = temp_c + 273.15;
    let (fo, env, _rec, cfg) = reconstruct(&st, body.fit_id)?;
    let kind = ModelKind::parse(&cfg.model).unwrap();
    let pred = fo.predict(
        &env,
        kind,
        temp_k,
        body.stress_mpa,
        (cfg.allow_temp_min_c + 273.15, cfg.allow_temp_max_c + 273.15),
        (cfg.allow_stress_min_mpa, cfg.allow_stress_max_mpa),
        body.heat.as_deref(),
        if body.pi_z > 0.0 { body.pi_z } else { 1.96 },
    );
    if pred.outside_allowed {
        return Ok(Json(json!({ "ok": false, "rejected": true, "prediction": pred })));
    }
    Ok(Json(json!({ "ok": true, "prediction": pred })))
}

#[derive(Deserialize)]
struct CompareQuery {
    a: i64,
    b: i64,
}

async fn compare(
    State(st): State<AppState>,
    Query(q): Query<CompareQuery>,
) -> Result<Json<serde_json::Value>, AppErr> {
    let (fo_a, _env_a, rec_a, _cfg_a) = reconstruct(&st, q.a)?;
    let (fo_b, _env_b, rec_b, _cfg_b) = reconstruct(&st, q.b)?;
    let inc_a: HashSet<String> = serde_json::from_value(rec_a["included"].clone()).unwrap_or_default();
    let inc_b: HashSet<String> = serde_json::from_value(rec_b["included"].clone()).unwrap_or_default();
    if rec_a["criterion"] != rec_b["criterion"] {
        return Err(err(
            "两个拟合使用了不同失效准则，按数据口径禁止直接比较，请分别查看",
        ));
    }
    let common: Vec<&String> = inc_a.intersection(&inc_b).collect();
    let mut only_a: Vec<&String> = inc_a.difference(&inc_b).collect();
    let mut only_b: Vec<&String> = inc_b.difference(&inc_a).collect();
    only_a.sort();
    only_b.sort();
    let ra: HashMap<String, serde_json::Value> = serde_json::from_value(
        serde_json::to_value(&fo_a.result.residuals).unwrap(),
    )
    .unwrap_or_default();
    let rb: HashMap<String, serde_json::Value> = serde_json::from_value(
        serde_json::to_value(&fo_b.result.residuals).unwrap(),
    )
    .unwrap_or_default();
    let mut rows = Vec::new();
    for id in &common {
        rows.push(json!({
            "sample_id": id,
            "model_a_residual": ra.get(*id).and_then(|v| v["residual_dex"].as_f64()),
            "model_b_residual": rb.get(*id).and_then(|v| v["residual_dex"].as_f64()),
            "model_a_nll": ra.get(*id).and_then(|v| v["nll_contribution"].as_f64()),
            "model_b_nll": rb.get(*id).and_then(|v| v["nll_contribution"].as_f64()),
        }));
    }
    rows.sort_by(|x, y| x["sample_id"].as_str().cmp(&y["sample_id"].as_str()));
    Ok(Json(json!({
        "criterion": rec_a["criterion"],
        "common_samples": common.len(),
        "only_in_a": only_a,
        "only_in_b": only_b,
        "same_sample_set": only_a.is_empty() && only_b.is_empty(),
        "fit_a": {"id": q.a, "aic": fo_a.result.aic, "bic": fo_a.result.bic, "nll": fo_a.result.nll, "sigma_dex": fo_a.result.sigma_dex, "model": fo_a.result.model},
        "fit_b": {"id": q.b, "aic": fo_b.result.aic, "bic": fo_b.result.bic, "nll": fo_b.result.nll, "sigma_dex": fo_b.result.sigma_dex, "model": fo_b.result.model},
        "rows": rows,
    })))
}

async fn reset(State(st): State<AppState>) -> impl IntoResponse {
    let conn = st.db.0.lock().unwrap();
    if let Err(e) = crate::db::wipe(&conn) {
        return Json(json!({"ok": false, "error": e.to_string()}));
    }
    crate::db::audit(&conn, "reset", "清空样本/拟合/审计记录");
    Json(json!({"ok": true}))
}

async fn export_bundle(State(st): State<AppState>) -> impl IntoResponse {
    let conn = st.db.0.lock().unwrap();
    let samples = crate::db::list_samples(&conn).unwrap_or_default();
    let fits = crate::db::list_fits(&conn).unwrap_or_default();
    let mut fit_records = Vec::new();
    for f in &fits {
        if let Some(rec) = crate::db::get_fit(&conn, f["id"].as_i64().unwrap()).unwrap_or(None) {
            fit_records.push(rec);
        }
    }
    let payload = json!({
        "format": "rupture-spectrum-bundle/v1",
        "exported_at": chrono_now(),
        "samples_raw": samples_to_import(&samples),
        "fit_specs": fit_records.iter().map(|r| r["config"].clone()).collect::<Vec<_>>(),
        "fit_records": fit_records,
    });
    Json(payload)
}

fn samples_to_import(samples: &[crate::ingest::ValidSample]) -> Vec<serde_json::Value> {
    samples
        .iter()
        .map(|s| {
            json!({
                "id": s.id, "heat": s.heat, "criterion": s.criterion, "load_kind": s.load_kind,
                "temp_value": s.temp_value, "temp_unit": s.temp_unit,
                "stress_mpa": s.stress_mpa, "time_h": s.time_h, "status": s.status,
                "segments": s.segments.iter().map(|g| json!({
                    "temp_value": g.temp_c, "temp_unit": "C",
                    "stress_mpa": g.stress_mpa, "hours_h": g.hours_h
                })).collect::<Vec<_>>(),
                "note": s.note,
            })
        })
        .collect()
}

fn chrono_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    format!("unix:{secs}")
}

#[derive(Deserialize)]
struct ReplayBody {
    #[serde(default)]
    samples_raw: Option<Vec<Sample>>,
    #[serde(default)]
    bundle: Option<serde_json::Value>,
}

async fn replay(
    State(st): State<AppState>,
    Json(body): Json<ReplayBody>,
) -> Result<Json<serde_json::Value>, AppErr> {
    // 取样本来源
    let bundle = body.bundle.clone();
    let raw_samples: Vec<Sample> = if let Some(v) = body.samples_raw {
        v
    } else if let Some(b) = bundle.as_ref() {
        serde_json::from_value(b["samples_raw"].clone()).map_err(|e| err(e.to_string()))?
    } else {
        return Err(err("replay 需要 samples_raw 或 bundle"));
    };

    let specs: Vec<FitBody> = match bundle.as_ref().and_then(|b| b.get("fit_specs")) {
        Some(v) => serde_json::from_value(v.clone()).map_err(|e| err(e.to_string()))?,
        None => vec![],
    };

    let mut seen = HashSet::new();
    let mut valid = Vec::new();
    let mut errors = Vec::new();
    for s in &raw_samples {
        match crate::ingest::validate(s, &mut seen) {
            Ok(v) => valid.push(v),
            Err(e) => errors.push(e),
        }
    }
    if !errors.is_empty() {
        return Err(err(format!("样本校验失败: {errors:?}")));
    }

    let mut refits = Vec::new();
    {
        let mut conn = st.db.0.lock().unwrap();
        crate::db::wipe(&mut conn).map_err(err)?;
        let tx = conn.transaction().map_err(err)?;
        for v in &valid {
            crate::db::insert_sample(&tx, v).map_err(err)?;
        }
        crate::db::audit(&tx, "replay", &format!("清空后重新导入 {} 个样本", valid.len()));
        tx.commit().map_err(err)?;
    }
    // 逐条复跑（perform_fit 自己开锁）
    for spec in &specs {
        let res = perform_fit(&st, spec)?;
        refits.push(json!({
            "model": spec.model,
            "criterion": spec.criterion,
            "damage_model": spec.damage_model,
            "new_fit_id": res["id"],
            "included": res["included"],
            "nll": res["result"]["nll"],
            "aic": res["result"]["aic"],
        }));
    }
    Ok(Json(json!({
        "ok": true,
        "samples": valid.len(),
        "refits": refits,
    })))
}

// 仅为保留 SegmentIn 导入在文档中的占位（编译期防止未用告警链）。
#[allow(dead_code)]
fn _seg_type(_: SegmentIn) {}
