use crate::db::Db;
use crate::fit::{fit, FitRequest};
use crate::models::{DomainError, Specimen, SpecimenIn};
use axum::{
    extract::State,
    http::StatusCode,
    response::{Html, IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Clone)]
pub struct AppState {
    pub db: Arc<Db>,
}

#[derive(Deserialize)]
struct ImportBody {
    source: Option<String>,
    specimens: Vec<SpecimenIn>,
}

#[derive(Serialize)]
struct ImportReport {
    received: usize,
    imported: usize,
    skipped_duplicate: usize,
    rejected: Vec<String>,
}

#[derive(Deserialize)]
struct CompareBody {
    run_a: i64,
    run_b: i64,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/api/specimens", get(list_specimens).post(import))
        .route("/api/specimens/clear", post(clear_all))
        .route("/api/fixtures/load", post(load_fixtures))
        .route("/api/fit", post(do_fit))
        .route("/api/runs", get(list_runs))
        .route("/api/runs/export", get(export_runs))
        .route("/api/compare", post(compare))
        .route("/api/imports", get(list_imports))
        .route("/api/health", get(health))
        .with_state(state)
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "ok": true, "service": "mastercurve-bench" }))
}

async fn index() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}

fn err(status: StatusCode, msg: String) -> Response {
    (status, Json(serde_json::json!({ "error": msg }))).into_response()
}

async fn list_specimens(State(st): State<AppState>) -> Response {
    match st.db.list_specimens() {
        Ok(v) => Json(serde_json::json!({ "specimens": v })).into_response(),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

async fn list_imports(State(st): State<AppState>) -> Response {
    match st.db.list_import_batches() {
        Ok(v) => Json(serde_json::json!({ "batches": v })).into_response(),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

async fn import(State(st): State<AppState>, Json(body): Json<ImportBody>) -> Response {
    let mut imported = 0usize;
    let mut skipped = 0usize;
    let mut rejected = Vec::new();
    for item in &body.specimens {
        match item.validate() {
            Ok(s) => match st.db.insert_specimen(&s) {
                Ok(true) => imported += 1,
                Ok(false) => skipped += 1,
                Err(e) => rejected.push(format!("{}: {}", item.specimen_id, e)),
            },
            Err(DomainError(msg)) => rejected.push(msg),
        }
    }
    let source = body.source.unwrap_or_else(|| "api-import".into());
    if let Err(e) = st
        .db
        .record_import(&source, imported, skipped, &rejected)
    {
        return err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string());
    }
    Json(ImportReport {
        received: body.specimens.len(),
        imported,
        skipped_duplicate: skipped,
        rejected,
    })
    .into_response()
}

async fn clear_all(State(st): State<AppState>) -> Response {
    match st.db.clear_specimens() {
        Ok(()) => Json(serde_json::json!({ "cleared": true })).into_response(),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

pub const FIXTURES: &str = include_str!("../data/fixtures.json");

async fn load_fixtures(State(st): State<AppState>) -> Response {
    let items: Vec<SpecimenIn> = match serde_json::from_str(FIXTURES) {
        Ok(v) => v,
        Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, format!("fixture 解析失败: {e}")),
    };
    let mut imported = 0usize;
    let mut skipped = 0usize;
    let mut rejected = Vec::new();
    for item in &items {
        match item.validate() {
            Ok(s) => match st.db.insert_specimen(&s) {
                Ok(true) => imported += 1,
                Ok(false) => skipped += 1,
                Err(e) => rejected.push(e.to_string()),
            },
            Err(e) => rejected.push(e.to_string()),
        }
    }
    if let Err(e) = st.db.record_import("builtin-fixtures", imported, skipped, &rejected) {
        return err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string());
    }
    Json(serde_json::json!({
        "source": "builtin-fixtures",
        "received": items.len(),
        "imported": imported,
        "skipped_duplicate": skipped,
        "rejected": rejected,
    }))
    .into_response()
}

async fn do_fit(State(st): State<AppState>, Json(req): Json<FitRequest>) -> Response {
    let specimens = match st.db.list_specimens() {
        Ok(v) => v,
        Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };
    let result = match fit(&specimens, &req) {
        Ok(r) => r,
        Err(DomainError(msg)) => return err(StatusCode::BAD_REQUEST, msg),
    };
    let req_json = serde_json::to_value(&req).unwrap_or(serde_json::json!({}));
    let res_json = serde_json::to_value(&result).unwrap_or(serde_json::json!({}));
    let run_id = match st.db.save_run(req.name.as_deref(), &req_json, &res_json) {
        Ok(id) => id,
        Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };
    Json(serde_json::json!({ "run_id": run_id, "result": result })).into_response()
}

async fn list_runs(State(st): State<AppState>) -> Response {
    match st.db.list_runs() {
        Ok(v) => Json(serde_json::json!({ "runs": v })).into_response(),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

async fn export_runs(State(st): State<AppState>) -> Response {
    let specimens: Vec<Specimen> = st.db.list_specimens().unwrap_or_default();
    let runs = match st.db.list_runs() {
        Ok(v) => v,
        Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };
    let batches = st.db.list_import_batches().unwrap_or_default();
    let payload = serde_json::json!({
        "export": "mastercurve-bench 运行记录",
        "specimens": specimens,
        "import_batches": batches,
        "runs": runs,
    });
    match serde_json::to_string_pretty(&payload) {
        Ok(s) => (
            StatusCode::OK,
            [("content-type", "application/json; charset=utf-8")],
            s,
        )
            .into_response(),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

async fn compare(State(st): State<AppState>, Json(body): Json<CompareBody>) -> Response {
    let a = match st.db.get_run_result(body.run_a) {
        Ok(Some(v)) => v,
        Ok(None) => return err(StatusCode::NOT_FOUND, format!("运行 {} 不存在", body.run_a)),
        Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };
    let b = match st.db.get_run_result(body.run_b) {
        Ok(Some(v)) => v,
        Ok(None) => return err(StatusCode::NOT_FOUND, format!("运行 {} 不存在", body.run_b)),
        Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };
    let ca = a.get("criterion").and_then(|v| v.as_str()).unwrap_or("");
    let cb = b.get("criterion").and_then(|v| v.as_str()).unwrap_or("");
    if ca != cb {
        return err(
            StatusCode::BAD_REQUEST,
            format!("两次运行失效准则不同（{ca} vs {cb}），不允许在同一样本组上比较"),
        );
    }
    let ida: std::collections::BTreeSet<String> = a
        .get("residuals")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|r| r.get("specimen_id").and_then(|x| x.as_str()).map(String::from))
                .collect()
        })
        .unwrap_or_default();
    let idb: std::collections::BTreeSet<String> = b
        .get("residuals")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|r| r.get("specimen_id").and_then(|x| x.as_str()).map(String::from))
                .collect()
        })
        .unwrap_or_default();
    if ida != idb {
        return err(
            StatusCode::BAD_REQUEST,
            format!(
                "两次运行纳入样本集合不同（仅在 A: {} / 仅在 B: {}）；请对齐损伤模型开关后再比较",
                ida.difference(&idb).cloned().collect::<Vec<_>>().join(","),
                idb.difference(&ida).cloned().collect::<Vec<_>>().join(",")
            ),
        );
    }
    let aic_a = a.get("aic").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let aic_b = b.get("aic").and_then(|v| v.as_f64()).unwrap_or(0.0);
    Json(serde_json::json!({
        "same_criterion": ca,
        "same_samples": ida.len(),
        "run_a": { "id": body.run_a, "model": a.get("model"), "aic": aic_a,
                   "damage_model": a.get("damage_model"), "tau_log10h": a.get("tau_log10h") },
        "run_b": { "id": body.run_b, "model": b.get("model"), "aic": aic_b,
                   "damage_model": b.get("damage_model"), "tau_log10h": b.get("tau_log10h") },
        "delta_aic": (aic_a - aic_b),
        "preferred_by_aic": if aic_a <= aic_b { body.run_a } else { body.run_b },
        "note": "两个模型在完全相同的样本集合上比较；不同失效准则的试验从未混为一组",
    }))
    .into_response()
}
