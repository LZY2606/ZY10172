use crate::ingest::ValidSample;
use rusqlite::{params, Connection};
use std::sync::Mutex;

pub struct Db(pub Mutex<Connection>);

pub fn init(path: &str) -> rusqlite::Result<Connection> {
    let conn = Connection::open(path)?;
    conn.execute_batch(
        r#"
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS samples (
            id TEXT PRIMARY KEY,
            heat TEXT NOT NULL,
            criterion TEXT NOT NULL,
            load_kind TEXT NOT NULL,
            temp_c REAL,
            temp_k REAL,
            temp_unit TEXT,
            temp_value REAL,
            stress_mpa REAL,
            time_h REAL NOT NULL,
            status TEXT NOT NULL,
            segments_json TEXT NOT NULL DEFAULT '[]',
            note TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS fits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            model TEXT NOT NULL,
            criterion TEXT NOT NULL,
            damage_model INTEGER NOT NULL,
            config_json TEXT NOT NULL,
            included_json TEXT NOT NULL,
            excluded_json TEXT NOT NULL,
            result_json TEXT NOT NULL,
            envelope_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            at TEXT NOT NULL DEFAULT (datetime('now')),
            action TEXT NOT NULL,
            detail TEXT NOT NULL DEFAULT ''
        );
        "#,
    )?;
    Ok(conn)
}

pub fn insert_sample(conn: &Connection, s: &ValidSample) -> rusqlite::Result<()> {
    conn.execute(
        r#"INSERT OR REPLACE INTO samples
           (id, heat, criterion, load_kind, temp_c, temp_k, temp_unit, temp_value,
            stress_mpa, time_h, status, segments_json, note)
           VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13)"#,
        params![
            s.id,
            s.heat,
            s.criterion,
            s.load_kind,
            s.temp_c,
            s.temp_k,
            s.temp_unit,
            s.temp_value,
            s.stress_mpa,
            s.time_h,
            s.status,
            serde_json::to_string(&s.segments).unwrap_or_default(),
            s.note
        ],
    )?;
    Ok(())
}

pub fn list_samples(conn: &Connection) -> rusqlite::Result<Vec<ValidSample>> {
    let mut stmt = conn.prepare(
        "SELECT id, heat, criterion, load_kind, temp_c, temp_k, temp_unit, temp_value,
                stress_mpa, time_h, status, segments_json, note FROM samples ORDER BY id",
    )?;
    let rows = stmt.query_map([], |r| {
        let segs_json: String = r.get(11)?;
        Ok(ValidSample {
            id: r.get(0)?,
            heat: r.get(1)?,
            criterion: r.get(2)?,
            load_kind: r.get(3)?,
            temp_c: r.get(4)?,
            temp_k: r.get(5)?,
            temp_unit: r.get(6)?,
            temp_value: r.get(7)?,
            stress_mpa: r.get(8)?,
            time_h: r.get(9)?,
            status: r.get(10)?,
            segments: serde_json::from_str(&segs_json).unwrap_or_default(),
            note: r.get(12)?,
        })
    })?;
    let mut out = Vec::new();
    for x in rows {
        out.push(x?);
    }
    Ok(out)
}

pub fn count_samples(conn: &Connection) -> rusqlite::Result<i64> {
    conn.query_row("SELECT COUNT(*) FROM samples", [], |r| r.get(0))
}

pub fn wipe(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute_batch(
        "DELETE FROM samples; DELETE FROM fits; DELETE FROM audit_log;",
    )?;
    Ok(())
}

pub fn audit(conn: &Connection, action: &str, detail: &str) {
    let _ = conn.execute(
        "INSERT INTO audit_log (action, detail) VALUES (?1, ?2)",
        params![action, detail],
    );
}

pub fn list_audit(conn: &Connection) -> rusqlite::Result<Vec<(i64, String, String)>> {
    let mut stmt = conn.prepare("SELECT id, action, at FROM audit_log ORDER BY id DESC LIMIT 200")?;
    let rows = stmt.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?)))?;
    Ok(rows.filter_map(|x| x.ok()).collect())
}

pub fn insert_fit(
    conn: &Connection,
    model: &str,
    criterion: &str,
    damage_model: bool,
    config_json: &str,
    included_json: &str,
    excluded_json: &str,
    result_json: &str,
    envelope_json: &str,
) -> rusqlite::Result<i64> {
    conn.execute(
        r#"INSERT INTO fits (model, criterion, damage_model, config_json, included_json,
                             excluded_json, result_json, envelope_json)
           VALUES (?1,?2,?3,?4,?5,?6,?7,?8)"#,
        params![
            model,
            criterion,
            damage_model as i64,
            config_json,
            included_json,
            excluded_json,
            result_json,
            envelope_json
        ],
    )?;
    Ok(conn.last_insert_rowid())
}

pub fn get_fit(conn: &Connection, id: i64) -> rusqlite::Result<Option<serde_json::Value>> {
    let mut stmt = conn.prepare(
        "SELECT model, criterion, damage_model, config_json, included_json,
                excluded_json, result_json, envelope_json, created_at FROM fits WHERE id=?1",
    )?;
    let mut rows = stmt.query_map(params![id], |r| {
        let cfg: String = r.get(3)?;
        let inc: String = r.get(4)?;
        let exc: String = r.get(5)?;
        let res: String = r.get(6)?;
        let env: String = r.get(7)?;
        Ok(serde_json::json!({
            "id": id,
            "created_at": r.get::<_, String>(8)?,
            "model": r.get::<_, String>(0)?,
            "criterion": r.get::<_, String>(1)?,
            "damage_model": r.get::<_, i64>(2)? != 0,
            "config": serde_json::from_str::<serde_json::Value>(&cfg).unwrap_or(serde_json::Value::Null),
            "included": serde_json::from_str::<serde_json::Value>(&inc).unwrap_or(serde_json::Value::Null),
            "excluded": serde_json::from_str::<serde_json::Value>(&exc).unwrap_or(serde_json::Value::Null),
            "result": serde_json::from_str::<serde_json::Value>(&res).unwrap_or(serde_json::Value::Null),
            "envelope": serde_json::from_str::<serde_json::Value>(&env).unwrap_or(serde_json::Value::Null),
        }))
    })?;
    if let Some(x) = rows.next() {
        Ok(Some(x?))
    } else {
        Ok(None)
    }
}

pub fn list_fits(conn: &Connection) -> rusqlite::Result<Vec<serde_json::Value>> {
    let mut stmt = conn.prepare(
        "SELECT id, created_at, model, criterion, damage_model FROM fits ORDER BY id DESC LIMIT 100",
    )?;
    let rows = stmt.query_map([], |r| {
        Ok(serde_json::json!({
            "id": r.get::<_, i64>(0)?,
            "created_at": r.get::<_, String>(1)?,
            "model": r.get::<_, String>(2)?,
            "criterion": r.get::<_, String>(3)?,
            "damage_model": r.get::<_, i64>(4)? != 0,
        }))
    })?;
    Ok(rows.filter_map(|x| x.ok()).collect())
}
