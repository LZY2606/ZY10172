use crate::models::*;
use rusqlite::Connection;
use std::sync::Mutex;

pub struct Db(pub Mutex<Connection>);

pub const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS specimens (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    specimen_id TEXT NOT NULL UNIQUE,
    heat TEXT NOT NULL,
    temp_kelvin REAL NOT NULL,
    stress_mpa REAL,
    time_h REAL NOT NULL,
    outcome TEXT NOT NULL,
    criterion TEXT NOT NULL,
    variable INTEGER NOT NULL DEFAULT 0,
    segments_json TEXT NOT NULL DEFAULT '[]',
    note TEXT
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    name TEXT,
    request_json TEXT NOT NULL,
    result_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS import_batches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    source TEXT NOT NULL,
    imported INTEGER NOT NULL,
    skipped INTEGER NOT NULL,
    errors_json TEXT NOT NULL DEFAULT '[]'
);
"#;

pub fn open(path: &str) -> DResult<Connection> {
    let conn = Connection::open(path)?;
    conn.execute_batch(
        "PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON; PRAGMA busy_timeout=5000;",
    )?;
    conn.execute_batch(SCHEMA)?;
    Ok(conn)
}

fn now_ts() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("{secs}")
}

impl Db {
    pub fn count_specimens(&self) -> DResult<i64> {
        let conn = self.0.lock().unwrap();
        Ok(conn.query_row("SELECT COUNT(*) FROM specimens", [], |r| r.get(0))?)
    }

    /// 插入试样；重复编号跳过而非报错，保证固定 fixture 可重放
    pub fn insert_specimen(&self, s: &Specimen) -> DResult<bool> {
        let conn = self.0.lock().unwrap();
        let seg_json = serde_json::to_string(&s.segments)?;
        let n = conn.execute(
            "INSERT OR IGNORE INTO specimens
             (specimen_id, heat, temp_kelvin, stress_mpa, time_h, outcome, criterion, variable, segments_json, note)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)",
            rusqlite::params![
                s.specimen_id,
                s.heat,
                s.temp_kelvin,
                s.stress_mpa,
                s.time_h,
                s.outcome,
                s.criterion,
                s.variable as i64,
                seg_json,
                s.note,
            ],
        )?;
        Ok(n > 0)
    }

    pub fn clear_specimens(&self) -> DResult<()> {
        let mut conn = self.0.lock().unwrap();
        let tx = conn.transaction()?;
        tx.execute("DELETE FROM specimens", [])?;
        tx.execute("DELETE FROM import_batches", [])?;
        tx.execute("DELETE FROM sqlite_sequence WHERE name='specimens'", [])?;
        tx.commit()?;
        Ok(())
    }

    pub fn list_specimens(&self) -> DResult<Vec<Specimen>> {
        let conn = self.0.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, specimen_id, heat, temp_kelvin, stress_mpa, time_h, outcome,
                    criterion, variable, segments_json, note
             FROM specimens ORDER BY criterion, specimen_id",
        )?;
        let rows = stmt.query_map([], row_to_specimen)?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    pub fn record_import(&self, source: &str, imported: usize, skipped: usize, errors: &[String]) -> DResult<()> {
        let conn = self.0.lock().unwrap();
        conn.execute(
            "INSERT INTO import_batches (created_at, source, imported, skipped, errors_json)
             VALUES (?1,?2,?3,?4,?5)",
            rusqlite::params![
                now_ts(),
                source,
                imported as i64,
                skipped as i64,
                serde_json::to_string(errors)?,
            ],
        )?;
        Ok(())
    }

    pub fn list_import_batches(&self) -> DResult<Vec<serde_json::Value>> {
        let conn = self.0.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, created_at, source, imported, skipped, errors_json
             FROM import_batches ORDER BY id",
        )?;
        let rows = stmt.query_map([], |r| {
            let errors: String = r.get(5)?;
            Ok(serde_json::json!({
                "id": r.get::<_,i64>(0)?,
                "created_at_unix": r.get::<_,String>(1)?,
                "source": r.get::<_,String>(2)?,
                "imported": r.get::<_,i64>(3)?,
                "skipped": r.get::<_,i64>(4)?,
                "errors": serde_json::from_str::<serde_json::Value>(&errors).unwrap_or(serde_json::json!([])),
            }))
        })?;
        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    pub fn save_run(&self, name: Option<&str>, req: &serde_json::Value, res: &serde_json::Value) -> DResult<i64> {
        let conn = self.0.lock().unwrap();
        conn.execute(
            "INSERT INTO runs (created_at, name, request_json, result_json) VALUES (?1,?2,?3,?4)",
            rusqlite::params![
                now_ts(),
                name,
                serde_json::to_string(req)?,
                serde_json::to_string(res)?
            ],
        )?;
        Ok(conn.last_insert_rowid())
    }

    pub fn list_runs(&self) -> DResult<Vec<serde_json::Value>> {
        let conn = self.0.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, created_at, name, request_json, result_json FROM runs ORDER BY id",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(serde_json::json!({
                "id": r.get::<_,i64>(0)?,
                "created_at_unix": r.get::<_,String>(1)?,
                "name": r.get::<_,Option<String>>(2)?,
                "request": serde_json::from_str::<serde_json::Value>(&r.get::<_,String>(3)?).ok(),
                "result": serde_json::from_str::<serde_json::Value>(&r.get::<_,String>(4)?).ok(),
            }))
        })?;
        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    pub fn get_run_result(&self, id: i64) -> DResult<Option<serde_json::Value>> {
        let conn = self.0.lock().unwrap();
        let mut stmt = conn.prepare("SELECT result_json FROM runs WHERE id=?1")?;
        let mut rows = stmt.query_map([id], |r| r.get::<_, String>(0))?;
        match rows.next() {
            Some(s) => Ok(Some(serde_json::from_str(&s?)?)),
            None => Ok(None),
        }
    }
}

fn row_to_specimen(r: &rusqlite::Row<'_>) -> rusqlite::Result<Specimen> {
    let seg_json: String = r.get(9)?;
    Ok(Specimen {
        id: r.get(0)?,
        specimen_id: r.get(1)?,
        heat: r.get(2)?,
        temp_kelvin: r.get(3)?,
        temp_c: r.get::<_, f64>(3)? - 273.15,
        stress_mpa: r.get(4)?,
        time_h: r.get(5)?,
        outcome: r.get(6)?,
        criterion: r.get(7)?,
        variable: r.get::<_, i64>(8)? != 0,
        segments: serde_json::from_str(&seg_json).unwrap_or_default(),
        note: r.get(10)?,
    })
}
