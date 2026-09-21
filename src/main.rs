mod db;
mod ingest;
mod models;
mod server;
mod stats;

use db::Db;
use std::collections::HashSet;
use std::sync::Arc;

const FIXTURE_JSON: &str = include_str!("../fixtures/samples.json");

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut listen = "127.0.0.1:5512".to_string();
    let mut db_path = "rupture.db".to_string();
    let mut seed = true;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--listen" => {
                i += 1;
                listen = args[i].clone();
            }
            "--db" => {
                i += 1;
                db_path = args[i].clone();
            }
            "--no-seed" => seed = false,
            "--help" | "-h" => {
                println!(
                    "持时破裂谱 — 蠕变持久断裂寿命主曲线工作台\n\
                     用法: rupture-spectrum [--listen 127.0.0.1:5512] [--db rupture.db] [--no-seed]"
                );
                return;
            }
            other => panic!("未知参数 {other}"),
        }
        i += 1;
    }

    let mut conn = db::init(&db_path).expect("打开/初始化 SQLite 失败");
    if seed && db::count_samples(&conn).unwrap_or(0) == 0 {
        seed_fixture(&mut conn);
    }
    let state = server::AppState {
        db: Arc::new(Db(std::sync::Mutex::new(conn))),
    };
    let app = server::router(state);
    let listener = tokio::net::TcpListener::bind(&listen).await.expect("绑定端口失败");
    println!("持时破裂谱 已启动: http://{listen}  (db={db_path})");
    axum::serve(listener, app).await.unwrap();
}

fn seed_fixture(conn: &mut rusqlite::Connection) {
    #[derive(serde::Deserialize)]
    struct Wrap {
        samples: Vec<ingest::Sample>,
    }
    let wrap: Wrap = serde_json::from_str(FIXTURE_JSON).expect("内置 fixture 解析失败");
    let mut seen = HashSet::new();
    let tx = conn.transaction().expect("fixture 事务失败");
    let mut n = 0;
    for s in &wrap.samples {
        match ingest::validate(s, &mut seen) {
            Ok(v) => {
                db::insert_sample(&tx, &v).expect("fixture 入库失败");
                n += 1;
            }
            Err(e) => panic!("内置 fixture 数据非法: {e}"),
        }
    }
    db::audit(&tx, "seed", &format!("空库自动种入固定 fixture：{n} 个样本"));
    tx.commit().expect("fixture 提交失败");
}
