pub mod db;
pub mod fit;
pub mod models;
pub mod server;

pub mod cli {
use super::{models, server};
use super::db::{open, Db};
use super::server::{router, AppState};
use std::sync::Arc;

#[derive(Default)]
struct Args {
    listen: String,
    db_path: String,
}

fn parse_args() -> Args {
    let mut args = Args {
        listen: "127.0.0.1:5512".into(),
        db_path: "mastercurve.db".into(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--listen" => {
                if let Some(v) = it.next() {
                    args.listen = v;
                }
            }
            "--db" => {
                if let Some(v) = it.next() {
                    args.db_path = v;
                }
            }
            "--help" | "-h" => {
                println!(
                    "持时破裂谱 - 蠕变持久主曲线工作台\n\
                     USAGE: mastercurve-bench [--listen 127.0.0.1:5512] [--db mastercurve.db]"
                );
                std::process::exit(0);
            }
            other => {
                eprintln!("忽略未知参数: {other}");
            }
        }
    }
    args
}

#[tokio::main]
pub async fn run() {
    let args = parse_args();
    // 必须在 open() 建库文件之前判断，用于区分“首次启动”与“清空后的重放”
    let fresh_db = !std::path::Path::new(&args.db_path).exists();
    let conn = open(&args.db_path).expect("打开 SQLite 失败");
    let db = Arc::new(Db(std::sync::Mutex::new(conn)));
    let state = AppState { db: db.clone() };

    // 仅首次启动自动灌入固定 fixture；清空后不自动重灌，
    // 需显式调用 /api/fixtures/load 以复核“清空→重新导入”重放流程
    if fresh_db && db.count_specimens().unwrap_or(0) == 0 {
        if let Ok(items) = serde_json::from_str::<Vec<models::SpecimenIn>>(server::FIXTURES) {
            let mut imported = 0;
            for item in &items {
                if let Ok(s) = item.validate() {
                    if db.insert_specimen(&s).unwrap_or(false) {
                        imported += 1;
                    }
                }
            }
            let _ = db.record_import("builtin-fixtures-autoload", imported, 0, &[]);
            println!("空数据库：已自动导入 {imported} 个固定 fixture 试样");
        }
    }

    let listener = tokio::net::TcpListener::bind(&args.listen)
        .await
        .expect("监听端口失败");
    println!("持时破裂谱 已启动: http://{}/", args.listen);
    axum::serve(listener, router(state)).await.expect("服务异常");
}

}

pub mod testkit {
    use crate::{db::open, server::{router, AppState}};
    use axum::Router;
    use std::sync::Arc;

    pub struct AppHandle {
        pub router: Router,
        _tmp: tempfile::TempDir,
    }

    pub async fn spawn_app() -> AppHandle {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("test.db");
        let conn = open(path.to_str().unwrap()).expect("open db");
        let db = Arc::new(crate::db::Db(std::sync::Mutex::new(conn)));
        AppHandle {
            router: router(AppState { db }),
            _tmp: tmp,
        }
    }
}
