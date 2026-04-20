pub mod backend;
pub use backend::{Backend, CpuBackend};

use fern::colors::{Color, ColoredLevelConfig};
use std::fs;

/// 初始化日志：终端彩色输出 + 写入 logs/<name>.log
/// 日志级别由环境变量 RUST_LOG 控制，默认 info
pub fn init_logger(name: &str) {
    fs::create_dir_all("logs").expect("无法创建 logs 目录");

    let log_file =
        fern::log_file(format!("logs/{}.log", name)).expect("无法创建日志文件");

    let colors = ColoredLevelConfig::new()
        .error(Color::Red)
        .warn(Color::Yellow)
        .info(Color::Green)
        .debug(Color::Cyan)
        .trace(Color::White);

    let level = std::env::var("RUST_LOG")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(log::LevelFilter::Info);

    let terminal = fern::Dispatch::new()
        .format(move |out, message, record| {
            out.finish(format_args!(
                "[{} {:<5} {}] {}",
                chrono::Local::now().format("%H:%M:%S"),
                colors.color(record.level()),
                record.target(),
                message
            ))
        })
        .level(level)
        .chain(std::io::stderr());

    let file = fern::Dispatch::new()
        .format(|out, message, record| {
            out.finish(format_args!(
                "[{} {:<5} {}] {}",
                chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
                record.level(),
                record.target(),
                message
            ))
        })
        .level(level)
        .chain(log_file);

    fern::Dispatch::new()
        .chain(terminal)
        .chain(file)
        .apply()
        .expect("日志初始化失败");
}
