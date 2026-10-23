//! Patine desktop shell.
//!
//! The shell does three things: host the UI, own the Python sidecar's
//! lifecycle, and open native dialogs. Everything else lives in the sidecar
//! (see docs/ARCHITECTURE.md).

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Mutex;

use tauri::{AppHandle, Emitter, Manager, RunEvent, State};
use tauri_plugin_dialog::DialogExt;
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;

const READY_PREFIX: &str = "PATINE_READY ";
const LOG_TAIL: usize = 200;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SidecarInfo {
    pub url: String,
    pub token: String,
}

#[derive(Default)]
pub struct SidecarState {
    info: Mutex<Option<SidecarInfo>>,
    child: Mutex<Option<CommandChild>>,
    log: Mutex<VecDeque<String>>,
    error: Mutex<Option<String>>,
}

impl SidecarState {
    fn push_log(&self, line: String) {
        let mut log = self.log.lock().unwrap();
        if log.len() >= LOG_TAIL {
            log.pop_front();
        }
        log.push_back(line);
    }
}

/// The UI polls this until it returns `Some`.
#[tauri::command]
fn sidecar_info(state: State<SidecarState>) -> Option<SidecarInfo> {
    state.info.lock().unwrap().clone()
}

#[derive(serde::Serialize)]
struct SidecarStatus {
    info: Option<SidecarInfo>,
    error: Option<String>,
    log: Vec<String>,
}

/// Status line + recent sidecar log, for the Transport pane's engine readout.
#[tauri::command]
fn sidecar_status(state: State<SidecarState>) -> SidecarStatus {
    SidecarStatus {
        info: state.info.lock().unwrap().clone(),
        error: state.error.lock().unwrap().clone(),
        log: state.log.lock().unwrap().iter().cloned().collect(),
    }
}

#[tauri::command]
async fn pick_audio_files(app: AppHandle) -> Vec<String> {
    let picked = app
        .dialog()
        .file()
        .add_filter(
            "Audio",
            &["wav", "aif", "aiff", "flac", "ogg", "mp3", "m4a", "aac", "opus", "caf"],
        )
        .blocking_pick_files();
    picked
        .unwrap_or_default()
        .into_iter()
        .filter_map(|p| p.into_path().ok())
        .map(|p| p.to_string_lossy().to_string())
        .collect()
}

#[tauri::command]
async fn pick_audio_folder(app: AppHandle) -> Option<String> {
    app.dialog()
        .file()
        .blocking_pick_folder()
        .and_then(|p| p.into_path().ok())
        .map(|p| p.to_string_lossy().to_string())
}

#[tauri::command]
async fn pick_directory(app: AppHandle) -> Option<String> {
    app.dialog()
        .file()
        .blocking_pick_folder()
        .and_then(|p| p.into_path().ok())
        .map(|p| p.to_string_lossy().to_string())
}

fn data_dir(app: &AppHandle) -> PathBuf {
    app.path()
        .app_data_dir()
        .unwrap_or_else(|_| std::env::temp_dir().join("patine"))
}

/// Spawn the sidecar, or attach to one named by `PATINE_SIDECAR_URL`.
fn start_sidecar(app: AppHandle) -> Result<(), String> {
    if let Ok(url) = std::env::var("PATINE_SIDECAR_URL") {
        let token = std::env::var("PATINE_SIDECAR_TOKEN").unwrap_or_else(|_| "dev".into());
        let info = SidecarInfo { url, token };
        let state = app.state::<SidecarState>();
        *state.info.lock().unwrap() = Some(info.clone());
        state.push_log("attached to external sidecar".into());
        app.emit("sidecar://ready", info).map_err(|e| e.to_string())?;
        return Ok(());
    }

    let data_dir = data_dir(&app);
    std::fs::create_dir_all(&data_dir).ok();

    let command = if cfg!(debug_assertions) {
        // Dev: run from source with uv. The repo root is three levels above
        // this crate (apps/desktop/src-tauri).
        let sidecar_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../sidecar")
            .canonicalize()
            .unwrap_or_else(|_| PathBuf::from("sidecar"));
        app.shell()
            .command("uv")
            .args([
                "run",
                "--project",
                &sidecar_dir.to_string_lossy(),
                "patine-sidecar",
                "--dev",
            ])
            .current_dir(sidecar_dir)
    } else {
        app.shell()
            .sidecar("patine-sidecar")
            .map_err(|e| format!("sidecar binary not found: {e}"))?
            .args(["--data-dir", &data_dir.to_string_lossy()])
    };

    let (mut rx, child) = command.spawn().map_err(|e| format!("spawn failed: {e}"))?;
    {
        let state = app.state::<SidecarState>();
        *state.child.lock().unwrap() = Some(child);
        state.push_log("sidecar spawned".into());
    }

    let handle = app.clone();
    tauri::async_runtime::spawn(async move {
        while let Some(event) = rx.recv().await {
            let state = handle.state::<SidecarState>();
            match event {
                CommandEvent::Stdout(bytes) => {
                    let line = String::from_utf8_lossy(&bytes).trim_end().to_string();
                    if let Some(json) = line.strip_prefix(READY_PREFIX) {
                        match serde_json::from_str::<SidecarInfo>(json) {
                            Ok(info) => {
                                log::info!("sidecar ready at {}", info.url);
                                *state.info.lock().unwrap() = Some(info.clone());
                                let _ = handle.emit("sidecar://ready", info);
                            }
                            Err(e) => {
                                state.push_log(format!("bad ready line: {e}"));
                            }
                        }
                    } else if !line.is_empty() {
                        state.push_log(line);
                    }
                }
                CommandEvent::Stderr(bytes) => {
                    let line = String::from_utf8_lossy(&bytes).trim_end().to_string();
                    if !line.is_empty() {
                        log::debug!("sidecar: {line}");
                        state.push_log(line);
                    }
                }
                CommandEvent::Error(e) => {
                    log::error!("sidecar error: {e}");
                    *state.error.lock().unwrap() = Some(e.clone());
                    state.push_log(format!("error: {e}"));
                }
                CommandEvent::Terminated(payload) => {
                    let msg = format!("sidecar exited (code {:?}, signal {:?})", payload.code, payload.signal);
                    log::warn!("{msg}");
                    *state.info.lock().unwrap() = None;
                    *state.error.lock().unwrap() = Some(msg.clone());
                    state.push_log(msg.clone());
                    let _ = handle.emit("sidecar://exited", msg);
                }
                _ => {}
            }
        }
    });
    Ok(())
}

fn stop_sidecar(app: &AppHandle) {
    let state = app.state::<SidecarState>();
    let child = state.child.lock().unwrap().take();
    if let Some(child) = child {
        let _ = child.kill();
    }
}

pub fn run() {
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).try_init();

    let app = tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .manage(SidecarState::default())
        .invoke_handler(tauri::generate_handler![
            sidecar_info,
            sidecar_status,
            pick_audio_files,
            pick_audio_folder,
            pick_directory
        ])
        .setup(|app| {
            if let Err(e) = start_sidecar(app.handle().clone()) {
                log::error!("could not start sidecar: {e}");
                let state = app.state::<SidecarState>();
                *state.error.lock().unwrap() = Some(format!("could not start sidecar: {e}"));
            }
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building Patine");

    app.run(|app, event| {
        if matches!(event, RunEvent::ExitRequested { .. } | RunEvent::Exit) {
            stop_sidecar(app);
        }
    });
}
