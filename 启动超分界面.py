"""Real-ESRGAN 图像超分辨率 - 双击启动 Web UI."""
import argparse
import os
import shutil
import socket
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path


ROOT = Path(__file__).resolve().parent
APP_DIR = ROOT / "Real-ESRGAN"
FRONTEND_DIR = ROOT / "frontend"
FRONTEND_DIST_INDEX = FRONTEND_DIR / "dist" / "index.html"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7860
PORT_SCAN_COUNT = 50


def find_venv_python():
    candidates = [
        ROOT / ".venv" / "Scripts" / "python.exe",
        ROOT / ".venv" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def relaunch_in_venv() -> None:
    venv_python = find_venv_python()
    if venv_python is None:
        return

    current_python = Path(sys.executable).resolve()
    target_python = venv_python.resolve()
    if current_python == target_python:
        return

    print("使用项目虚拟环境: {}".format(target_python))
    os.execv(str(target_python), [str(target_python), str(Path(__file__).resolve()), *sys.argv[1:]])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动 Real-ESRGAN Web UI")
    parser.add_argument("--host", default=os.environ.get("GRADIO_SERVER_NAME", DEFAULT_HOST))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("GRADIO_SERVER_PORT", DEFAULT_PORT)),
        help="首选端口，默认 7860；占用时自动尝试后续端口。",
    )
    parser.add_argument("--no-browser", action="store_true", help="启动服务但不自动打开浏览器。")
    parser.add_argument("--strict-port", action="store_true", help="端口被占用时直接失败，不自动换端口。")
    parser.add_argument("--gradio", action="store_true", help="使用旧 Gradio 界面启动。")
    parser.add_argument("--skip-frontend-build", action="store_true", help="跳过 Vue 前端构建检查。")
    parser.add_argument("--rebuild-frontend", action="store_true", help="强制重新构建 Vue 前端。")
    return parser.parse_args()


def port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def pick_port(host: str, preferred_port: int, strict: bool) -> int:
    if port_is_free(host, preferred_port):
        return preferred_port
    if strict:
        raise OSError("端口 {} 已被占用。".format(preferred_port))

    for port in range(preferred_port + 1, preferred_port + PORT_SCAN_COUNT + 1):
        if port_is_free(host, port):
            print("端口 {} 已被占用，改用 {}。".format(preferred_port, port))
            return port

    raise OSError("端口 {}-{} 都不可用。".format(preferred_port, preferred_port + PORT_SCAN_COUNT))


def npm_executable():
    if os.name == "nt":
        return shutil.which("npm.cmd") or shutil.which("npm")
    return shutil.which("npm")


def frontend_source_files():
    files = [
        FRONTEND_DIR / "package.json",
        FRONTEND_DIR / "index.html",
        FRONTEND_DIR / "vite.config.js",
    ]
    src_dir = FRONTEND_DIR / "src"
    if src_dir.exists():
        files.extend(path for path in src_dir.rglob("*") if path.is_file())
    return [path for path in files if path.exists()]


def frontend_needs_build():
    if not FRONTEND_DIST_INDEX.exists():
        return True
    source_files = frontend_source_files()
    if not source_files:
        return False
    latest_source = max(path.stat().st_mtime for path in source_files)
    return latest_source > FRONTEND_DIST_INDEX.stat().st_mtime


def ensure_frontend_build(force: bool, skip: bool) -> None:
    if skip:
        return
    if not FRONTEND_DIR.exists():
        raise FileNotFoundError("找不到 Vue 前端目录: {}".format(FRONTEND_DIR))
    if not force and not frontend_needs_build():
        return

    npm = npm_executable()
    if npm is None:
        raise RuntimeError("未找到 npm，无法构建 Vue 前端。请安装 Node.js 后重试。")

    if not (FRONTEND_DIR / "node_modules").exists():
        print("首次构建 Vue 前端，正在安装 npm 依赖...")
        subprocess.run([npm, "install"], cwd=FRONTEND_DIR, check=True)

    print("正在构建 Vue 前端...")
    subprocess.run([npm, "run", "build"], cwd=FRONTEND_DIR, check=True)


def open_browser_later(url: str) -> None:
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()


def run_gradio(args: argparse.Namespace, port: int) -> None:
    os.chdir(APP_DIR)
    sys.path.insert(0, str(APP_DIR))

    import app

    print("启动 Real-ESRGAN Gradio Web UI...")
    print("访问地址: http://{}:{}".format(args.host, port))
    app.demo.launch(
        server_name=args.host,
        server_port=port,
        share=False,
        inbrowser=not args.no_browser,
        theme=app.UI_THEME,
        css=app.CUSTOM_CSS,
        footer_links=[],
    )


def run_vue(args: argparse.Namespace, port: int) -> None:
    ensure_frontend_build(args.rebuild_frontend, args.skip_frontend_build)
    os.chdir(APP_DIR)
    sys.path.insert(0, str(APP_DIR))

    import uvicorn
    import vue_server

    url = "http://{}:{}".format(args.host, port)
    print("启动 Real-ESRGAN Vue Web UI...")
    print("访问地址: {}".format(url))
    if not args.no_browser:
        open_browser_later(url)
    uvicorn.run(vue_server.api, host=args.host, port=port, log_level="info")


def main() -> None:
    relaunch_in_venv()
    args = parse_args()
    port = pick_port(args.host, args.port, args.strict_port)
    if args.gradio:
        run_gradio(args, port)
    else:
        run_vue(args, port)


if __name__ == "__main__":
    main()
