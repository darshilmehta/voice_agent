#!/usr/bin/env python3
"""Start everything poc_gibberlink needs on a Mac, in order, with one command.

    python3 start.py            # start (or reuse) Ollama, Docker + Qdrant, backend, frontend; open Chrome
    python3 start.py status     # what is running
    python3 start.py stop       # stop everything it started: backend, frontend, SearXNG, Qdrant (Ollama keeps running)

Steps (each one is skipped when it is already done):
  1. Ollama: start it if needed, with its prompt cache capped (LLAMA_ARG_CACHE_RAM, docs/DESIGN.md §8) so a 16 GB
     Mac doesn't swap; pull the chat model if it is missing.
  2. Docker Desktop, then the Qdrant container (`docker compose ... up -d qdrant`), and SearXNG when the config turns
     live web search on (tools.web_search.enabled with the local searxng provider).
  3. Model weights: offers to run scripts/setup/download_models.sh when some are missing.
  4. Backend: `uv sync --group ml`, `uv run python -m app`, waits until /health says preload: ready.
  5. Frontend: `npm install` when package-lock.json changed, `npm run dev`.
  6. Opens http://localhost:3000 in Chrome.

The backend and frontend run as children of this script: their output is shown here (and written to data/logs/),
and Ctrl+C (or closing the terminal) stops both. Qdrant, SearXNG and Ollama keep running; `python3 start.py stop` stops
everything but Ollama, from any terminal, including a backend or frontend left running by a start.py that is gone.
The config read is the backend's: $APP_CONFIG_FILE, else config/local.config.json. Standard library only.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import plistlib
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND_DIR = ROOT / "backend"
FRONTEND_DIR = ROOT / "frontend"
COMPOSE = ["docker", "compose", "-f", str(ROOT / "infra" / "docker-compose.yml")]
LOG_DIR = ROOT / "data" / "logs"
PID_FILE = ROOT / "data" / "run" / "start.pid"

OLLAMA_URL = "http://127.0.0.1:11434"
QDRANT_URL = "http://127.0.0.1:6333"
BACKEND_URL = "http://127.0.0.1:8000"
FRONTEND_URL = "http://127.0.0.1:3000"
APP_URL = "http://localhost:3000"  # the backend's CORS list allows localhost:3000

CACHE_RAM_VAR = "LLAMA_ARG_CACHE_RAM"
CACHE_RAM_MB = "1024"
HF_HUB = ROOT / "data" / "models" / "huggingface" / "hub"
REQUIRED_WEIGHTS = [  # what the local config loads (config/local.config.json)
    "models--BAAI--bge-m3",
    "models--BAAI--bge-reranker-v2-m3",
    "models--mlx-community--whisper-small-mlx",
    "models--hexgrad--Kokoro-82M",
]
DOCLING_DIR = ROOT / "data" / "models" / "docling"

USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def say(text: str, end: str = "\n") -> None:
    """print, but never fail: after the terminal closes (SIGHUP) stdout is gone and the shutdown must still run."""
    with contextlib.suppress(OSError, ValueError):
        print(text, end=end, flush=True)


def step(text: str) -> None:
    say(_c("1;36", f"==> {text}"))


def ok(text: str) -> None:
    say(_c("32", f"    ✓ {text}"))


def warn(text: str) -> None:
    say(_c("33", f"    ! {text}"))


def fail(text: str) -> None:
    say(_c("31", f"    ✗ {text}"))
    sys.exit(1)


def run(
    cmd: list[str], cwd: Path | None = None, check: bool = True, quiet: bool = False
) -> subprocess.CompletedProcess:
    if not quiet:
        print(_c("2", f"    $ {' '.join(cmd)}"), flush=True)
    return subprocess.run(cmd, cwd=cwd, check=check, text=True, capture_output=quiet)


def http_json(url: str, timeout: float = 2.0) -> dict | list | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode() or "null")
    except (urllib.error.URLError, OSError, ValueError):
        return None


def http_up(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout):
            return True
    except urllib.error.HTTPError:
        return True  # it answered
    except (urllib.error.URLError, OSError):
        return False


def wait_for(check, what: str, timeout_s: float, interval_s: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if check():
            return True
        time.sleep(interval_s)
    warn(f"timed out after {timeout_s:.0f} s waiting for {what}")
    return False


def need(tool: str, hint: str) -> str:
    path = shutil.which(tool)
    if path is None:
        fail(f"`{tool}` not found. {hint}")
    return path


def ask(question: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        return False
    return input(f"    ? {question} [Y/n] ").strip().lower() in ("", "y", "yes")


# ---------------------------------------------------------------- Ollama


def load_config() -> dict:
    """The config file the backend will load (same rule as backend/app/settings.py)."""
    path = Path(os.environ.get("APP_CONFIG_FILE") or "config/local.config.json").expanduser()
    if not path.is_absolute():
        path = ROOT / path
    try:
        cfg = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        fail(f"can't read the config {path}: {e}")
    return cfg if isinstance(cfg, dict) else {}


def chat_model(cfg: dict) -> str:
    return (cfg.get("llm") or {}).get("chat_model") or "qwen3:4b-instruct"


def searxng_url(cfg: dict) -> str | None:
    """The local SearXNG URL when the config turns live web search on with the self-hosted provider, else None."""
    ws = (cfg.get("tools") or {}).get("web_search") or {}
    if not ws.get("enabled") or ws.get("provider") != "searxng":
        return None
    url = (ws.get("url") or "http://127.0.0.1:8888").rstrip("/")
    host = urllib.parse.urlparse(url).hostname or ""
    return url if host in ("127.0.0.1", "localhost", "::1") else None  # a remote SearXNG isn't ours to start


def brew_ollama_service() -> bool:
    """True when `brew services` manages Ollama (the README's setup)."""
    if shutil.which("brew") is None:
        return False
    done = run(["brew", "services", "list"], check=False, quiet=True)
    return any(line.split()[:1] == ["ollama"] for line in done.stdout.splitlines())


def launchctl_getenv(name: str) -> str:
    if shutil.which("launchctl") is None:
        return ""
    return run(["launchctl", "getenv", name], check=False, quiet=True).stdout.strip()


def cache_cap_set() -> bool:
    """The cap is set for this login (`launchctl setenv`) or permanently in Ollama's launch agent."""
    if launchctl_getenv(CACHE_RAM_VAR) or os.environ.get(CACHE_RAM_VAR):
        return True
    for plist in (Path.home() / "Library" / "LaunchAgents").glob("*ollama*.plist"):
        try:
            with plist.open("rb") as f:
                env = plistlib.load(f).get("EnvironmentVariables") or {}
        except (OSError, ValueError, plistlib.InvalidFileException, AttributeError):
            continue
        if env.get(CACHE_RAM_VAR):
            return True
    return False


def start_ollama(args, cfg: dict) -> subprocess.Popen | None:
    """Ensure Ollama is up with a capped prompt cache. Returns a child process only when this script had to run
    `ollama serve` itself (no brew service)."""
    step("Ollama")
    need("ollama", "Install it: brew install ollama")
    running = http_up(f"{OLLAMA_URL}/api/tags")
    capped = cache_cap_set()
    child = None

    if brew_ollama_service():
        if not capped and not args.no_ollama_cap:
            # Lasts until reboot; Ollama started by launchd after this inherits it (docs/DESIGN.md §8).
            run(["launchctl", "setenv", CACHE_RAM_VAR, CACHE_RAM_MB])
            run(["brew", "services", "restart", "ollama"])
            ok(f"prompt cache capped at {CACHE_RAM_MB} MB (Ollama restarted)")
        elif not running:
            run(["brew", "services", "start", "ollama"])
        elif not capped:
            warn("prompt cache uncapped (--no-ollama-cap): expect swap on a 16 GB Mac")
    elif not running:
        env = dict(os.environ)
        if not args.no_ollama_cap:
            env.setdefault(CACHE_RAM_VAR, CACHE_RAM_MB)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log = open(LOG_DIR / "ollama.log", "a")  # noqa: SIM115 - lives as long as the child
        print(_c("2", "    $ ollama serve   (output: data/logs/ollama.log)"), flush=True)
        child = subprocess.Popen(
            ["ollama", "serve"], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
    elif not capped:
        warn(
            "Ollama is running outside brew services; restart it with "
            f"{CACHE_RAM_VAR}={CACHE_RAM_MB} in its environment to cap its prompt cache"
        )

    if not wait_for(lambda: http_up(f"{OLLAMA_URL}/api/tags"), "Ollama", 60):
        fail("Ollama didn't start (see `brew services info ollama` or data/logs/ollama.log)")

    model = chat_model(cfg)
    tags = http_json(f"{OLLAMA_URL}/api/tags") or {}
    names = {m.get("name") for m in tags.get("models", [])} if isinstance(tags, dict) else set()
    if model not in names and f"{model}:latest" not in names:
        if not ask(f"Model {model} is missing (~2.5 GB). Pull it now?", args.yes):
            fail(f"Run: ollama pull {model}")
        run(["ollama", "pull", model])
    ok(f"Ollama ready with {model}")
    return child


# ---------------------------------------------------------------- Docker + Qdrant


def start_qdrant() -> None:
    step("Docker + Qdrant")
    need("docker", "Install Docker Desktop: https://www.docker.com/products/docker-desktop/")
    if run(["docker", "info"], check=False, quiet=True).returncode != 0:
        if sys.platform == "darwin":
            print(_c("2", "    $ open -a Docker   (Docker Desktop takes ~20-60 s to start)"), flush=True)
            subprocess.run(["open", "-a", "Docker"], check=False)
        if not wait_for(lambda: run(["docker", "info"], check=False, quiet=True).returncode == 0, "Docker", 180, 3):
            fail("Docker isn't running. Start Docker Desktop and run this again.")
    run([*COMPOSE, "up", "-d", "qdrant"])
    if not wait_for(lambda: http_up(f"{QDRANT_URL}/readyz"), "Qdrant", 60):
        fail("Qdrant didn't become ready: docker logs gibberlink-qdrant")
    ok("Qdrant ready on 127.0.0.1:6333")


def start_searxng(cfg: dict) -> None:
    """Live web search (docs/DESIGN.md §3.7): start the self-hosted SearXNG when the config turns it on."""
    ws = (cfg.get("tools") or {}).get("web_search") or {}
    url = searxng_url(cfg)
    if not ws.get("enabled"):
        ok("live web search is off in the config; SearXNG not needed")
        return
    if cfg.get("strict_offline", True) and "web_search" not in (cfg.get("strict_offline_exceptions") or []):
        fail(
            'tools.web_search.enabled is true but "web_search" is not in strict_offline_exceptions: the backend '
            "refuses to start like that. Add it to the config, or set enabled to false."
        )
    if url is None:
        ok(f"live web search uses {ws.get('provider')} at {ws.get('url')}; nothing to start here")
        return
    run([*COMPOSE, "--profile", "websearch", "up", "-d", "searxng"])
    if wait_for(lambda: http_up(f"{url}/healthz"), "SearXNG", 60):
        ok(f"SearXNG ready on {url} (live web search on: only search queries leave this machine)")
    else:
        warn("SearXNG didn't become ready (docker logs gibberlink-searxng); answers needing live data will say so")


def docker_running() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=10, check=False).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def stop_qdrant() -> None:
    step("Stopping Qdrant and SearXNG")
    if not docker_running():
        ok("Docker isn't running, so Qdrant and SearXNG are already stopped")
        return
    # `stop`, never `down`: down acts on the whole compose project (infra/docker-compose.yml header).
    done = run([*COMPOSE, "--profile", "websearch", "stop", "qdrant", "searxng"], check=False, quiet=True)
    if done.returncode == 0:
        ok("Qdrant and SearXNG stopped")
    else:
        warn(f"docker compose stop failed: {(done.stderr or done.stdout).strip()}")


# ---------------------------------------------------------------- model weights


def check_weights(args) -> None:
    step("Model weights")
    missing = [w for w in REQUIRED_WEIGHTS if not (HF_HUB / w).is_dir()]
    docling_ok = DOCLING_DIR.is_dir() and any(DOCLING_DIR.iterdir())
    if not missing and docling_ok:
        ok("all present in data/models")
        return
    names = [w.removeprefix("models--").replace("--", "/") for w in missing] + ([] if docling_ok else ["docling"])
    warn("missing: " + ", ".join(names))
    if not ask("Download them now (several GB, one time)?", args.yes):
        fail("Run: scripts/setup/download_models.sh all")
    need("uvx", "Install uv: brew install uv")
    run([str(ROOT / "scripts" / "setup" / "download_models.sh"), "all"])


# ---------------------------------------------------------------- backend + frontend


class Child:
    """A long-running child whose output is shown with a prefix and appended to data/logs/<name>.log."""

    def __init__(self, name: str, color: str, cmd: list[str], cwd: Path, env: dict | None = None):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.prefix = _c(color, f"[{name}]")
        self.log = open(LOG_DIR / f"{name}.log", "a", encoding="utf-8")  # noqa: SIM115 - closed in stop()
        self.log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
        print(_c("2", f"    $ {' '.join(cmd)}   (output: data/logs/{name}.log)"), flush=True)
        self.proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env={**os.environ, **(env or {})},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,  # own process group: Ctrl+C reaches this script only
        )
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self.log.write(line)
            self.log.flush()
            say(f"{self.prefix} {line}", end="")  # keep draining the pipe even if the terminal is gone

    def alive(self) -> bool:
        return self.proc.poll() is None

    def stop(self) -> None:
        if self.alive():
            try:
                os.killpg(self.proc.pid, signal.SIGINT)
                self.proc.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self.proc.pid, signal.SIGKILL)
        self.log.close()


def port_owner(port: int) -> str:
    if shutil.which("lsof") is None:
        return "another process"
    out = run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"], check=False, quiet=True).stdout.splitlines()
    return " ".join(out[1].split()[:2]) if len(out) > 1 else "another process"


def listener_pid(port: int) -> int | None:
    if shutil.which("lsof") is None:
        return None
    out = run(["lsof", "-nP", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"], check=False, quiet=True).stdout.split()
    return int(out[0]) if out else None


def process_cwd(pid: int) -> Path | None:
    out = run(["lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"], check=False, quiet=True).stdout.splitlines()
    return next((Path(line[1:]) for line in out if line.startswith("n")), None)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def stop_group(pid: int, name: str) -> None:
    """Stop ``pid``'s process group (uv + python, or npm + next), gently first."""
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return
    target = pgid if pgid != os.getpgid(0) else None  # never signal our own group
    for sig, wait_s in ((signal.SIGINT, 10), (signal.SIGTERM, 5), (signal.SIGKILL, 2)):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(target, sig) if target else os.kill(pid, sig)
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline and pid_alive(pid):
            time.sleep(0.2)
        if not pid_alive(pid):
            ok(f"{name} stopped")
            return
    warn(f"couldn't stop the {name} (pid {pid})")


def stop_apps() -> None:
    """Stop the backend and frontend: through the start.py that runs them when it is alive, else directly (a start.py
    that was killed or lost its terminal can leave them running). Only processes running from this repo are touched."""
    step("Stopping the backend and frontend")
    try:
        owner = int(PID_FILE.read_text().strip())
    except (OSError, ValueError):
        owner = None
    if owner and owner != os.getpid() and pid_alive(owner):
        os.kill(owner, signal.SIGTERM)  # its shutdown stops its children in order
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and pid_alive(owner):
            time.sleep(0.2)
        if not pid_alive(owner):
            ok(f"the running start.py (pid {owner}) stopped them")
    for port, name, folder in ((3000, "frontend", FRONTEND_DIR), (8000, "backend", BACKEND_DIR)):
        pid = listener_pid(port)
        if pid is None:
            ok(f"{name} not running")
            continue
        cwd = process_cwd(pid)
        if cwd is None or not str(cwd).startswith(str(folder)):
            warn(f"port {port} is used by {port_owner(port)}, not this project's {name}; left alone")
            continue
        stop_group(pid, name)


def backend_health() -> dict:
    h = http_json(f"{BACKEND_URL}/health", timeout=3)
    return h if isinstance(h, dict) else {}


def start_backend(children: list[Child], cfg: dict) -> None:
    step("Backend (127.0.0.1:8000)")
    reused = bool(backend_health().get("version"))
    if reused:
        ok("already running; reusing it")
    else:
        if http_up(BACKEND_URL):
            fail(f"port 8000 is taken by {port_owner(8000)}; stop it and run this again")
        need("uv", "Install it: brew install uv")
        run(["uv", "sync", "--group", "ml", "--quiet"], cwd=BACKEND_DIR)
        children.append(
            Child("backend", "35", ["uv", "run", "python", "-m", "app"], BACKEND_DIR, {"PYTHONUNBUFFERED": "1"})
        )
    print("    waiting for the models to load (~40-70 s)…", flush=True)

    def ready() -> bool:
        if children and children[-1].name == "backend" and not children[-1].alive():
            fail("the backend exited; see the [backend] lines above or data/logs/backend.log")
        state = (backend_health().get("preload") or {}).get("state")
        return state in ("ready", "degraded", "off")

    wait_for(ready, "the backend's models", 300, 2)
    h = backend_health()
    state = (h.get("preload") or {}).get("state", "unknown")
    (ok if state == "ready" else warn)(f"backend preload: {state}")
    for p in h.get("providers", []):
        if p.get("status") not in ("ok", "disabled"):
            warn(f"{p.get('capability')}: {p.get('status')} — {p.get('detail')}")
    for w in h.get("warnings", []):
        warn(f"{w.get('message')} Fix: {w.get('fix')}")
    search = next((p for p in h.get("providers", []) if p.get("capability") == "web_search"), {})
    if reused and searxng_url(cfg) and search.get("status") == "disabled":
        warn(
            "the config turns live web search on, but the running backend started with it off: restart the backend "
            "(Ctrl+C in the terminal that started it, then run start.py again)"
        )


def start_frontend(children: list[Child]) -> None:
    step("Frontend (127.0.0.1:3000)")
    if http_up(FRONTEND_URL):
        ok("something already answers on port 3000; reusing it")
        return
    need("npm", "Install Node 24: brew install node@24")
    lock, stamp = FRONTEND_DIR / "package-lock.json", FRONTEND_DIR / "node_modules" / ".package-lock.json"
    if not stamp.exists() or lock.stat().st_mtime > stamp.stat().st_mtime:
        run(["npm", "install", "--no-audit", "--no-fund"], cwd=FRONTEND_DIR)
    children.append(Child("frontend", "34", ["npm", "run", "dev"], FRONTEND_DIR))
    if not wait_for(lambda: http_up(FRONTEND_URL, timeout=5), "the frontend", 120, 1):
        fail("the frontend didn't start; see data/logs/frontend.log")
    ok("frontend ready")


def open_browser() -> None:
    if sys.platform != "darwin":
        return
    chrome = Path("/Applications/Google Chrome.app")
    cmd = ["open", "-a", "Google Chrome", APP_URL] if chrome.exists() else ["open", APP_URL]
    subprocess.run(cmd, check=False)


# ---------------------------------------------------------------- commands


def cmd_status() -> int:
    def line(name: str, up: bool, detail: str = "") -> None:
        mark = _c("32", "up  ") if up else _c("31", "down")
        print(f"  {mark} {name:<9} {detail}")

    line("Ollama", http_up(f"{OLLAMA_URL}/api/tags"), f"prompt cache cap: {'set' if cache_cap_set() else 'NOT set'}")
    docker = docker_running()
    line("Docker", docker, "" if docker else "Docker Desktop isn't running (start.py opens it)")
    line("Qdrant", http_up(f"{QDRANT_URL}/readyz"), QDRANT_URL)
    url = searxng_url(load_config())
    if url:
        line("SearXNG", http_up(f"{url}/healthz"), f"{url} (live web search on in the config)")
    else:
        print(f"  {_c('2', 'off ')} SearXNG   live web search off in the config")
    h = backend_health()
    line("Backend", bool(h), f"preload: {(h.get('preload') or {}).get('state', '-')}" if h else BACKEND_URL)
    line("Frontend", http_up(FRONTEND_URL), APP_URL)
    return 0


def cmd_start(args) -> int:
    if sys.platform != "darwin":
        warn("this script is written for macOS (Apple Silicon); on other systems see README.md")
    os.chdir(ROOT)
    children: list[Child] = []
    ollama_child = None

    def shutdown(*_):
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)  # one shutdown, even if Ctrl+C is pressed again
        say(_c("1;36", "\n==> Stopping backend and frontend…"))
        for c in reversed(children):
            c.stop()
        if ollama_child is not None and ollama_child.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(ollama_child.pid, signal.SIGTERM)
        if args.stop_qdrant:
            stop_qdrant()
        with contextlib.suppress(OSError):
            if PID_FILE.read_text().strip() == str(os.getpid()):
                PID_FILE.unlink()
        kept = "Ollama keeps" if args.stop_qdrant else "Qdrant, SearXNG and Ollama keep"
        say(f"    stopped. {kept} running (python3 start.py stop stops everything but Ollama).")
        os._exit(0)  # don't wait for the output threads

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGHUP, shutdown)  # the terminal was closed: don't leave the backend and frontend orphaned
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))

    cfg = load_config()
    ollama_child = start_ollama(args, cfg)
    start_qdrant()
    start_searxng(cfg)
    check_weights(args)
    start_backend(children, cfg)
    start_frontend(children)

    step(f"Ready: {APP_URL}")
    print("    Open it in Chrome and allow the microphone. Demo script: docs/DEMO.md.", flush=True)
    print("    Ctrl+C here (or `python3 start.py stop` anywhere) stops the backend and frontend.", flush=True)
    if not args.no_browser:
        open_browser()

    if not children:  # everything was already running
        with contextlib.suppress(OSError):
            PID_FILE.unlink()
        return 0
    while all(c.alive() for c in children):
        time.sleep(1)
    dead = next(c for c in children if not c.alive())
    warn(f"the {dead.name} exited (code {dead.proc.returncode}); see data/logs/{dead.name}.log")
    shutdown()
    return 1


def main() -> int:
    p = argparse.ArgumentParser(description="Start poc_gibberlink and everything it needs (macOS).")
    p.add_argument("command", nargs="?", default="start", choices=["start", "status", "stop"])
    p.add_argument("-y", "--yes", action="store_true", help="answer yes to downloads (model weights, Ollama model)")
    p.add_argument("--no-browser", action="store_true", help="don't open Chrome")
    p.add_argument(
        "--no-ollama-cap",
        action="store_true",
        help=f"don't set {CACHE_RAM_VAR} / restart Ollama (not recommended on 16 GB)",
    )
    p.add_argument("--stop-qdrant", action="store_true", help="also stop Qdrant and SearXNG on Ctrl+C")
    args = p.parse_args()
    if args.command == "status":
        return cmd_status()
    if args.command == "stop":
        stop_apps()
        stop_qdrant()
        say("    Ollama keeps running (it is shared; `brew services stop ollama` stops it).")
        return 0
    return cmd_start(args)


if __name__ == "__main__":
    sys.exit(main())
