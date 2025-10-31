#!/usr/bin/env python3
import json
import os
import re
import signal
import socket
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import socketserver

_SERVER_HOST = os.environ.get("TRITON_INSTANCE_STATUS_HOST", "0.0.0.0")
_SERVER_PORT = int(os.environ.get("TRITON_INSTANCE_STATUS_PORT", "8081"))
_PADDLEX_STATUS_ENDPOINT_PATH = os.environ.get(
    "TRITON_INSTANCE_STATUS_PATH", "/paddlex_instance_status"
)
_TRITON_STATUS_FILE_PATTERN = "triton_instance_status_*.json"
_STATUS_DIR = Path("/tmp")
_TRITON_STATUS_FILE_TTL_SECONDS = float(
    os.environ.get("TRITON_INSTANCE_STATUS_TTL_SECONDS", "15.0")
)
_INSTANCE_STATUS_ENDPOINT_PATH = os.environ.get(
    "TRITON_COMBINED_INSTANCE_STATUS_PATH", "/instance_status"
)
_BLIP_STATUS_FILE_PATTERN = "blip_instance_status_*.json"
_BLIP_STATUS_FILE_TTL_SECONDS = float(
    os.environ.get(
        "BLIP_INSTANCE_STATUS_TTL_SECONDS",
        str(_TRITON_STATUS_FILE_TTL_SECONDS),
    )
)
_MODEL_REPO_ROOT = Path(__file__).resolve().parent / "model_repo"


def _read_configured_count_from_directories(directories) -> int:
    for directory in directories:
        directory_path = Path(directory)
        for config_name in ("config_gpu.pbtxt", "config.pbtxt", "config_cpu.pbtxt"):
            config_path = directory_path / config_name
            if not config_path.is_file():
                continue
            try:
                config_text = config_path.read_text(encoding="utf-8")
            except OSError:
                continue
            instance_counts = re.findall(r"\bcount\s*:\s*(\d+)", config_text)
            configured_total = sum(int(match) for match in instance_counts)
            if configured_total > 0:
                return configured_total
    return 0


def _collect_instance_metrics(
    file_pattern: str,
    ttl_seconds: float,
    *,
    now=None,
    fallback_directories=None,
) -> dict:
    current_time = now if now is not None else time.time()
    status_summary = {"active_instances": 0, "configured_instances": 0}
    for status_file in _STATUS_DIR.glob(file_pattern):
        try:
            file_stats = status_file.stat()
            if file_stats.st_size == 0 or status_file.suffix == ".tmp":
                continue
            with status_file.open("r", encoding="utf-8") as handle:
                status_payload = json.load(handle)

            configured_instances = int(status_payload.get("configured_instances", 0))
            status_summary["configured_instances"] = max(
                status_summary["configured_instances"], configured_instances
            )

            is_entry_stale = False
            if ttl_seconds > 0:
                last_updated_raw = status_payload.get("last_updated")
                try:
                    last_updated_ts = float(last_updated_raw)
                except (TypeError, ValueError):
                    last_updated_ts = None

                if last_updated_ts is None:
                    is_entry_stale = True
                else:
                    entry_age = max(current_time - last_updated_ts, 0.0)
                    if entry_age > ttl_seconds:
                        is_entry_stale = True

            if is_entry_stale:
                continue

            status_summary["active_instances"] += int(
                status_payload.get("active_instances", 0)
            )
        except Exception:
            continue

    status_summary["idle_instances"] = max(
        status_summary["configured_instances"] - status_summary["active_instances"],
        0,
    )

    if status_summary["configured_instances"] == 0 and fallback_directories:
        inferred_total = _read_configured_count_from_directories(fallback_directories)
        if inferred_total > 0:
            status_summary["configured_instances"] = inferred_total
            status_summary["idle_instances"] = max(
                inferred_total - status_summary["active_instances"],
                0,
            )

    total_capacity = (
        status_summary["active_instances"] + status_summary["idle_instances"]
    )
    if total_capacity > status_summary["configured_instances"]:
        status_summary["configured_instances"] = total_capacity

    status_summary["last_updated"] = int(current_time)
    return status_summary


def _read_configured_instance_count(
    file_pattern: str,
    ttl_seconds: float,
    *,
    now=None,
    fallback_directories=None,
) -> int:
    current_time = now if now is not None else time.time()
    configured_instances = 0
    for status_file in _STATUS_DIR.glob(file_pattern):
        try:
            file_stats = status_file.stat()
            if file_stats.st_size == 0 or status_file.suffix == ".tmp":
                continue
            with status_file.open("r", encoding="utf-8") as handle:
                status_payload = json.load(handle)

            if ttl_seconds > 0:
                last_updated_raw = status_payload.get("last_updated")
                try:
                    last_updated_ts = float(last_updated_raw)
                except (TypeError, ValueError):
                    last_updated_ts = None
                if last_updated_ts is None:
                    continue
                if (current_time - last_updated_ts) > ttl_seconds:
                    continue

            configured_instances = max(
                configured_instances,
                int(status_payload.get("configured_instances", 0)),
            )
        except Exception:
            continue

    if configured_instances == 0 and fallback_directories:
        configured_instances = _read_configured_count_from_directories(
            fallback_directories
        )
    return configured_instances


class _ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class _InstanceStatusRequestHandler(BaseHTTPRequestHandler):
    def _send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        try:
            normalized_path = urlsplit(self.path).path.rstrip("/") or "/"
            paddlex_status_path = _PADDLEX_STATUS_ENDPOINT_PATH.rstrip("/") or "/"
            instance_status_path = (
                _INSTANCE_STATUS_ENDPOINT_PATH.rstrip("/") or "/"
            )

            if normalized_path == paddlex_status_path:
                payload = _collect_instance_metrics(
                    _TRITON_STATUS_FILE_PATTERN, _TRITON_STATUS_FILE_TTL_SECONDS
                )
                self._send_json(200, payload)
                return

            if normalized_path == instance_status_path:
                current_time = time.time()
                layout_status = _collect_instance_metrics(
                    _TRITON_STATUS_FILE_PATTERN,
                    _TRITON_STATUS_FILE_TTL_SECONDS,
                    now=current_time,
                    fallback_directories=[
                        _MODEL_REPO_ROOT / "layout-parsing",
                        _MODEL_REPO_ROOT / "layout-parsing" / "1",
                    ],
                )
                blip_configured = _read_configured_instance_count(
                    _BLIP_STATUS_FILE_PATTERN,
                    _BLIP_STATUS_FILE_TTL_SECONDS,
                    now=current_time,
                    fallback_directories=[
                        _MODEL_REPO_ROOT / "blip-caption",
                        _MODEL_REPO_ROOT / "blip-caption" / "1",
                    ],
                )
                self._send_json(
                    200,
                    {
                        "layout_parsing": layout_status,
                        "blip_caption": {"configured_instances": blip_configured},
                    },
                )
                return

            self._send_json(404, {"error": "Not Found"})
        except BrokenPipeError:
            pass
        except Exception as exc:
            print(f"[STATUS-SERVER] Error in status handler: {exc}")
            try:
                self._send_json(500, {"error": str(exc)})
            except Exception:
                pass

    def log_message(self, format, *args):
        pass


def _install_signal_handlers(server: HTTPServer) -> None:
    def _shutdown(signum, frame):
        print("[STATUS-SERVER] Shutting down status server...")
        server.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)


def run_server():
    try:
        httpd = _ThreadedHTTPServer(
            (_SERVER_HOST, _SERVER_PORT),
            _InstanceStatusRequestHandler,
        )
        httpd.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        print(
            "[STATUS-SERVER] Instance status server listening on "
            f"{_SERVER_HOST}:{_SERVER_PORT} (instance: {_INSTANCE_STATUS_ENDPOINT_PATH}, "
            f"paddlex: {_PADDLEX_STATUS_ENDPOINT_PATH})"
        )
        _install_signal_handlers(httpd)
        httpd.serve_forever()
    except Exception as exc:
        print(f"[STATUS-SERVER] Fatal error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    run_server()
