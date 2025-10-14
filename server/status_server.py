#!/usr/bin/env python3
import json
import logging
import os
import signal
import socket
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import socketserver

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
_LOGGER = logging.getLogger(__name__)

_STATUS_SERVER_HOST = os.environ.get("TRITON_INSTANCE_STATUS_HOST", "0.0.0.0")
_STATUS_SERVER_PORT = int(os.environ.get("TRITON_INSTANCE_STATUS_PORT", "8081"))
_STATUS_ENDPOINT_PATH = os.environ.get(
    "TRITON_INSTANCE_STATUS_PATH", "/instance_status"
)
_STATUS_FILE_PATTERN = "triton_instance_status_*.json"
_STATUS_DIR = Path("/tmp")


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
            request_path = urlsplit(self.path).path.rstrip("/") or "/"
            endpoint_path = _STATUS_ENDPOINT_PATH.rstrip("/") or "/"
            if request_path != endpoint_path:
                self._send_json(404, {"error": "Not Found"})
                return

            now = time.time()
            aggregate = {"active_instances": 0, "configured_instances": 0}

            for status_file in _STATUS_DIR.glob(_STATUS_FILE_PATTERN):
                try:
                    stat_info = status_file.stat()
                    if stat_info.st_size == 0 or status_file.suffix == ".tmp":
                        continue
                    with status_file.open("r", encoding="utf-8") as handle:
                        data = json.load(handle)
                    aggregate["active_instances"] += int(
                        data.get("active_instances", 0)
                    )
                    configured = int(data.get("configured_instances", 0))
                    aggregate["configured_instances"] = max(
                        aggregate["configured_instances"], configured
                    )
                except Exception as exc:
                    _LOGGER.debug("Failed to read %s: %s", status_file, exc)

            aggregate["idle_instances"] = max(
                aggregate["configured_instances"] - aggregate["active_instances"], 0
            )
            total_capacity = (
                aggregate["active_instances"] + aggregate["idle_instances"]
            )
            if total_capacity > aggregate["configured_instances"]:
                aggregate["configured_instances"] = total_capacity

            aggregate["last_updated"] = int(now)
            self._send_json(200, aggregate)
        except BrokenPipeError:
            pass
        except Exception as exc:
            _LOGGER.error("Error in status handler: %s", exc)
            try:
                self._send_json(500, {"error": str(exc)})
            except Exception:
                pass

    def log_message(self, format, *args):
        pass


def _install_signal_handlers(server: HTTPServer) -> None:
    def _shutdown(signum, frame):
        _LOGGER.info("Shutting down status server...")
        server.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)


def run_server():
    try:
        httpd = _ThreadedHTTPServer(
            (_STATUS_SERVER_HOST, _STATUS_SERVER_PORT),
            _InstanceStatusRequestHandler,
        )
        httpd.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        _LOGGER.info(
            "Instance status server started on %s:%s%s",
            _STATUS_SERVER_HOST,
            _STATUS_SERVER_PORT,
            _STATUS_ENDPOINT_PATH,
        )
        _install_signal_handlers(httpd)
        httpd.serve_forever()
    except Exception as exc:
        _LOGGER.error("Status server error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    run_server()
