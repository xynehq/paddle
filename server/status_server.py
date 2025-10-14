#!/usr/bin/env python3
"""
Standalone instance status server for Triton.
This runs independently and monitors instance status via shared state.
"""
import json
import logging
import os
import signal
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import socketserver

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
_LOGGER = logging.getLogger(__name__)

_STATUS_SERVER_HOST = os.environ.get("TRITON_INSTANCE_STATUS_HOST", "0.0.0.0")
_STATUS_SERVER_PORT = int(os.environ.get("TRITON_INSTANCE_STATUS_PORT", "8081"))
_STATUS_ENDPOINT_PATH = os.environ.get("TRITON_INSTANCE_STATUS_PATH", "/instance_status")
_STATUS_FILE_PATTERN = "/tmp/triton_instance_status_*.json"


class _ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class _InstanceStatusRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            normalized_path = self.path.rstrip("/") or "/"
            normalized_endpoint = _STATUS_ENDPOINT_PATH.rstrip("/") or "/"
            
            if normalized_path != normalized_endpoint:
                self.send_response(404)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", "9")
                self.end_headers()
                self.wfile.write(b"Not Found")
                return

            # Aggregate status from all instance files
            try:
                status_files = list(Path("/tmp").glob("triton_instance_status_*.json"))
                
                if not status_files:
                    status_data = {
                        "active_instances": 0,
                        "configured_instances": 0,
                        "idle_instances": 0
                    }
                else:
                    aggregate = {
                        "active_instances": 0,
                        "configured_instances": 0,
                        "idle_instances": 0
                    }
                    
                    for status_file in status_files:
                        try:
                            with open(status_file, 'r') as f:
                                data = json.load(f)
                            aggregate["active_instances"] += data.get("active_instances", 0)
                            aggregate["configured_instances"] += data.get("configured_instances", 0)
                            aggregate["idle_instances"] += data.get("idle_instances", 0)
                        except Exception as e:
                            _LOGGER.debug(f"Failed to read {status_file}: {e}")
                            continue
                    
                    status_data = aggregate
                    
            except Exception as e:
                _LOGGER.error(f"Error aggregating status files: {e}")
                status_data = {
                    "active_instances": 0,
                    "configured_instances": 0,
                    "idle_instances": 0
                }

            response = json.dumps(status_data, separators=(",", ":")).encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(response)
            self.wfile.flush()
            
        except BrokenPipeError:
            pass
        except Exception as e:
            _LOGGER.error(f"Error in status handler: {e}")
            try:
                self.send_error(500, f"Internal error: {e}")
            except:
                pass

    def log_message(self, format, *args):
        # Suppress default logging
        pass


def run_server():
    """Run the status server."""
    try:
        httpd = _ThreadedHTTPServer(
            (_STATUS_SERVER_HOST, _STATUS_SERVER_PORT),
            _InstanceStatusRequestHandler,
        )
        httpd.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        httpd.timeout = 30
        
        _LOGGER.info(
            f"Instance status server started on {_STATUS_SERVER_HOST}:{_STATUS_SERVER_PORT}{_STATUS_ENDPOINT_PATH}"
        )
        
        # Handle shutdown gracefully
        def signal_handler(sig, frame):
            _LOGGER.info("Shutting down status server...")
            httpd.shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        httpd.serve_forever()
        
    except OSError as exc:
        _LOGGER.error(
            f"Instance status server failed to bind {_STATUS_SERVER_HOST}:{_STATUS_SERVER_PORT} ({exc})"
        )
        sys.exit(1)
    except Exception as exc:
        _LOGGER.error(f"Instance status server error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    run_server()
