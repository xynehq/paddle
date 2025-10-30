import atexit
import errno
import json
import os
import re
import threading
import time
from typing import Any, List, Optional


def _safe_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        if exc.errno != errno.ENOENT:
            # Ignore unlink failures for non-existent or transient files.
            pass


class InstanceStatusTracker:
    """Persist configured instance counts for monitoring."""

    def __init__(self, status_file: str, instance_id: str, write_interval: float):
        self._status_file = status_file
        self._instance_id = instance_id
        self._write_interval = write_interval
        self._status_lock = threading.Lock()
        self._configured_instances = 0
        self._last_status_write = 0.0
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None
        atexit.register(self._cleanup)

    def start(self) -> None:
        if self._heartbeat_thread is not None:
            return
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="blip-status-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def update_total_instances(self, candidate_total: int, *, force: bool = False) -> None:
        if candidate_total <= 0:
            return
        with self._status_lock:
            if force or self._configured_instances <= 0:
                self._configured_instances = candidate_total
            else:
                self._configured_instances = max(
                    self._configured_instances, candidate_total
                )
        self.write_status(force=True)

    def write_status(self, *, force: bool = False) -> None:
        try:
            now = time.time()
            if not force and (now - self._last_status_write) < self._write_interval:
                return

            with self._status_lock:
                total = self._configured_instances

            status_data = {
                "configured_instances": max(0, total),
                "last_updated": int(now),
                "instance_id": self._instance_id,
            }
            tmp_path = self._status_file + ".tmp"
            with open(tmp_path, "w") as file_obj:
                json.dump(status_data, file_obj, separators=(",", ":"))
            os.replace(tmp_path, self._status_file)
            self._last_status_write = now
        except Exception as exc:
            print(
                f"[BLIP-STATUS] Failed to write status file {self._status_file}: {exc}"
            )

    def _heartbeat_loop(self):
        while not self._heartbeat_stop.wait(self._write_interval):
            self.write_status()

    def _cleanup(self):
        self._heartbeat_stop.set()
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=1.0)
        _safe_unlink(self._status_file)
        _safe_unlink(self._status_file + ".tmp")


def infer_configured_instance_count(model_config: Any) -> int:
    if model_config is None:
        return 0

    def _extract_groups(config_obj: Any) -> List[Any]:
        if config_obj is None:
            return []
        if isinstance(config_obj, dict):
            if "instance_group" in config_obj:
                return config_obj["instance_group"] or []
            if "config" in config_obj:
                return _extract_groups(config_obj["config"])
            return []
        groups = getattr(config_obj, "instance_group", None)
        if groups is not None:
            return list(groups)
        nested = getattr(config_obj, "config", None)
        if nested is not None:
            return _extract_groups(nested)
        return []

    groups = _extract_groups(model_config)
    total = 0
    for group in groups:
        if isinstance(group, dict):
            candidate = group.get("count", 0)
        else:
            candidate = getattr(group, "count", 0)
        try:
            total += int(candidate)
        except (TypeError, ValueError):
            continue
    return max(total, 0)


def _candidate_config_paths(args: Any) -> List[str]:
    candidates: List[str] = []

    def _add(path: Optional[str]):
        if path and path not in candidates:
            candidates.append(path)

    directories: List[str] = []

    def _add_directory(path: Optional[str]):
        if path and path not in directories:
            directories.append(path)

    if isinstance(args, dict):
        model_dir = args.get("model_directory")
        if model_dir:
            _add_directory(model_dir)
        repo_dir = args.get("model_repository")
        if repo_dir:
            _add_directory(repo_dir)
            model_version = args.get("model_version")
            if model_version:
                _add_directory(os.path.join(repo_dir, model_version))

    module_dir = os.path.dirname(__file__)
    _add_directory(module_dir)
    parent_dir = os.path.dirname(module_dir)
    _add_directory(parent_dir)
    grandparent_dir = os.path.dirname(parent_dir)
    _add_directory(grandparent_dir)

    for directory in directories:
        _add(os.path.join(directory, "config_gpu.pbtxt"))
        _add(os.path.join(directory, "config.pbtxt"))
        _add(os.path.join(directory, "config_cpu.pbtxt"))

    env_config_path = os.environ.get("TRITON_MODEL_CONFIG_PATH")
    _add(env_config_path)
    return candidates


def infer_instance_count_from_config_files(args: Any) -> int:
    for path in _candidate_config_paths(args):
        try:
            with open(path, "r", encoding="utf-8") as config_file:
                contents = config_file.read()
        except FileNotFoundError:
            continue
        except OSError:
            continue
        matches = re.findall(r"\bcount\s*:\s*(\d+)", contents)
        candidate_total = sum(int(match) for match in matches)
        if candidate_total > 0:
            return candidate_total
    return 0
