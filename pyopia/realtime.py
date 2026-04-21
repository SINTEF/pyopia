"""Utilities for running PyOPIA pipelines in realtime mode."""

import fnmatch
import logging
import pathlib
import queue
import threading
import time

import pandas as pd
from rich import print as rich_print
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

import pyopia.pipeline


def _resolve_watch_settings(raw_files_pattern: str, watch_folder: str | None) -> tuple[str, str]:
    raw_path = pathlib.Path(raw_files_pattern)
    file_pattern = raw_path.name if raw_path.name else "*"
    inferred_watch = str(raw_path.parent) if str(raw_path.parent) != "." else "."

    if watch_folder is None:
        watch_folder = inferred_watch

    return watch_folder, file_pattern


def _enqueue_file_if_new(
    file_path: pathlib.Path,
    file_queue: queue.Queue,
    file_pattern: str,
    seen_files: set[str],
    seen_lock: threading.Lock,
) -> bool:
    if not file_path.exists() or file_path.is_dir():
        return False

    if not fnmatch.fnmatch(file_path.name, file_pattern):
        return False

    file_key = file_path.as_posix()
    with seen_lock:
        if file_key in seen_files:
            return False
        seen_files.add(file_key)

    file_queue.put(file_path)
    return True


def _enqueue_existing_files(
    watch_folder: str,
    file_pattern: str,
    file_queue: queue.Queue,
    seen_files: set[str],
    seen_lock: threading.Lock,
    logger: logging.Logger,
):
    for file_path in sorted(pathlib.Path(watch_folder).glob(file_pattern)):
        if _enqueue_file_if_new(file_path, file_queue, file_pattern, seen_files, seen_lock):
            logger.info(f"Existing file queued: {file_path}")


def _build_event_handler(
    file_queue: queue.Queue,
    file_pattern: str,
    seen_files: set[str],
    seen_lock: threading.Lock,
    logger: logging.Logger,
):
    class NewFileHandler(FileSystemEventHandler):
        def _handle_path(self, path):
            try:
                file_path = pathlib.Path(path)
                if _enqueue_file_if_new(
                    file_path,
                    file_queue,
                    file_pattern,
                    seen_files,
                    seen_lock,
                ):
                    logger.info(f"New file detected: {file_path}")
            except Exception:
                logger.exception("Error handling filesystem event")

        def on_moved(self, event):
            destination = getattr(event, "dest_path", None)
            if destination:
                self._handle_path(destination)

    return NewFileHandler()


def _worker_loop(
    stop_event: threading.Event,
    file_queue: queue.Queue,
    processing_pipeline: pyopia.pipeline.Pipeline,
    logger: logging.Logger,
    runtime_state: dict,
    state_lock: threading.Lock,
):
    while not stop_event.is_set():
        try:
            filepath = file_queue.get(timeout=1)
        except queue.Empty:
            continue

        try:
            with state_lock:
                runtime_state["current_file"] = filepath.name

            start = time.time()
            logger.info(f"Starting processing: {filepath.name}")
            processing_pipeline.run(filepath.as_posix())
            elapsed = time.time() - start
            logger.info(f"Completed {filepath.name} in {elapsed:.1f}s")
            with state_lock:
                runtime_state["processed_count"] += 1
        except Exception as exc:
            logger.exception(f"Error processing {filepath}: {exc}")
        finally:
            with state_lock:
                runtime_state["current_file"] = "idle"
            file_queue.task_done()


def run_realtime(pipeline_config: dict, watch_folder: str | None = None):
    """Run a PyOPIA processing pipeline in realtime by watching a folder.

    Parameters
    ----------
    pipeline_config : dict
        Loaded PyOPIA pipeline config.
    watch_folder : str, optional
        Folder to monitor. If not provided, inferred from ``general.raw_files``.
    """
    logger = logging.getLogger("rich")
    logger.info(f"PyOPIA realtime process started {pd.Timestamp.now()}")
    t1 = time.time()

    raw_files_pattern = pipeline_config["general"].get("raw_files", "*")
    watch_folder, file_pattern = _resolve_watch_settings(raw_files_pattern, watch_folder)

    processing_pipeline = pyopia.pipeline.Pipeline(pipeline_config)

    file_queue = queue.Queue()
    stop_event = threading.Event()
    seen_files: set[str] = set()
    seen_lock = threading.Lock()
    runtime_state = {"processed_count": 0, "current_file": "idle"}
    state_lock = threading.Lock()

    observer = Observer()
    handler = _build_event_handler(
        file_queue,
        file_pattern,
        seen_files,
        seen_lock,
        logger,
    )
    observer.schedule(handler, path=watch_folder, recursive=False)
    observer.start()

    _enqueue_existing_files(
        watch_folder,
        file_pattern,
        file_queue,
        seen_files,
        seen_lock,
        logger,
    )

    worker_thread = threading.Thread(
        target=_worker_loop,
        args=(
            stop_event,
            file_queue,
            processing_pipeline,
            logger,
            runtime_state,
            state_lock,
        ),
        daemon=True,
    )
    worker_thread.start()

    logger.info(
        f"Watching folder {watch_folder} for moved files matching '{file_pattern}'"
    )
    rich_print(
        f"[blue]Realtime started. Watching {watch_folder} for moved files matching '{file_pattern}'"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        task_id = progress.add_task("[blue]Realtime active", total=None)

        try:
            while True:
                with state_lock:
                    processed_count = runtime_state["processed_count"]
                    current_file = runtime_state["current_file"]
                progress.update(
                    task_id,
                    description=(
                        "[blue]Realtime active"
                        f" | processed: {processed_count}"
                        f" | queued: {file_queue.qsize()}"
                        f" | current: {current_file}"
                    ),
                )
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutdown requested, stopping observer and worker...")
        finally:
            stop_event.set()
            observer.stop()
            observer.join()
            worker_thread.join(timeout=5)

        time_total = pd.to_timedelta(time.time() - t1, "seconds")
        progress.console.print(f"[blue]REALTIME PROCESSING STOPPED AFTER {time_total}")
