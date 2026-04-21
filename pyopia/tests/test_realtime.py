import logging
import queue
import threading
from pathlib import Path

import pyopia.realtime


class _DummyObserver:
    def __init__(self):
        self.scheduled_path = None
        self.recursive = None
        self.started = False
        self.stopped = False
        self.joined = False

    def schedule(self, _handler, path, recursive):
        self.scheduled_path = path
        self.recursive = recursive

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def join(self):
        self.joined = True


class _DummyThread:
    def __init__(self, target, args, daemon):
        self.target = target
        self.args = args
        self.daemon = daemon
        self.started = False
        self.join_timeout = None

    def start(self):
        self.started = True

    def join(self, timeout=None):
        self.join_timeout = timeout


class _DummyProgress:
    def __init__(self, *args, transient=True):
        self.args = args
        self.transient = transient
        self.console = self
        self.task_id = 1
        self.descriptions = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def print(self, *_args, **_kwargs):
        return None

    def add_task(self, description, total=None):
        self.descriptions.append(description)
        return self.task_id

    def update(self, _task_id, description=None):
        if description is not None:
            self.descriptions.append(description)


class _DummyPipeline:
    def __init__(self, config):
        self.config = config


def _patch_realtime_runtime(monkeypatch, created):
    def observer_factory():
        created["observer"] = _DummyObserver()
        return created["observer"]

    def thread_factory(target, args, daemon):
        created["thread"] = _DummyThread(target, args, daemon)
        return created["thread"]

    def raise_keyboard_interrupt(_seconds):
        raise KeyboardInterrupt()

    monkeypatch.setattr(pyopia.realtime, "Observer", observer_factory)
    monkeypatch.setattr(pyopia.realtime.threading, "Thread", thread_factory)
    monkeypatch.setattr(pyopia.realtime, "Progress", _DummyProgress)
    monkeypatch.setattr(pyopia.realtime.pyopia.pipeline, "Pipeline", _DummyPipeline)
    monkeypatch.setattr(pyopia.realtime.time, "sleep", raise_keyboard_interrupt)


def test_resolve_watch_settings_infers_from_raw_pattern(tmp_path: Path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    raw_pattern = str(images_dir / "*.silc")
    watch_folder, file_pattern = pyopia.realtime._resolve_watch_settings(raw_pattern, None)

    assert watch_folder == str(images_dir)
    assert file_pattern == "*.silc"


def test_resolve_watch_settings_prefers_explicit_watch_folder(tmp_path: Path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    raw_pattern = str(images_dir / "*.silc")
    watch_folder, file_pattern = pyopia.realtime._resolve_watch_settings(
        raw_pattern,
        watch_folder="/custom/watch",
    )

    assert watch_folder == "/custom/watch"
    assert file_pattern == "*.silc"


def test_event_handler_enqueues_only_matching_moved_files(tmp_path: Path):
    file_queue = queue.Queue()
    logger = logging.getLogger("test")
    seen_files = set()
    seen_lock = threading.Lock()
    handler = pyopia.realtime._build_event_handler(
        file_queue,
        "*.silc",
        seen_files,
        seen_lock,
        logger,
    )

    matched = tmp_path / "image.silc"
    matched.write_text("ok")
    ignored = tmp_path / "other.txt"
    ignored.write_text("skip")

    moved_event_match = type("Event", (), {"dest_path": str(matched)})()
    moved_event_no_match = type("Event", (), {"dest_path": str(ignored)})()

    handler.on_moved(moved_event_match)
    handler.on_moved(moved_event_no_match)

    queued = file_queue.get_nowait()
    assert queued == matched
    assert file_queue.empty()


def test_event_handler_deduplicates_same_moved_file(tmp_path: Path):
    file_queue = queue.Queue()
    logger = logging.getLogger("test")
    seen_files = set()
    seen_lock = threading.Lock()
    handler = pyopia.realtime._build_event_handler(
        file_queue,
        "*.silc",
        seen_files,
        seen_lock,
        logger,
    )

    matched = tmp_path / "image.silc"
    matched.write_text("ok")
    moved_event = type("Event", (), {"dest_path": str(matched)})()

    handler.on_moved(moved_event)
    handler.on_moved(moved_event)

    queued = file_queue.get_nowait()
    assert queued == matched
    assert file_queue.empty()


def test_enqueue_existing_files_matches_pattern_and_deduplicates(tmp_path: Path):
    file_queue = queue.Queue()
    seen_files = set()
    seen_lock = threading.Lock()
    logger = logging.getLogger("test")

    matched = tmp_path / "a.silc"
    matched.write_text("ok")
    ignored = tmp_path / "b.txt"
    ignored.write_text("skip")

    pyopia.realtime._enqueue_existing_files(
        str(tmp_path),
        "*.silc",
        file_queue,
        seen_files,
        seen_lock,
        logger,
    )
    pyopia.realtime._enqueue_existing_files(
        str(tmp_path),
        "*.silc",
        file_queue,
        seen_files,
        seen_lock,
        logger,
    )

    queued = file_queue.get_nowait()
    assert queued == matched
    assert file_queue.empty()


def test_integration_existing_then_moved_files_processed_once(tmp_path: Path):
    file_queue = queue.Queue()
    seen_files = set()
    seen_lock = threading.Lock()
    logger = logging.getLogger("test")

    existing = tmp_path / "existing.silc"
    existing.write_text("old")
    moved_in = tmp_path / "moved.silc"
    moved_in.write_text("new")

    pyopia.realtime._enqueue_existing_files(
        str(tmp_path),
        "*.silc",
        file_queue,
        seen_files,
        seen_lock,
        logger,
    )

    handler = pyopia.realtime._build_event_handler(
        file_queue,
        "*.silc",
        seen_files,
        seen_lock,
        logger,
    )
    handler.on_moved(type("Event", (), {"dest_path": str(moved_in)})())
    handler.on_moved(type("Event", (), {"dest_path": str(existing)})())

    stop_event = threading.Event()
    processed = []
    runtime_state = {"processed_count": 0, "current_file": "idle"}
    state_lock = threading.Lock()

    class DummyPipeline:
        def run(self, filename):
            processed.append(filename)
            if len(processed) == 2:
                stop_event.set()

    pyopia.realtime._worker_loop(
        stop_event=stop_event,
        file_queue=file_queue,
        processing_pipeline=DummyPipeline(),
        logger=logger,
        runtime_state=runtime_state,
        state_lock=state_lock,
    )

    assert set(processed) == {existing.as_posix(), moved_in.as_posix()}
    assert runtime_state["processed_count"] == 2


def test_worker_loop_processes_one_file_and_uses_posix_path(tmp_path: Path):
    class DummyPipeline:
        def __init__(self):
            self.processed = []

        def run(self, filename):
            self.processed.append(filename)
            stop_event.set()

    image_file = tmp_path / "sample.silc"
    image_file.write_text("content")

    stop_event = threading.Event()
    file_queue = queue.Queue()
    file_queue.put(image_file)
    pipeline = DummyPipeline()
    runtime_state = {"processed_count": 0, "current_file": "idle"}
    state_lock = threading.Lock()

    pyopia.realtime._worker_loop(
        stop_event=stop_event,
        file_queue=file_queue,
        processing_pipeline=pipeline,
        logger=logging.getLogger("test"),
        runtime_state=runtime_state,
        state_lock=state_lock,
    )

    assert pipeline.processed == [image_file.as_posix()]
    assert runtime_state["processed_count"] == 1
    assert runtime_state["current_file"] == "idle"


def test_run_realtime_schedules_observer_and_stops_on_keyboard_interrupt(monkeypatch, tmp_path: Path):
    images_dir = tmp_path / "incoming"
    images_dir.mkdir()
    pipeline_config = {
        "general": {
            "raw_files": str(images_dir / "*.silc"),
        },
        "steps": {},
    }

    created = {}
    progress_calls = {"added": 0, "updated": 0}

    class TrackingProgress(_DummyProgress):
        def add_task(self, description, total=None):
            progress_calls["added"] += 1
            return super().add_task(description, total)

        def update(self, _task_id, description=None):
            progress_calls["updated"] += 1
            return super().update(_task_id, description)

    _patch_realtime_runtime(monkeypatch, created)
    monkeypatch.setattr(pyopia.realtime, "Progress", TrackingProgress)

    pyopia.realtime.run_realtime(pipeline_config)

    assert created["observer"].scheduled_path == str(images_dir)
    assert created["observer"].recursive is False
    assert created["observer"].started is True
    assert created["observer"].stopped is True
    assert created["observer"].joined is True
    assert created["thread"].started is True
    assert created["thread"].daemon is True
    assert created["thread"].join_timeout == 5
    assert progress_calls["added"] == 1
    assert progress_calls["updated"] >= 1
