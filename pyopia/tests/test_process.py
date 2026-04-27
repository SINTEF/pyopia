import numpy as np
import pytest

from pyopia.pipeline import Pipeline
import pyopia.process


def test_segment_selects_accurate_method(monkeypatch):
    img = np.ones((4, 4), dtype=float)
    call_state = {"fast": 0, "accurate": 0}

    def fake_fast(_img, _threshold):
        call_state["fast"] += 1
        return np.zeros_like(_img, dtype=bool)

    def fake_accurate(_img, _threshold):
        call_state["accurate"] += 1
        return np.ones_like(_img, dtype=bool)

    monkeypatch.setattr(pyopia.process, "image2blackwhite_fast", fake_fast)
    monkeypatch.setattr(pyopia.process, "image2blackwhite_accurate", fake_accurate)
    monkeypatch.setattr(pyopia.process, "clean_bw", lambda imbw, _minimum_area: imbw)

    pyopia.process.segment(img, segmentation_method="accurate")

    assert call_state["fast"] == 0
    assert call_state["accurate"] == 1


def test_segment_raises_on_unknown_method():
    img = np.ones((4, 4), dtype=float)

    with pytest.raises(ValueError, match="Unknown segmentation_method"):
        pyopia.process.segment(img, segmentation_method="unknown")


def test_pipeline_segmentation_step_accepts_segmentation_method_config(monkeypatch):
    captured = {}

    def fake_segment(img, threshold, minimum_area, fill_holes, segmentation_method):
        captured["segmentation_method"] = segmentation_method
        return np.zeros_like(img, dtype=bool)

    monkeypatch.setattr(pyopia.process, "segment", fake_segment)

    config = {
        "general": {"raw_files": []},
        "steps": {
            "segmentation": {
                "pipeline_class": "pyopia.process.Segment",
                "threshold": 0.9,
                "minimum_area": 5,
                "fill_holes": True,
                "segment_source": "im_corrected",
                "segmentation_method": "accurate",
            }
        },
    }

    pipeline = Pipeline(config, initial_steps=[])
    pipeline.data["im_corrected"] = np.ones((4, 4), dtype=float)
    pipeline.run_step("segmentation")

    assert captured["segmentation_method"] == "accurate"
