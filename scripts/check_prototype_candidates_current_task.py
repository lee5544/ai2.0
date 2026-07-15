#!/usr/bin/env python3
"""Regression check: Prototype candidates must be limited to the open task."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forvia_label_v2.backend import prototype


class _LabelTable:
    def latest_label_map(self):
        base = {
            "line": "epump3",
            "sn": "SN",
            "timestamp": "2026-07-01T00:00:00",
            "source": "expert",
            "result_key": "nok",
            "result_id": "1",
            "result_name": "异常",
            "reason_key": "friction",
            "reason_id": "103",
            "reason_name": "摩擦",
            "reason_confidence": "0.9",
            "label_version": "v1.1",
            "note": prototype.TYPICAL_TAG,
        }
        return {
            "A_up": {**base, "sample_id": "A_up"},
            "B_up": {**base, "sample_id": "B_up"},
        }


class _SampleView:
    def __getitem__(self, key):
        if key != "sample_id":
            raise KeyError(key)
        return ["A_up"]


class _Session:
    sample_view = _SampleView()
    label_table = _LabelTable()
    path_map = {}


def main() -> None:
    original = (
        prototype._load_tagged_label_map,
        prototype._load_sample_paths,
        prototype._load_manifest_paths,
        prototype._resolve_candidate_path,
        prototype._exists,
    )
    try:
        def fake_tagged(_session, tag):
            if tag == prototype.TYPICAL_TAG:
                return _session.label_table.latest_label_map()
            if tag == prototype.PROTOTYPE_TAG:
                return {"B_up": {"note": prototype.PROTOTYPE_TAG}}
            return {}

        prototype._load_tagged_label_map = fake_tagged
        prototype._load_sample_paths = lambda _session: {}
        prototype._load_manifest_paths = lambda _session: {}
        prototype._resolve_candidate_path = lambda *_args, **_kwargs: Path("/tmp/sample.tdms.zst")
        prototype._exists = lambda _path: True

        rows = prototype.candidates(_Session())
        sample_ids = [row["sample_id"] for row in rows]
        assert sample_ids == ["A_up"], sample_ids
        assert not rows[0]["in_prototype"]
    finally:
        (
            prototype._load_tagged_label_map,
            prototype._load_sample_paths,
            prototype._load_manifest_paths,
            prototype._resolve_candidate_path,
            prototype._exists,
        ) = original


if __name__ == "__main__":
    main()
