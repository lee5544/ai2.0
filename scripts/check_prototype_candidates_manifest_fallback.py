#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forvia_label_v2.backend import prototype


class _Values(list):
    def tolist(self):
        return list(self)


class _SampleView:
    def __getitem__(self, key):
        if key == "sample_id":
            return _Values(["SN_A_up"])
        raise KeyError(key)


class _LabelTable:
    def latest_label_map(self):
        return {
            "SN_A_up": {
                "line": "epump3",
                "sn": "SN_A",
                "sample_id": "SN_A_up",
                "reason_name": "摩擦",
                "source": "expert",
                "note": prototype.TYPICAL_TAG,
            }
        }


class _Session:
    sample_view = _SampleView()
    label_table = _LabelTable()
    path_map = {"SN_A_up": Path("/tmp/stale_missing.tdms.zst")}


def main():
    existing = Path("/tmp/factory_raw/epump3/prototype/摩擦/E-pump_SN_A.tdms.zst")
    saved = (
        prototype._load_tagged_label_map,
        prototype._load_manifest_paths,
        prototype._data_root,
        prototype._exists,
    )
    try:
        prototype._load_tagged_label_map = lambda session, tag: (
            session.label_table.latest_label_map() if tag == prototype.TYPICAL_TAG else {"SN_A_up": {}}
        )
        prototype._load_manifest_paths = lambda session: {("epump3", "sn_a"): existing, ("", "sn_a"): existing}
        prototype._data_root = lambda session: Path("/tmp")
        prototype._exists = lambda path: Path(path) == existing

        rows = prototype.candidates(_Session())
        assert rows and rows[0]["tdms"] == str(existing), rows
        assert rows[0]["resolvable"] is True, rows
        assert rows[0]["in_prototype"] is True, rows
    finally:
        (
            prototype._load_tagged_label_map,
            prototype._load_manifest_paths,
            prototype._data_root,
            prototype._exists,
        ) = saved


if __name__ == "__main__":
    main()
