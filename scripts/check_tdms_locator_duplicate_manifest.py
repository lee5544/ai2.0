import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forvia_label_v2.backend.tdms_locator import ManifestAdapter, TdmsLocator


with tempfile.TemporaryDirectory() as d:
    root = Path(d) / "data_root"
    factory = root / "factory_raw"
    line_dir = factory / "epump3"
    good = line_dir / "prototype" / "摩擦" / "E-pump_14APL198_2358490_16082024_112056.tdms.zst"
    good.parent.mkdir(parents=True)
    good.write_bytes(b"tdms")

    meta = root / "metadata"
    meta.mkdir(parents=True)
    manifest = meta / "tdms_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "line", "sn", "reference", "time", "created_time",
            "tdms_storage_root", "relative_path", "tdms_path",
        ])
        w.writeheader()
        w.writerow({
            "line": "epump3",
            "sn": "14APL198",
            "reference": "2358490",
            "time": "20240816_112056",
            "created_time": "2026-04-07T20:54:20",
            "tdms_storage_root": "factory_raw",
            "relative_path": "prototype/epump3/source/摩擦/E-pump_14APL198_2358490_16082024_112056.tdms.zst",
            "tdms_path": "",
        })
        w.writerow({
            "line": "epump3",
            "sn": "14APL198",
            "reference": "2358490",
            "time": "20240816_112056",
            "created_time": "2026-06-22T10:49:21",
            "tdms_storage_root": "factory_raw",
            "relative_path": "epump3/prototype/摩擦/E-pump_14APL198_2358490_16082024_112056.tdms.zst",
            "tdms_path": str(good),
        })

    locator = TdmsLocator(line_dir, ManifestAdapter(meta))
    path, status = locator.resolve({"line": "epump3", "sn": "14APL198", "sample_id": "14APL198_up"})
    assert status == "registered"
    assert path == good, path

    cross_line = factory / "epump2" / "prototype" / "秒表" / "E-pump_BB6L0460_180652204_18012026_000149.tdms.zst"
    cross_line.parent.mkdir(parents=True)
    cross_line.write_bytes(b"tdms")
    resolved = TdmsLocator._overlap_join(
        factory / "epump4",
        "factory_raw/epump2/prototype/秒表/E-pump_BB6L0460_180652204_18012026_000149.tdms.zst",
    )
    assert resolved == cross_line, resolved

    # 回归：ManifestAdapter 不得把 tdms_storage_root 与 relative_path
    # 预先拼成 epump4/factory_raw/epump2/... 的错误绝对路径。
    cross_manifest = meta / "tdms_manifest_cross_line.csv"
    with cross_manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["line", "sn", "tdms_storage_root", "relative_path"])
        w.writeheader()
        w.writerow({
            "line": "epump2",
            "sn": "BB6L0460",
            "tdms_storage_root": str(factory / "epump4"),
            "relative_path": "factory_raw/epump2/prototype/秒表/"
                              "E-pump_BB6L0460_180652204_18012026_000149.tdms.zst",
        })

    adapter = ManifestAdapter(meta)
    raw = adapter.get("epump2", "BB6L0460")
    assert raw.startswith("factory_raw/epump2/"), raw
    assert "/epump4/factory_raw/" not in raw, raw
    path, status = TdmsLocator(factory / "epump4", adapter).resolve(
        {"line": "epump2", "sn": "BB6L0460", "sample_id": "BB6L0460_up"}
    )
    assert status == "registered"
    assert path == cross_line, path
