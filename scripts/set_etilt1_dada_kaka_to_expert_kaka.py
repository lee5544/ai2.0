#!/usr/bin/env python3
"""Set etilt1 dada/kaka label-history records to expert kaka labels."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_manager.label_database import load_label_dataframe, replace_confirmed_labels  # noqa: E402

DEFAULT_PATH = Path("/Volumes/18555440521/fault/data_root/metadata/label_records.db")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    frame = load_label_dataframe(args.path).fillna("").astype(str)
    mask = frame["line"].eq("etilt1") & frame["reason_name"].isin(
        ["哒哒_咔咔", "哒哒/咔咔"]
    )

    print(f"matched_rows={int(mask.sum())}")
    print(f"matched_samples={frame.loc[mask, ['line', 'sn', 'sample_id']].drop_duplicates().shape[0]}")
    print(f"matched_sns={frame.loc[mask, 'sn'].nunique()}")
    print("source_before:")
    print(frame.loc[mask, "source"].value_counts().to_string())

    if not args.execute:
        print("dry_run=true")
        return

    frame.loc[mask, "reason_name"] = "咔咔"
    frame.loc[mask, "source"] = "expert"
    replace_confirmed_labels(args.path, frame.to_dict("records"))

    verify = load_label_dataframe(args.path).fillna("").astype(str)
    changed = verify["line"].eq("etilt1") & verify["reason_name"].eq("咔咔")
    old_left = verify["line"].eq("etilt1") & verify["reason_name"].isin(
        ["哒哒_咔咔", "哒哒/咔咔"]
    )
    print(f"changed_rows={int(changed.sum())}")
    print(f"old_rows_left={int(old_left.sum())}")
    print(f"changed_non_expert={int((changed & ~verify['source'].eq('expert')).sum())}")


if __name__ == "__main__":
    main()
