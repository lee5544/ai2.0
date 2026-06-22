#!/usr/bin/env python3
"""Audit all etilt1 CSV files for dada/kaka-related labels and notes."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


TERMS = {
    "哒哒": re.compile(r"哒哒"),
    "咔咔": re.compile(r"咔咔"),
    "咚咚": re.compile(r"咚咚"),
    "dada": re.compile(r"dada", re.IGNORECASE),
    "kaka": re.compile(r"kaka", re.IGNORECASE),
    "click": re.compile(r"click", re.IGNORECASE),
    "clack": re.compile(r"clack", re.IGNORECASE),
}
SN_RE = re.compile(r"(?<![A-Z0-9])([0-9]{2}[A-Z]{3}[0-9]{4})(?![A-Z0-9])")


def read_csv(path: Path) -> tuple[pd.DataFrame, str]:
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return (
                pd.read_csv(
                    path,
                    dtype=str,
                    keep_default_na=False,
                    on_bad_lines="skip",
                    encoding=encoding,
                ),
                encoding,
            )
        except Exception as exc:
            errors.append(f"{encoding}: {exc}")
    raise RuntimeError("; ".join(errors))


def matching_terms(text: str) -> list[str]:
    return [name for name, pattern in TERMS.items() if pattern.search(text)]


def extract_sns(text: str) -> list[str]:
    return sorted(set(SN_RE.findall(text.upper())))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/Volumes/18555440521/fault/data_root/factory_raw/etilt1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
    )
    parser.add_argument("--date", default="20260608")
    args = parser.parse_args()

    csv_paths = sorted(
        path for path in args.root.rglob("*.csv") if not path.name.startswith(".")
    )
    summaries: list[dict[str, object]] = []
    matches: list[dict[str, object]] = []

    for path in csv_paths:
        relative_path = str(path.relative_to(args.root))
        try:
            frame, encoding = read_csv(path)
        except Exception as exc:
            summaries.append(
                {
                    "csv_path": relative_path,
                    "rows_total": 0,
                    "matched_rows": 0,
                    "unique_sn_count": 0,
                    "terms": "",
                    "encoding": "",
                    "read_error": str(exc),
                }
            )
            continue

        file_sns: set[str] = set()
        file_terms: set[str] = set()
        matched_count = 0
        for index, row in frame.iterrows():
            populated = [
                f"{column}={value}"
                for column, value in row.items()
                if str(value).strip()
            ]
            row_text = " | ".join(populated)
            terms = matching_terms(row_text)
            if not terms:
                continue

            sns = extract_sns(row_text)
            matched_count += 1
            file_sns.update(sns)
            file_terms.update(terms)
            matches.append(
                {
                    "csv_path": relative_path,
                    "row_number": int(index) + 2,
                    "sn": "|".join(sns),
                    "terms": "|".join(terms),
                    "matched_row": row_text,
                }
            )

        summaries.append(
            {
                "csv_path": relative_path,
                "rows_total": len(frame),
                "matched_rows": matched_count,
                "unique_sn_count": len(file_sns),
                "terms": "|".join(sorted(file_terms)),
                "encoding": encoding,
                "read_error": "",
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = (
        args.output_dir / f"etilt1_dada_kaka_csv_summary_{args.date}.csv"
    )
    matches_path = (
        args.output_dir / f"etilt1_dada_kaka_csv_matches_{args.date}.csv"
    )
    summary_frame = pd.DataFrame(summaries)
    matches_frame = pd.DataFrame(matches)
    summary_frame.to_csv(summary_path, index=False, encoding="utf-8-sig")
    matches_frame.to_csv(matches_path, index=False, encoding="utf-8-sig")

    matched_files = summary_frame[summary_frame["matched_rows"] > 0]
    unique_sns: set[str] = set()
    for value in matches_frame.get("sn", pd.Series(dtype=str)):
        unique_sns.update(item for item in str(value).split("|") if item)

    print(f"csv_files={len(summary_frame)}")
    print(f"read_errors={(summary_frame['read_error'] != '').sum()}")
    print(f"matched_files={len(matched_files)}")
    print(f"matched_rows={int(summary_frame['matched_rows'].sum())}")
    print(f"unique_sns={len(unique_sns)}")
    print(f"summary={summary_path.resolve()}")
    print(f"matches={matches_path.resolve()}")
    if not matched_files.empty:
        print("\nmatched_files_detail:")
        print(
            matched_files[
                ["csv_path", "rows_total", "matched_rows", "unique_sn_count", "terms"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
