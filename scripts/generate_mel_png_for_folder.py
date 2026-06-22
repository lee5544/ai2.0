#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate up/down mel PNG files for TDMS files in a folder.")
    parser.add_argument("--folder", required=True, help="Folder containing TDMS files.")
    parser.add_argument("--model-dir", required=True, help="Model result directory containing runtime/.")
    parser.add_argument("--line", default="epump2", help="Line name for TDMS parsing.")
    parser.add_argument("--prediction-csv", default="", help="Optional prediction CSV for titles.")
    parser.add_argument("--output-dir", default="", help="Output PNG directory.")
    return parser.parse_args()


def _mel_db(librosa, x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return np.empty((0, 0), dtype=np.float32), np.empty(0), np.empty(0)
    n_fft = max(32, int(sr // 78))
    hop_length = max(1, n_fft // 2)
    n_mels = 64
    mel = librosa.feature.melspectrogram(
        y=x,
        sr=int(sr),
        n_mels=n_mels,
        fmax=int(sr) // 2,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    times = librosa.frames_to_time(np.arange(mel_db.shape[1]), sr=int(sr), hop_length=hop_length)
    freqs = librosa.mel_frequencies(n_mels=n_mels, fmax=int(sr) // 2)
    return mel_db, times, freqs


def main() -> None:
    args = _parse_args()
    folder = Path(args.folder).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else folder / f"mel_png_{model_dir.name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime_dir = model_dir / "runtime"
    feature_dir = runtime_dir / "features"
    sys.path.insert(0, str(runtime_dir))
    sys.path.insert(0, str(feature_dir))

    from tdms_read import read_tdms  # noqa: E402
    from extract_features_v5 import librosa, process_data  # noqa: E402

    pred_map: dict[str, dict] = {}
    pred_path = Path(args.prediction_csv).expanduser() if args.prediction_csv else folder / f"{model_dir.name}_predictions.csv"
    if pred_path.exists():
        pred_df = pd.read_csv(pred_path, encoding="utf-8-sig")
        pred_map = {str(row.get("filename") or ""): row for row in pred_df.to_dict(orient="records")}

    written: list[Path] = []
    errors: list[tuple[str, str, str]] = []
    for tdms_path in sorted(folder.glob("*.tdms")):
        try:
            data = read_tdms(tdms_path, line=args.line)
            sr = int(data.get("sampling_rate") or 20000)
            up = process_data(data["up_data"], sr=sr, cutoff_low=20, cutoff_high=None)
            down = process_data(data["down_data"], sr=sr, cutoff_low=20, cutoff_high=None)

            pred = pred_map.get(tdms_path.name, {})
            up_label = str(pred.get("up") or "") if pred else ""
            down_label = str(pred.get("down") or "") if pred else ""

            fig, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
            for ax, signal, title in [
                (axes[0], up, f"UP  {up_label}"),
                (axes[1], down, f"DOWN  {down_label}"),
            ]:
                mel_db, times, freqs = _mel_db(librosa, signal, sr)
                if mel_db.size == 0:
                    ax.text(0.5, 0.5, "empty signal", ha="center", va="center")
                    continue
                extent = [
                    float(times[0]) if times.size else 0.0,
                    float(times[-1]) if times.size else 0.0,
                    float(freqs[0]),
                    float(freqs[-1]),
                ]
                im = ax.imshow(mel_db, origin="lower", aspect="auto", extent=extent, cmap="magma")
                ax.set_title(title)
                ax.set_ylabel("Mel freq (Hz)")
                fig.colorbar(im, ax=ax, format="%+2.0f dB")
            axes[-1].set_xlabel("Time (s)")
            fig.suptitle(
                f"{tdms_path.name} | SN={data.get('sn')} | REF={data.get('reference')} | SR={sr}"
            )
            out_path = output_dir / f"{tdms_path.stem}_mel.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            written.append(out_path)
        except Exception as exc:  # noqa: BLE001
            errors.append((tdms_path.name, type(exc).__name__, str(exc)))

    print(f"WROTE {len(written)} PNG files to {output_dir}")
    for path in written:
        print(path)
    if errors:
        print("ERRORS:")
        for item in errors:
            print(item)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
