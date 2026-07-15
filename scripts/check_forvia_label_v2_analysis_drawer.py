#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "forvia_label_v2" / "frontend" / "index.html"


def main():
    text = HTML.read_text(encoding="utf-8")
    required = [
        "analysisDrawerOpen",
        "analysis-drawer",
        "openAnalysis",
        "closeAnalysis",
        "currentAnalysisChannel",
        "analysisChannels",
        "selectAnalysisChannel",
        "分析对象",
        "analysisDrawerWidth",
        "startResizeAnalysisDrawer",
        "analysis-resizer",
        "高级分析",
        "analysis-drawer-mask",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"missing advanced analysis drawer hooks: {missing}"


if __name__ == "__main__":
    main()
