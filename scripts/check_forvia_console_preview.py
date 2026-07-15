#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "forvia_console_preview.html"
MODULES = ROOT / "forvia_console_modules.js"


def main():
    text = HTML.read_text(encoding="utf-8")
    modules = MODULES.read_text(encoding="utf-8")
    required = [
        "Forvia AI2.0 Console",
        "上海帆古自动化",
        "./forvia_console_modules.js",
        "FORVIA_CONSOLE_MODULES",
        "renderNav",
        "ai-stage",
        "ai-orbit",
        "ai-core",
        "ai-scan",
        "AI 2.0",
        "window.location.href",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"missing console preview hooks: {missing}"
    assert "<iframe" not in text, "console preview must launch apps directly, not embed iframe"
    module_required = [
        "http://127.0.0.1:8012/",
        "http://127.0.0.1:8001/#database:operation",
        "配置数据库",
        "标注 Label",
        "训练 Train",
        "预测/导出",
    ]
    module_missing = [item for item in module_required if item not in modules]
    assert not module_missing, f"missing console module hooks: {module_missing}"
    assert "database-operation" not in modules
    assert "database-labels" not in modules
    assert "database-distribution" not in modules


if __name__ == "__main__":
    main()
