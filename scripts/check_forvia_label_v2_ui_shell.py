from pathlib import Path


html = Path("forvia_label_v2/frontend/index.html").read_text(encoding="utf-8")

required = [
    'class="app-shell"',
    'class="sidebar"',
    "上海帆古自动化",
    'class="welcome-hero"',
    "Industrial Acoustic Labeling Console",
    "sidebarCollapsed",
    "uiScale",
    "--side-w",
    "字体",
]

missing = [token for token in required if token not in html]
assert not missing, f"missing UI shell markers: {missing}"
