from pathlib import Path

root = Path("forvia_label_v2")
session = (root / "backend" / "session.py").read_text(encoding="utf-8")
main = (root / "backend" / "main.py").read_text(encoding="utf-8")
front = (root / "frontend" / "index.html").read_text(encoding="utf-8")

assert '"sample_index": i' in session
assert "def _resolve_sample_index" in main
assert "sample_id=" in main
assert "encodeURIComponent(ch.sample_id)" in front
assert "row.sample_index" in front
