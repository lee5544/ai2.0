#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "forvia_train_v2" / "frontend" / "index.html"


def main():
    text = HTML.read_text(encoding="utf-8")
    required = [
        "项目数据库",
        "projectDatabaseRoot",
        "selectedProjectDatabase",
        "applyProjectDatabase",
        "c.data_root=db.data_root",
        "c.database={...(c.database||{}),configured:true,data_root:db.data_root",
        "c.dataset={...(c.dataset||{}),data_root:db.data_root",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"missing project database hooks: {missing}"


if __name__ == "__main__":
    main()
