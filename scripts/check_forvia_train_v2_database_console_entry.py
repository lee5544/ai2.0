#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "forvia_train_v2" / "frontend" / "index.html"


def main():
    text = HTML.read_text(encoding="utf-8")
    required = [
        "＋ 新建数据库",
        "↻ 刷新",
        "databaseRegistry",
        "openRegisteredDatabase",
        "removeRegisteredDatabase",
        "forvia_train_v2_databases",
        "openDatabaseSubTab",
        "database-home",
        "database-registry-panel",
        "database-card-list",
        "database-path-row",
        "database-line-card",
        "toggleDatabaseLine",
        "databaseLineExpanded",
        "selectedDistributionRefs",
        "toggleDistributionRef",
        "toggleAllDistributionRefs",
        "allDistributionRefsSelected",
        "勾选全部",
        "取消勾选全部",
        "selectedDistributionStats",
        "databaseOpening",
        "正在打开数据库",
        "Forvia 数据库 v2",
        "startCreateDatabase",
        "refreshDatabaseHome",
        "openDatabaseProject",
        "updateDatabaseProject",
        "← 返回开始",
        "window.location.href='/console'",
        "databaseUi.tab=sub",
        "raw.split(':')",
        "view!=='database'&&view!=='predict'&&(!current||view==='projects')",
        "<div class=\"side-group-title\">项目</div>",
        "trainStepText(step)",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"missing database console entry hooks: {missing}"
    assert "databaseActionChoices" not in text, "database home should not keep the old three-card action list"
    assert "database-choice-grid" not in text, "database home should not keep old action-card CSS"
    assert "addDatabaseRoot" not in text, "database home should use new/create or row open/update actions"
    assert "<th>数据库</th><th>data_root</th><th>label_records.db</th><th>manifest</th><th>操作</th>" not in text, "database list should not be a single wide row table"
    assert "@click=\"openDatabaseSubTab('distribution')\"" not in text, "distribution should be shown after opening a database, not as a left sidebar entry"
    assert "🗄 配置数据库</button>" not in text, "Train sidebar should not keep database as first-level nav"
    assert '<button v-if="view!==\'database\'&&view!==\'predict\'&&(!current||view===\'projects\')" class="side-tab"' in text, "database/predict views and opened projects should not show the training project list"
    assert '<div class="train-step-console">' not in text, "train step navigation should live in the sidebar"
    assert "file:///Users/liyong" not in text, "return-home must use the served /console route, not a file URL"
    assert '<div class="topbar">\\n        <button class="mini-btn" onclick' not in text, "return-home belongs in the sidebar, not the topbar"


if __name__ == "__main__":
    main()
