from pathlib import Path

from forvia_train_v2.backend.run_manager import (
    cleanup_feature_artifacts,
    cleanup_model_artifacts,
)


def _config(root: Path) -> dict:
    return {
        "line_name": "epump3",
        "model": {"model_name": "demo"},
        "results_path": str(root),
    }


def test_feature_cleanup_removes_feature_outputs_only(tmp_path):
    results = tmp_path / "epump3_demo"
    (results / "dataset_csv").mkdir(parents=True)
    (results / "dataset_csv" / "features_batch_1.csv").write_text("x", encoding="utf-8")
    (results / "feature_selection").mkdir()
    (results / "model.pkl").write_text("model", encoding="utf-8")

    cleanup_feature_artifacts(_config(tmp_path))

    assert not (results / "dataset_csv").exists()
    assert not (results / "feature_selection").exists()
    assert (results / "model.pkl").exists()


def test_model_cleanup_preserves_features_and_removes_model_outputs(tmp_path):
    results = tmp_path / "epump3_demo"
    (results / "dataset_csv").mkdir(parents=True)
    (results / "dataset_csv" / "sample_view.csv").write_text("x", encoding="utf-8")
    (results / "eval").mkdir()
    (results / "eval" / "confusion.png").write_text("image", encoding="utf-8")
    (results / "model.pkl").write_text("model", encoding="utf-8")

    cleanup_model_artifacts(_config(tmp_path))

    assert (results / "dataset_csv" / "sample_view.csv").exists()
    assert not (results / "eval").exists()
    assert not (results / "model.pkl").exists()
