from ml.training.results import is_result_current


def test_project_timestamp_after_training_does_not_hide_current_result():
    successful = {"id": "old", "finished_at": "2026-07-19T10:00:00"}
    latest = {"id": "old", "status": "succeeded"}
    assert is_result_current("2026-07-19T10:01:00", successful, latest) is True


def test_newer_run_invalidates_old_result_until_success():
    successful = {"id": "old", "finished_at": "2026-07-19T10:00:00"}
    latest = {"id": "new", "status": "running"}
    assert is_result_current("2026-07-19T09:00:00", successful, latest) is False


def test_latest_successful_run_is_current():
    successful = {"id": "new", "finished_at": "2026-07-19T10:00:00"}
    latest = {"id": "new", "status": "succeeded"}
    assert is_result_current("2026-07-19T09:00:00", successful, latest) is True
