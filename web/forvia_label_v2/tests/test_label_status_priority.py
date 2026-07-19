from web.forvia_label_v2.backend.session import LabelSession


def event(source, reason, timestamp):
    return {
        "sample_id": "SN001_up",
        "source": source,
        "reason_key": reason,
        "reason_name": reason,
        "result_key": "nok",
        "result_name": "异常",
        "timestamp": timestamp,
    }


def test_expert_status_excludes_all_operator_categories():
    events = [
        event("operator_a", "摩擦", "2026-01-01T10:00:00"),
        event("operator_b", "摩擦", "2026-01-01T11:00:00"),
        event("expert", "震颤", "2026-01-01T12:00:00"),
    ]

    statuses = LabelSession._label_status_map_from_events(events)

    assert statuses["SN001_up"] == {"key": "expert", "name": "专家标注"}
    assert LabelSession._status_counts_for_sample_ids(
        ["SN001_up"], statuses
    ) == {
        "expert": 1,
        "operator_consistent": 0,
        "operator_conflict": 0,
        "operator_single": 0,
        "unlabeled": 0,
    }


def test_status_map_keeps_employee_categories_only_without_expert():
    cases = {
        "consistent": [event("operator_a", "摩擦", "2026-01-01T10:00:00"),
                       event("operator_b", "摩擦", "2026-01-01T11:00:00")],
        "conflict": [event("operator_a", "摩擦", "2026-01-01T10:00:00"),
                     event("operator_b", "震颤", "2026-01-01T11:00:00")],
        "single": [event("operator_a", "摩擦", "2026-01-01T10:00:00")],
        "none": [],
    }
    expected = {
        "consistent": "operator_consistent",
        "conflict": "operator_conflict",
        "single": "operator_single",
        "none": "unlabeled",
    }

    for key, events in cases.items():
        if events:
            events = [dict(item, sample_id=f"SN_{key}_up") for item in events]
        statuses = LabelSession._label_status_map_from_events(events)
        assert statuses.get(f"SN_{key}_up", {"key": "unlabeled"})["key"] == expected[key]
