from forvia_label_v2.backend.label_table import (
    SYSTEM_UNREGISTERED_REASON,
    normalize_reason_display,
)


def test_unregistered_reason_is_grouped_for_display():
    row = {"reason_key": "legacy_noise_name", "reason_name": "历史未登记名称", "reason_id": "9999"}

    assert normalize_reason_display(row)["reason_name"] == SYSTEM_UNREGISTERED_REASON


def test_registered_reason_is_not_changed():
    row = {"reason_key": "friction", "reason_name": "摩擦", "reason_id": 103}

    assert normalize_reason_display(row)["reason_name"] == "摩擦"
