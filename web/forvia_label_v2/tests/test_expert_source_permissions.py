from data_manager.label_internal_registry import normalize_label_source_category
from web.forvia_label_v2.backend.main import _is_expert_source


def test_named_expert_source_is_expert_category():
    assert normalize_label_source_category("expert_hetao") == "expert"
    assert normalize_label_source_category("expert-zhang") == "expert"


def test_named_expert_source_is_not_downgraded_to_operator():
    source = "expert_hetao"
    assert normalize_label_source_category(source) != "operator"


def test_named_expert_source_is_allowed_by_label_permissions():
    assert _is_expert_source("expert_hetao") is True
