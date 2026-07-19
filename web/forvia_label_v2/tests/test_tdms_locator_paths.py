from web.forvia_label_v2.backend.tdms_locator import TdmsLocator


def test_data_root_manifest_path_includes_factory_raw(tmp_path):
    data_root = tmp_path / "data_root"
    relative = "epump3/selected/sample.tdms.zst"
    expected = data_root / "factory_raw" / relative
    expected.parent.mkdir(parents=True)
    expected.write_bytes(b"")

    locator = TdmsLocator(data_root)
    candidates = locator._manifest_abs_candidates(relative)

    assert expected in candidates
