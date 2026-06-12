from types import SimpleNamespace

from sdr.apps.ai.services.analysis.asvs_level import filter_parameters_for_asvs_level


def _param(parameter_id: int, level):
    return SimpleNamespace(id=parameter_id, asvs_level=level)


def test_filter_parameters_for_asvs_level_is_cumulative_and_includes_unknowns():
    params = [_param(1, 1), _param(2, 2), _param(3, 3), _param(4, None)]

    filtered, stats = filter_parameters_for_asvs_level(params, 2)

    assert [item.id for item in filtered] == [1, 2, 4]
    assert stats == {
        "before_count": 4,
        "after_count": 3,
        "excluded_by_level_count": 1,
        "unknown_level_included_count": 1,
    }


def test_filter_parameters_for_asvs_level_l3_keeps_all_known_levels():
    params = [_param(1, 1), _param(2, 2), _param(3, 3)]

    filtered, stats = filter_parameters_for_asvs_level(params, 3)

    assert [item.id for item in filtered] == [1, 2, 3]
    assert stats["excluded_by_level_count"] == 0
