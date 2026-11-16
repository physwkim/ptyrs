"""The algorithms' keyword parameters: a value of the wrong type is refused naming its keyword."""

import pytest

import ptyrs


@pytest.mark.parametrize("value, error", [("1", TypeError), (-1, OverflowError)],
                         ids=["str", "negative"])
def test_a_refused_keyword_is_named(scenario, value, error):
    scan, obj, probe, _ = scenario
    with pytest.raises(error, match="^keyword 'calc_llk': ") as e:
        ptyrs.Ptycho(scan, obj, probe).ap(1, calc_llk=value)
    cause = e.value.__cause__
    assert type(e.value) is error and type(cause) is error
    assert str(e.value) == f"keyword 'calc_llk': {cause}"
