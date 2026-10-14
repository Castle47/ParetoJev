"""Analytic decision tests, not real benchmark evidence."""
import copy
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_multiobjective_confirmation import confirmation_gates

NAMES=['is_click','long_view','is_like','is_follow','is_comment','is_forward','is_hate']
CONFIG={'objectives':NAMES,'evaluation':{'maximum_positive_goal_absolute_point_drop':.005,'maximum_hate_risk_absolute_point_increase':.005}}


def report():
    return dict(objectives={k:{'absolute_gain':.01} for k in NAMES},positive_mean={'confidence_interval':[.001,.02]},
        per_seed={str(i):{'emer':[.4]*6+[.99],'emer_jev':[.41]*6+[.99]} for i in (7,17,29)})


def test_overall_gain_with_small_comment_tradeoff_is_not_pareto_claim():
    r=report();r['objectives']['is_comment']['absolute_gain']=-.001
    actual=confirmation_gates(r,CONFIG)
    assert actual['all_conditions'] and 'not simultaneous population Pareto' in actual['scope']


def test_no_gain_or_material_single_goal_loss_does_not_pass():
    for key in NAMES:
        r=report();r['objectives'][key]['absolute_gain']=-.006
        assert not confirmation_gates(r,CONFIG)['all_conditions']
    r=report();r['positive_mean']['confidence_interval']=[0,.02]
    assert not confirmation_gates(r,CONFIG)['all_conditions']
    r=report();r['per_seed']['17']['emer_jev']=[.4]*6+[.99]
    assert not confirmation_gates(r,CONFIG)['all_conditions']


def test_missing_feedback_coverage_is_an_error():
    r=report();r['objectives']['is_comment']['absolute_gain']=float('nan')
    with pytest.raises(ValueError,match='coverage'):confirmation_gates(r,CONFIG)
