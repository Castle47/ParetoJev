import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_emer_matched_fallback import use_jev

LIMITS={'maximum_positive_goal_absolute_point_drop':.005,'maximum_hate_risk_absolute_point_increase':.005}


def test_mean_better_than_old_anchor_is_not_enough_when_matched_control_is_better():
    anchor=[.3]*6+[.99];matched=[.4]*6+[.99];candidate=[.35]*6+[.99]
    assert not use_jev(candidate,matched,anchor,LIMITS)
    assert not use_jev(matched,matched,anchor,LIMITS)
    assert use_jev([.41]*6+[.99],matched,anchor,LIMITS)


def test_policy_checks_every_goal_against_both_references():
    anchor=[.4]*6+[.99];matched=[.4]*6+[.99]
    candidate=[.5,.39,.5,.5,.5,.5,.99]
    assert not use_jev(candidate,matched,anchor,LIMITS)
    assert not use_jev([.5]*6+[.98],matched,anchor,LIMITS)
    assert not use_jev([.41]*6+[.99],matched,[.5]*6+[.99],LIMITS)
    with pytest.raises(ValueError):use_jev([float('nan')]*7,matched,anchor,LIMITS)
    with pytest.raises(ValueError):use_jev([.5]*6,matched,anchor,LIMITS)
