"""Analytic selection guards: no improvement may purchase a forbidden loss."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from develop_native_bounded import choose, feasible

GUARDS=dict(long_view_max_absolute_drop=0.,positive_mean_max_absolute_drop=0.,
    click_max_absolute_drop=.005,no_hate_max_absolute_drop=.005,numerical_tolerance=1e-12)


def test_long_view_gain_cannot_override_click_hate_or_mean_guard():
    base=np.array([.4]*6+[.999])
    for index,drop in [(0,.006),(6,.006),(2,.2)]:
        points=base.copy();points[1]+=.05;points[index]-=drop
        assert not feasible(points,base,GUARDS)
    points=base.copy();points[1]-=.001;points[2]+=.1
    assert not feasible(points,base,GUARDS)


def test_zero_fallback_ties_and_primary_selection():
    base=np.array([.4]*6+[.999]);tie=base.copy();tie[2]+=.1
    candidates=[dict(validation=base.tolist()),dict(validation=tie.tolist())]
    assert choose(candidates,GUARDS)==0
    first=base.copy();first[1]+=.01
    second=base.copy();second[1]+=.02
    candidates.extend([dict(validation=first.tolist()),dict(validation=second.tolist()),dict(validation=second.tolist())])
    assert choose(candidates,GUARDS)==3


def test_nonfinite_candidate_rejected():
    base=np.array([.4]*6+[.999]);bad=base.copy();bad[1]=np.nan
    assert not feasible(bad,base,GUARDS)
