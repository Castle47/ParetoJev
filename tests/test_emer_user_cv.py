import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_emer_user_cv import roles_for_fold,train_only_encoding

CONFIG={'outer_folds':3,'partition_seed':2026101407,'inner_validation_modulus':5}


def test_users_never_cross_roles_and_every_row_is_tested_exactly_once():
    users=np.repeat(np.arange(600),3);coverage=np.zeros(len(users),int)
    for fold in range(3):
        roles=roles_for_fold(users,fold,CONFIG);coverage+=roles==2
        sets=[set(users[roles==i]) for i in range(3)]
        assert all(sets) and not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
        permutation=np.random.default_rng(8).permutation(len(users))
        np.testing.assert_array_equal(roles_for_fold(users[permutation],fold,CONFIG),roles[permutation])
    np.testing.assert_array_equal(coverage,np.ones(len(users),int))


def test_outer_and_inner_values_cannot_change_training_feature_statistics():
    rng=np.random.default_rng(7);x=rng.normal(size=(100,10,26)).astype('float32');x[...,0]=0
    roles=np.repeat([0,1,2],[60,20,20]);weights=np.ones(100)
    _,before=train_only_encoding(x,weights,roles)
    x[roles!=0,:,1:]=1e8
    _,after=train_only_encoding(x,weights,roles)
    assert before==after
