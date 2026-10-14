import copy
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('test_confirmation_scorer',Path(__file__).resolve().parents[1]/'deployment/autodl/score_confirmation.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


@pytest.mark.parametrize('field,value', [('weights','different-checkpoint'),('dtype','BF16'),('implementation','different-service')])
def test_confirmation_refuses_any_producer_drift(field,value):
    source={'weights':'fixed-checkpoint','dtype':'FP16','implementation':'group-v3'}
    development={'source':copy.deepcopy(source)}
    module.require_same_producer(source,development)
    source[field]=value
    with pytest.raises(ValueError,match='producer differs'):
        module.require_same_producer(source,development)
