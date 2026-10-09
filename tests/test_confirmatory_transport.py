import importlib.util
from pathlib import Path
import pytest


def load(monkeypatch):
    directory = Path(__file__).parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location('resume_transport_test', directory / 'resume_microlens_confirmatory_transport.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_transport_resume_does_not_accept_identity_or_certificate_errors(monkeypatch):
    m = load(monkeypatch)
    assert m.is_transient_transport({'error_type':'SSLError','error':'UNEXPECTED_EOF_WHILE_READING'})
    assert m.is_transient_transport({'error_type':'ReadTimeout','error':'timed out'})
    assert not m.is_transient_transport({'error_type':'SSLError','error':'CERTIFICATE_VERIFY_FAILED'})
    assert not m.is_transient_transport({'error_type':'DecisionAPIError','error':'model/revision changed'})
    assert not m.is_transient_transport({'error_type':'HTTPError','error':'429 Too Many Requests'})


def test_transport_contingency_cannot_expand_samples_or_budget_arbitrarily(monkeypatch):
    m = load(monkeypatch); reg = {'cohort':{'max_new_http_calls_lifetime':14000}}
    assert m.amended_limit(reg,{'original_max_attempts':14000,'extra_transport_attempts':20}) == 14020
    with pytest.raises(ValueError): m.amended_limit(reg,{'original_max_attempts':14000,'extra_transport_attempts':21})
