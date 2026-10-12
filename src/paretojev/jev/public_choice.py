"""Validate native Choice scores without mislabeling them as event probabilities."""
import math
from .client import DecisionAPIError


def validate_choice(response, *, model, revision, options):
    if not isinstance(response,dict) or response.get('model')!=model or response.get('revision')!=revision:
        raise DecisionAPIError('Unexpected model/revision')
    if response.get('kind')!='choice' or response.get('options')!=options or response.get('image_used') is not False:
        raise DecisionAPIError('Unexpected Choice schema or modality')
    p=response.get('probabilities')
    if not isinstance(p,list) or len(p)!=len(options) or any(type(x) not in (int,float) or not math.isfinite(x) or not 0<=x<=1 for x in p):
        raise DecisionAPIError('Invalid Choice probabilities')
    if abs(sum(p)-1)>.001: raise DecisionAPIError('Choice probabilities do not sum to one')
    index=response.get('choice_index')
    if type(index) is not int or not 0<=index<len(options) or p[index]!=max(p) or response.get('choice')!=options[index]:
        raise DecisionAPIError('Choice identity inconsistent')
    elapsed=response.get('elapsed_seconds')
    if type(elapsed) not in (int,float) or not math.isfinite(elapsed) or elapsed<0:
        raise DecisionAPIError('Invalid inference time')
    return p
