import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_multiobjective_cohort import lookup_occurrences


def test_chunk_lookup_preserves_request_order_and_handles_source_gaps():
    events=np.array([(2,10),(5,11),(8,12),(2**48+1,13),(2**48+7,14)],dtype=[('record','i8'),('value','i4')])
    wanted=np.array([2**48+7,2,8,2**48+1,5])
    got=lookup_occurrences(events,wanted,chunk_size=2)
    np.testing.assert_array_equal(got['record'],wanted)
    np.testing.assert_array_equal(got['value'],[14,10,12,13,11])
    with pytest.raises(ValueError):lookup_occurrences(events,np.array([7]),2)
    with pytest.raises(ValueError):lookup_occurrences(events,np.array([2**49]),2)


def test_source_order_corruption_is_rejected_across_chunks():
    events=np.array([(2,),(5,),(4,),(8,)],dtype=[('record','i8')])
    with pytest.raises(ValueError,match='strictly increasing'):lookup_occurrences(events,np.array([2,8]),2)


def test_user_partitions_preserve_every_occurrence_and_original_order(tmp_path):
    from audit_multiobjective_cohort import partition_by_user
    events=np.array([(2,513),(5,0),(8,514),(9,513),(13,0)],dtype=[('record','i8'),('user','i4')])
    files=partition_by_user(events,tmp_path/'partitions','source-hash')
    parts=[np.fromfile(p,dtype=events.dtype) for p in files]
    np.testing.assert_array_equal(parts[0]['record'],[5,13])
    np.testing.assert_array_equal(parts[1]['record'],[2,8,9])
    assert partition_by_user(events,tmp_path/'partitions','source-hash')==files
    with pytest.raises(ValueError):partition_by_user(events,tmp_path/'partitions','changed-source')
