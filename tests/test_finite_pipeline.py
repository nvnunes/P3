"""Bounded CPU preparation, exact consumption and deterministic cleanup."""
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from p3.aoSystem.finitePipeline import DirectionalPreparation
from test_finite_two_stage import coupled_model,sampling


def test_pipeline_prepares_same_float64_weights_and_releases_owner(tmp_path):
    native=coupled_model(tmp_path/'case');b=native._finite_aperture_backend()
    expected=b.prepare_science_rows((0,))[0][0]
    with DirectionalPreparation(b,(0,),1) as preparation:
        actual,_=preparation.take(0)
        np.testing.assert_array_equal(actual,expected)
        assert actual.dtype==np.float64
        with pytest.raises(RuntimeError,match='already active'):
            with DirectionalPreparation(b,(0,),1):pass
    assert b._directional_preparation is None


def test_pipeline_joins_and_clears_after_worker_failure():
    worker=[]
    def fail(fields):
        worker.append(threading.current_thread())
        raise ValueError('CPU preparation failed')
    backend=SimpleNamespace(prepare_science_rows=fail)
    with pytest.raises(ValueError,match='CPU preparation failed'):
        with DirectionalPreparation(backend,(1,2),1) as prepared:
            prepared.take(1)
    assert backend._directional_preparation is None
    assert not worker[0].is_alive()


def test_pipeline_orders_batches_and_does_not_run_device_methods():
    calls=[]
    def prepare(fields):
        calls.append(tuple(fields))
        return {f:(np.array([f],float),0.) for f in fields}
    backend=SimpleNamespace(prepare_science_rows=prepare)
    with DirectionalPreparation(backend,(2,0,3),2) as prepared:
        for field in (2,0,3):
            rows,_=prepared.take(field)
            assert rows[0]==field
    assert calls==[(2,0),(3,)]
    assert backend._directional_preparation is None


def test_pipeline_preserves_complete_residual(tmp_path):
    native=coupled_model(tmp_path/'case');sample=sampling(native)
    b=native._finite_aperture_backend()
    expected,_=b.residual_covariance(sample,0)
    with DirectionalPreparation(b,(0,),1):
        actual,_=b.residual_covariance(sample,0)
    np.testing.assert_array_equal(actual,expected)


def test_forward_view_does_not_consume_another_backends_pending_rows(tmp_path):
    from copy import copy
    native=coupled_model(tmp_path/'case');sample=sampling(native)
    b=native._finite_aperture_backend();b.residual_blocks(sample,0)
    with DirectionalPreparation(b,(0,),1) as preparation:
        view=copy(b)
        _,profile=view.residual_blocks(sample,0)
        assert not profile['science_rows_pipelined']
        _,profile=b.residual_blocks(sample,0)
        assert profile['science_rows_pipelined']


def test_pipeline_starts_next_batch_before_it_is_consumed():
    started=threading.Event();release=threading.Event();workers=[]
    def prepare(fields):
        workers.append(threading.current_thread())
        if fields==(1,):
            started.set()
            if not release.wait(2):raise RuntimeError('Test producer was not released')
        return {f:(np.array([f],float),0.) for f in fields}
    backend=SimpleNamespace(prepare_science_rows=prepare)
    try:
        with DirectionalPreparation(backend,(0,1),1) as prepared:
            prepared.take(0)
            assert started.wait(2)
            release.set()
            prepared.take(1)
    finally:release.set()
    assert all(not worker.is_alive() for worker in workers)


@pytest.mark.parametrize('batch',[0,-1,1.5])
def test_pipeline_rejects_invalid_bounds(batch):
    with pytest.raises(ValueError,match='positive integer'):
        DirectionalPreparation(SimpleNamespace(),(0,),batch)


def test_pipeline_clears_owner_when_worker_start_fails(monkeypatch):
    import p3.aoSystem.finitePipeline as pipeline
    backend=SimpleNamespace(prepare_science_rows=lambda fields:{})
    def fail(*args,**kwargs):raise RuntimeError('Cannot start worker')
    monkeypatch.setattr(pipeline.ThreadPoolExecutor,'submit',fail)
    with pytest.raises(RuntimeError,match='Cannot start worker'):
        with DirectionalPreparation(backend,(0,),1):pass
    assert backend._directional_preparation is None
