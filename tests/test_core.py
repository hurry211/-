"""Smoke tests for mathematical identities and basic output ranges."""
from pathlib import Path
import importlib.util
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def load(rel):
    path=ROOT/rel
    spec=importlib.util.spec_from_file_location(path.stem,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fft_equivalence():
    m=load('lab02_frequency/frequency_lab.py')
    x=np.zeros((8,8),float);x[2:5,3:6]=1
    assert np.max(np.abs(m.dft2_manual(x)-np.fft.fft2(x)))<1e-10


def test_shift_inverse():
    a=np.arange(64).reshape(8,8)
    assert np.array_equal(np.fft.ifftshift(np.fft.fftshift(a)),a)


def test_transfer_filter_symmetry():
    m=load('lab02_frequency/frequency_lab.py')
    for fam in ('Ideal','Butterworth','Gaussian'):
        low=m.transfer((32,32),fam,'low',8)
        high=m.transfer((32,32),fam,'high',8)
        assert np.allclose(low+high,1)
        assert np.isfinite(low).all()


def test_lhe_clahe_shape_and_range():
    m=load('lab00_local_hist/lhe_clahe_lab.py')
    x=np.tile(np.arange(32,dtype=np.uint8),(32,1))
    for y in (m.local_hist_equalization(x,5),m.clahe(x,(4,4),2.0)):
        assert y.shape==x.shape and y.dtype==np.uint8
        assert y.min()>=0 and y.max()<=255


def test_spatial_histogram():
    m=load('lab01_spatial/spatial_lab.py')
    x=np.array([[0,0,128,255],[0,128,128,255]],dtype=np.uint8)
    assert m.histogram(x).sum()==x.size
    y,_=m.equalize_manual(x)
    assert y.shape==x.shape and y.dtype==np.uint8
