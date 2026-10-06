import numpy as np
import pytest
from sberforecast.detection import CUSUM,EWMA,robust_center_scale

@pytest.mark.parametrize("constructor",[CUSUM,EWMA])
def test_zero_residual_has_no_alarm(constructor):
    model=constructor()
    assert not any(model.update(0.0)[1] for _ in range(50))

@pytest.mark.parametrize("constructor",[CUSUM,EWMA])
def test_sustained_shift_eventually_triggers(constructor):
    model=constructor()
    assert any(model.update(4.0)[1] for _ in range(10))

@pytest.mark.parametrize("constructor",[CUSUM,EWMA])
def test_detector_is_causal(constructor):
    a,b=constructor(),constructor()
    prefix=[0.1,0.5,-0.3,0.2,1.0]
    out_a=[a.update(z) for z in prefix+[50.0]*5]
    out_b=[b.update(z) for z in prefix+[-50.0]*5]
    assert out_a[:len(prefix)]==out_b[:len(prefix)]

@pytest.mark.parametrize("constructor",[CUSUM,EWMA])
def test_nonfinite_error_rejected(constructor):
    with pytest.raises(ValueError):constructor().update(float("nan"))


def test_mad_scale_floor():
    center,scale=robust_center_scale(np.array([0.,0.,0.]),0.03)
    assert center==0 and scale==0.03
