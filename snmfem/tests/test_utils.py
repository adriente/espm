import numpy as np
import snmfem.utils as u


def test_rescale() :
    # for k small
    W = np.random.rand(27,5)
    H = np.random.rand(5,150)
    W_r, H_r = u.rescaled_DA(W,H)
    assert(np.abs(np.mean(H_r.sum(axis=0)) -1) < np.abs(np.mean(H.sum(axis=0)) -1))
    np.testing.assert_array_almost_equal(W@H,W_r@H_r)

    # for k large
    W = np.random.rand(50,10)
    H = np.random.rand(10,5)
    W_r, H_r = u.rescaled_DA(W,H)
    np.testing.assert_allclose(H_r.sum(axis=0),1) 
    np.testing.assert_array_almost_equal(W@H,W_r@H_r)