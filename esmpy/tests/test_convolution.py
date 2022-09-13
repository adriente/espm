import numpy as np


from msc_thesis.convolution import build_noncentral_filter_weights

def test_build_noncentral_filter_weights():
    A1, A2, A3 = 100, 100, 100
    F1, F2, F3 = 1, 11, 11

    A = np.random.uniform(0,1, (A1,A2,A3))
    F_size  = (F1,F2,F3)

    filter = build_noncentral_filter_weights(A, 100000, F_size, False)

    # Assert weights add up to 1
    assert abs(np.sum(filter)-1) < 0.000001, f"Weights do not add up to almost 1"

    # Assert central weight is 0
    assert abs(filter[F1//2, F2//2,F3//2]) < 0.000001, f"Central weight is not almost 0"

    # Assert all weights are non-negative
    assert np.all(filter > 0.000000000001), f"Some weights are negative"

    # Assert output shape equals input shape
    assert filter.shape == F_size, f"Incorrect resulting filter shape. Input filter size: {F_size}, Output filter size: {filter.shape}"

