import numpy as np
from msc_thesis.binning import bin_dataset_3D, bin_dataset_2D
def test_3D_binning():
    A1, A2, A3 = 100, 100, 100
    F1, F2, F3 = 2,4,5
    A = np.random.rand(A1,A2,A3)
    A_binned = bin_dataset_3D(A, (F1, F2, F3))

    # Assert output shape is correct
    assert A_binned.shape == (A1//F1, A2//F2, A3//F3), f"Incorrect binned shape"

    # Assert pixel density hasn't changed after binning
    np.testing.assert_almost_equal(np.sum(A_binned), np.sum(A)/(2*4*5)),  f"Voxel density changed when binning"

def test_2D_binning():
    A1, A2 = 100, 100
    F1, F2 = 2,4
    A = np.random.rand(A1,A2)
    A_binned = bin_dataset_2D(A, (F1, F2))

    # Assert output shape is correct
    assert A_binned.shape == (A1//F1, A2//F2), f"Incorrect binned shape"

    # Assert pixel density hasn't changed after binning
    np.testing.assert_almost_equal(np.sum(A_binned), np.sum(A)/(2*4)), f"Pixel density changed when binning"