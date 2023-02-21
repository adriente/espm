import numpy as np
from espm.utils import create_laplacian_matrix
from espm.conf import sigmaL
from scipy.sparse.linalg import eigs

def my_laplacian_op(image):
    nx, ny = image.shape

    dx = np.concatenate([np.zeros([1, ny]), np.diff(image, axis=0)], axis=0)
    dxx = np.diff(np.concatenate([dx, np.zeros([1, ny])], axis=0), axis=0)

    dy = np.concatenate([np.zeros([nx, 1]), np.diff(image, axis=1)], axis=1)
    dyy = np.diff(np.concatenate([dy, np.zeros([nx, 1])], axis=1), axis=1)

    return -(dxx + dyy)

def test_laplacian_matrix():
    for nx in range(2, 5):
        for ny in range(2,8):

            L1 = []
            for i in range(nx):
                for j in range(ny):
                    # image = np.random.rand(nx,ny)
                    image = np.zeros([nx, ny])
                    image[i, j] = 1
                    L1.append(my_laplacian_op(image).flatten())
            L1 = np.array(L1)
            L2 = create_laplacian_matrix(nx, ny)

            np.testing.assert_array_almost_equal(L2.todense(), L1)

def test_sigma_L():
    for nx in range(4, 100, 20):
        for ny in range(5, 100, 30):
            Delta = create_laplacian_matrix(nx, ny)
            l2 = np.abs(eigs(Delta, k=1)[0][0])
            assert(l2 <= sigmaL)
