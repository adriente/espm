import numpy as np
# Takes a measurement tensor X and downsamples it spatially by a factor of n on along each dimension
def bin_dataset_3D(Y_vol,B_size):

    # Assert correct dimensionality
    assert len(Y_vol.shape) == 3, f"input is not three dimensional, got: {len(Y_vol.shape)} dimensions"
    assert len(B_size) == 3, f"filter is not three dimensional, got: {len(B_size)} dimensions"

    # Assert non-negative filter size
    assert B_size[0] > 0 and B_size[1] > 0 and B_size[2] > 0, f"filter shape contains non-positive dimensions, filter size: {B_size}"

    # Assert filter fits in input
    assert Y_vol.shape[0] >= B_size[0] and Y_vol.shape[1] >= B_size[1] and Y_vol.shape[2] >= B_size[2], f"filter does not fit in input, filter shape: {B_size}, input shape: {Y_vol.shape}"

    # Assert compatible filter shape
    assert Y_vol.shape[0]%B_size[0] == 0 and Y_vol.shape[1]%B_size[1] == 0 and Y_vol.shape[2]%B_size[2] == 0, f"filter shape is not compative with imput shape, filter shape: {B_size}, input shape: {Y_vol.shape}"

    # Bin 1st dimension
    Y_vol = Y_vol.reshape(Y_vol.shape[0]//B_size[0], B_size[0], Y_vol.shape[1], Y_vol.shape[2]).sum(1)

    # Bin 2nd dimension
    Y_vol = Y_vol.reshape(Y_vol.shape[0], Y_vol.shape[1]//B_size[1], B_size[1], Y_vol.shape[2]).sum(2)

    # Bin 3rd dimension
    Y_vol = Y_vol.reshape(Y_vol.shape[0], Y_vol.shape[1], Y_vol.shape[2]//B_size[2], B_size[2]).sum(3)

    # Normalize
    Y_hat_vol = Y_vol/(B_size[0]*B_size[1]*B_size[2])

    return Y_hat_vol


# Takes a measurement tensor X and downsamples it spatially by a factor of n on along each dimension
def bin_dataset_2D(M ,B_size):

    # Assert correct dimensionality
    assert len(M.shape) == 2, f"input is not two dimensional, got: {len(M.shape)} dimensions"
    assert len(B_size) == 2, f"filter is not two dimensional, got: {len(B_size)} dimensions"

    # Assert non-negative filter size
    assert B_size[0] > 0 and B_size[1] > 0, f"filter shape contains non-positive dimensions, filter size: {B_size}"

    # Assert filter fits in input
    assert M.shape[0] > B_size[0] and M.shape[1] > B_size[1], f"filter does not fit in input, filter shape: {B_size}, input shape: {M.shape}"

    # Assert compatible filter shape
    assert M.shape[0]%B_size[0] == 0 and M.shape[1]%B_size[1] == 0, f"filter shape is not compative with imput shape, filter shape: {B_size}, input shape: {M.shape}"

    # Bin 1st dimension
    M = M.reshape(M.shape[0]//B_size[0], B_size[0], M.shape[1]).sum(1)

    # Bin 2nd dimension
    M = M.reshape(M.shape[0], M.shape[1]//B_size[1], B_size[1]).sum(2)

    # Normalize
    M = M/(B_size[0]*B_size[1])

    return M