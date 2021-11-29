from scipy.sparse import lil_matrix, block_diag
import numpy as np

sigmaL = 8

def create_laplacian_matrix(nx, ny=None):
    """
    Helper method to create the laplacian matrix for the laplacian regularization
    :param n: width of the original image
    :return:the n x n laplacian matrix
    """
    if ny is None:
        ny = nx
    assert(nx>1)
    assert(ny>1)
    #Blocks corresponding to the corner of the image (linking row elements)
    top_block=lil_matrix((ny,ny),dtype=np.float32)
    top_block.setdiag([2]+[3]*(ny-2)+[2])
    top_block.setdiag(-1,k=1)
    top_block.setdiag(-1,k=-1)
    #Blocks corresponding to the middle of the image (linking row elements)
    mid_block=lil_matrix((ny,ny),dtype=np.float32)
    mid_block.setdiag([3]+[4]*(ny-2)+[3])
    mid_block.setdiag(-1,k=1)
    mid_block.setdiag(-1,k=-1)
    #Construction of the diagonal of blocks
    list_blocks=[top_block]+[mid_block]*(nx-2)+[top_block]
    blocks=block_diag(list_blocks)
    #Diagonals linking different rows
    blocks.setdiag(-1,k=ny)
    blocks.setdiag(-1,k=-ny)
    return blocks