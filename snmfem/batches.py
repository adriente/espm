from scipy.sparse import lil_matrix, block_diag
import numpy as np

class Batches () :
    """
    Utility class to make data batches for CrossVal
    """
    def __init__(self,X,method="checker",n=4) :
        """
        Creates an instanc of the Batches class
        :X: Data (np.ndarray)
        :method: Method used to generate the batches (string)
        :n: Number of batches when applicable (int)
        """

        self.x_shape=X.shape
        self.x_matr = self.reshape_x(X)
        self.method = method
        self.n=n
        # Data batches
        self.batches = None
        # For each batch the adj correspond to the rest of the data on which to apply the model for cross validation
        self.adj = None
        
        
    def reshape_x(self,X) :
        """
        Flattens 3d data to 2D
        :X: Data (np.ndarray)
        """
        return X.reshape((X.shape[0] * X.shape[1], X.shape[2])).T
            
    def prepare_batches(self) :
        """
        Function to standardize the access to batches independantly of the method
        """
        if self.method == "lines" :
            lines = self.line_batches()
            self.batches = lines[0]
            self.adj = lines[1]
        elif self.method == "checker" :
            checker = self.checker_batches()
            self.batches = checker[0]
            self.adj = checker[1]
            
    def line_batches (self) :
        """
        Prepare the batches as n consecutive slices of the data. 
        The slices contain all the data
        """    
        # Checks for 
        if self.x_matr.shape[1]%self.n != 0 :
            print("n ne divise pas la taille des données")
            raise IndexError
        else : 
            batches = []
            adj = []
            for i in range(self.n) :
                first_ind = self.n*i
                last_ind = (self.n+1)*i
                # Takes the n-th slice
                adj.append(self.x_matr[:,first_ind:last_ind])
                # Takes the rest of the data
                batches.append(self.x_matr[:,0:first_ind],self.x_matr[:,last_ind:-1])
        return batches,adj
    
    def checker_vector(self,odd=True) :
        """
        Construct vectors used to select flattened data in a checker board manner 
        :odd: Selection of the checker pattern "white" or "black" (bool)
        """
        # Selecting a checker pattern on a 2D array is much more easy
        a = np.zeros(self.x_shape[:2])
        if odd :
            a[::2,::2] = 1
            a[1::2,1::2] = 1
        else : 
            a[1::2,::2] = 1
            a[::2,1::2] = 1
        return a.flatten().astype("bool")
            
            
    def checker_batches (self) :
        """
        Creates the two possibles batches (even and odd) for the checkerboard pattern and the corresponding adjacency
        """
        # vectors to apply the checkerboard pattern on flattened data
        checker_vec_odd = self.checker_vector(odd=True)
        checker_vec_even = self.checker_vector(odd = False)
        # Builds the adjacency matrix
        adj_matr = self.checker_adjacency_matrix(self.x_matr.shape[0])
        
        # For the odd batches the adjacency on even columns is selected
        adj_odd_cols=adj_matr[:,checker_vec_even]
        # The odd rows are then selected to have the correct matrix shape for multiplication
        # By selecting the even colums and odd rows we obtain the elements adjacent to the odd batch
        adj_odd = adj_odd_cols[checker_vec_odd,:]
        
        # For the even batches the adjacency on odd columns is selected
        adj_even_cols=adj_matr[:,checker_vec_odd]
        # The even rows are then selected to have the correct matrix shape for multiplication
        # By selecting the odd colums and even rows we obtain the elements adjacent to the even batch
        adj_even = adj_even_cols[checker_vec_even,:]

        # batches, adj
        return [self.x_matr[:,checker_vec_odd],self.x_matr[:,checker_vec_even]], [adj_odd,adj_even]
    
    def checker_adjacency_matrix(self,n):
        """
        The connection between pixels on a 2D image can be described with an adjacency matrix.
        This method constructs the weighted adjacency matrix of an image of size n.
        It currently only works on square images
        :n: size of the side of the 2D image (int)
        """
        #Blocks corresponding to the corner of the image (linking row elements)
        top_block=lil_matrix((n,n),dtype=np.float32)
        top_block.setdiag([1.0/3.0]*(n-2)+[0.5],k=1)
        top_block.setdiag([0.5]+[1.0/3.0]*(n-2),k=-1)
        #Blocks corresponding to the middle of the image (linking row elements)
        mid_block=lil_matrix((n,n),dtype=np.float32)
        mid_block.setdiag([0.25]*(n-2)+[1.0/3.0],k=1)
        mid_block.setdiag([1.0/3.0] + [0.25]*(n-2),k=-1)
        #Construction of the diagonal of blocks
        list_blocks=[top_block]+[mid_block]*(n-2)+[top_block]
        blocks=block_diag(list_blocks)
        #Diagonals linking different rows
        outer_diag = [1.0/3.0] + (n-2)*[0.25] + [1.0/3.0]
        blocks.setdiag([0.5]+[1.0/3.0]*(n-2)+[0.5]+outer_diag*(n-2) ,k=n)
        blocks.setdiag(outer_diag*(n-2) +[0.5]+[1.0/3.0]*(n-2)+[0.5] ,k=-n)

        return blocks.tocsr()
    
    def apply_model (self,model,batch_ind) :
        """
        Function to apply a model from one batch to another
        :model: Matrix factorization algorithm (algorithm object)
        :batch_ind: index of the batch to apply the model to (int)
        """
        if self.method == "checker" :
            # Averaging
            return (model.d_matr@model.a_matr)@self.adj[batch_ind]
        elif self.method == "lines" :
            # Linear regression
            return np.linalg.inv(model.d_matr.T@model.d_matr)@model.d_matr.T@self.adj[batch_ind]