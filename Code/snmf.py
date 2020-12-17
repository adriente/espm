import numpy as np
import matplotlib.pyplot as plt
import time

# For the unused laplacian function
#from scipy.sparse import lil_matrix, block_diag

from utils_V2 import Gaussians


class SNMF:
    """
    Matrix factorization algorithm adapted for EDXS data. The data are learned with the following model :
    Y = (GP + B) A where Y is the data, P and B are matrices modelling the spectra and A is the abundance of the modelled spectra.
    The G matrix is built from a database of X-ray lines. (Work in progress)
    The data are learned by minimizing a log likelihood function through multiplicative updates for A and P and through gradient descent for B. The multiplicatives updates for A and P and the proximal method for B ensure the positivity of A, P and B. 
    A sum-to-one constraint is imposed to the coefficients of A through a Lagrange multiplier.
    It is possible to penalize a part of the abundances of A through a regularization.
    When the regularization is activated, the algorithm is decomposed in two steps : a regularized step and a data fitting step.
    """

    def __init__(self, max_iter = 10000, tol = 1e-4, b_tol = 1.0, step_a = True, step_p = True, mu_sparse=0., eps_sparse = 1.0, num_phases=2, edxs_model=None, debug=False, brstlg_pars = None, init_a = None, init_p = None, init_spectrum = None):
        """
        Create an instance of the SNMF algorithm. All the relevant parameters are defined here or through the set_hypars or set_init_pars functions.
        :num_phases: Number of phases to estimate (int)
        :max_iter: maximum number of iterations (int)
        :tol: A decrease of the loss function below tol stops the algorithm (float)
        :b_tol: A decrease of the loss function below b_tol starts the learning of B (float)
        :step_a: Allows steps in A to be taken (bool)
        :step_p: Allows steps in P to be taken (bool)
        :mu_sparse: strength of the particles regularisation (float > 0)
        :eps_sparse: Determines the slope at 0 of the particles regularisation (float > 0)
        :edxs_model: EDXS_Model object generated from tables, contains the g_matr. In the future it will contain the absorption coefficient. (EDXS_Model)
        :brstlg_pars: list of input parameters for initalisation of the bremsstrahlung. (list of float or np.array of float)
        :init_a: Input the initial values of a (np.ndarray)
        :init_p: Input the initial values of p (np.ndarray)
        :init_spectrum: initialize the first phase to the input spectrum. Useful to set the first phase as the matrix for the particles regularization.
        :debug: For monitoring the steps taken by the algorithm --> slows down the algorithm
        """
        
        ################
        # hyper params #
        ################
        
        self.p_ = num_phases
        self.max_iter = max_iter
        self.tol = tol
        self.b_tol = b_tol
        # Particles regularization coefficients
        self.mu_sparse = mu_sparse
        self.eps_sparse= eps_sparse


        ###############
        # init params #
        ###############
        
        self.init_p = init_p
        self.init_a = init_a
        self.step_a = step_a
        self.step_p = step_p
        self.init_spectrum = init_spectrum

        # The same starting brstlg parameters are used for all the phases
        if brstlg_pars is None :
            self.b1 = -np.ones((self.p_,))
            self.b2 = np.ones((self.p_,))
            self.c0 = np.ones((self.p_,))
            self.c1 = np.ones((self.p_,))
        else : 
            self.b1 = brstlg_pars[0]*np.ones((self.p_,))
            self.b2 = brstlg_pars[1]*np.ones((self.p_,))
            self.c0 = brstlg_pars[2]*np.ones((self.p_,))
            self.c1 = brstlg_pars[3]*np.ones((self.p_,))
            self.c2 = brstlg_pars[4]*np.ones((self.p_,))
        
        ######################
        # internal variables #
        ######################
        
        self.debug = debug
        # The mask and stop are used if mu_sparse!=0 in the data fitting step
        self.sparse_mask = None
        self.sparse_stop = None
        # The steps in B are taken after the tol_b criterin is met
        self.step_b = False

        # Lipschitz constant associated with the different parameters of the brstlg
        self.eta_b1 = 1.
        self.eta_b2 = 1.
        self.eta_c0 = 1.
        self.eta_c1 = 1.
        self.eta_c2 = 1.

        # parameters for the line-search algorithm
        self.beta = 1.5
        self.alpha = 1.2
        
        # Lagrange multiplier to apply the simplex constraint
        self.lambda_s = None 

        # EDXS model loading
        self.g_matr = edxs_model.g_matr
        self.bremsstrahlung = edxs_model.bremsstrahlung
        self.E = edxs_model.x

        # the activations matrix
        self.a_matr = None
        # intermediate matrix for log regularization
        self.fixed_a = None
        # the matrix containing the intensity of every signal in the G matrix, for every phase
        self.p_matr = None
        # The matrix of the learned brstlg model
        self.b_matr = None
        # D = GP + B (cached for efficiency reasons)
        self.d_matr = None
        # the data matrix that we will try to fit
        self.x_matr = None
        # original shape of the data matrix (kept track of because X matrix is immediately reshaped when loaded)
        self.x_shape = None
        # to keep track of the number of iterations when fit is called multiple times
        self.num_iterations = None


        #############################
        # Troubleshooting variables #
        #############################

        # for debugging purposes
        self.eta_list = []
        self.base_losses = []
        self.losses = []
        self.a_norm = []
        self.p_norm = []
        self.b_norm = []

    #################
    # Params Setter #
    #################

    def set_hypars(self,num_phases=None,mu_sparse=None,eps_sparse=None,tol = None, b_tol = None, max_iter=None) :
        """
        Setter of the hyper parameters of the algorithm. It's main use is for the cross_validation class. It can be used to reduce the number of parameters in the __init__ function. 
        """
        if num_phases is None :
            pass
        else :
            self.num_phases = num_phases
        if mu_sparse is None :
            pass
        else :
            self.mu_sparse = mu_sparse
        if eps_sparse is None :
            pass
        else : 
            self.eps_sparse = eps_sparse
        if tol is None :
            pass
        else :
            self.tol = tol
        if b_tol is None :
            pass
        else :
            self.b_tol = b_tol
        if max_iter is None :
            pass
        else :
            self.max_iter = max_iter

    def set_init_pars(self,init_a=None,init_p=None,b1 = None, b2 = None, c0 = None, c1 = None, c2 = None) :
        """
        Setter of the init parameters of the algorithm. It's main use is for the cross_validation class. It can be used to reduce the number of parameters in the __init__ function. 
        """
        if init_a is None :
            pass
        else : 
            self.init_a = init_a
        if init_p is None :
            pass
        else :
            self.init_p = init_p
        if b1 is None :
            pass
        else : 
            self.b1 = b1
        if b2 is None :
            pass
        else : 
            self.b2 = b2
        if c0 is None :
            pass
        else : 
            self.c0 = c0
        if c1 is None :
            pass
        else : 
            self.c1 = c1
        if c2 is None :
            self.c2 = c2

    ############################
    # Loss function estimators #
    ############################

    def _eval_function(self,d_matr = None):
        """
        Evaluates the loss function that is being optimized.
        The d_matr argument is used for evaluation of temporary values of the d_matr that are not stored in self.d_matr
        """
        if d_matr is None :
            d_matr = self.d_matr
        
        if (self.mu_sparse != 0.0) and not(self.sparse_stop)  :
            sparse_loss = self.mu_sparse*np.sum(np.log(self.a_matr[1:,:]+self.eps_sparse))
        else : 
            sparse_loss = 0.0

        return self._base_loss(d_matr) + sparse_loss

    def _base_loss(self,d_matr = None):
        """
        Evaluates the data-fitting part of the loss function.
        The d_matr argument is used for evaluation of temporary values of the d_matr that are not stored in self.d_matr
        """
        if d_matr is None :
            d_matr = self.d_matr

        d_a=d_matr@self.a_matr
        # X_ij * log[(d@a)_ij] 
        x_log =np.einsum("ij,ij",self.x_matr,np.log(d_a))
        # Lagrangian multiplier for the application of the simplex constraint.
        lambda_loss = (self.lambda_s * (np.sum(self.a_matr,axis=0)-1)).sum()
        return d_a.sum() - x_log + lambda_loss

    #######################
    # Dichotomy functions #
    #######################

    # There is probably a better way to implement the dichotomy via 3 functions : 
    # 1) dichotomy algorithm, 2) initialisation of dichotomy values, 3) functions of the equations to solve 
    
    def dichotomy_simplex (self,num,denum,tol) :
        """
        Function to solve the num/(x+denum) -1 = 0 equation. Here, x is the Lagragian multiplier which is used to apply the simplex constraint.
        The first part consists in finding a and b such that num/(a+denum) -1 > 0 and num/(b+denum) -1  < 0. (line search)
        In this function, the largest root is found.
        The second part applies the dichotomy algorithm to solve the equation.
        In the future a vectorized version of Dekker Brent could be implemented.
        """
        # The function has exactly one root at the right of the first singularity (the singularity at min(denum))
        # So a and b are set to -min(denum) plus an offset.
        f_div = np.min(np.where(num!=0,denum,np.inf),axis=0) # There are several min values in the case of particle regularization. In the case where there are exact 0 in num, the min of denum needs to be changed to the next nearest min.
        a_off = 100*np.ones(num.shape[1])
        b_off = 0.01*np.ones(num.shape[1])
        a=-f_div*np.ones(num.shape[1]) + a_off
        b=-f_div*np.ones(num.shape[1]) + b_off

        # Search for a elements which give positive value
        constr = (np.sum(num/(a+denum),axis=0) -1)
        while np.any(constr<= 0) :
            # We exclude a elements which give positive values
            # We use <= because constr == 0 is problematic.
            constr_bool = constr <= 0
            # Reducing a will ensure that the function will diverge toward positive values
            a_off[constr_bool]/=1.2
            a = -f_div*np.ones(num.shape[1]) + a_off
            constr = (np.sum(num/(a+denum),axis=0) -1)
            
        # Search for b elements which give negative values
        constr = (np.sum(num/(b+denum),axis=0) -1)
        while np.any(constr>= 0) :
            # We exclude b elements which give negative values
            constr_bool = constr >= 0
            # increasing b will ensure that the function will converge towards negative values
            b_off[constr_bool]*=1.2
            b = -f_div*np.ones(num.shape[1]) + b_off
            constr = (np.sum(num/(b+denum),axis=0) -1)

        # Dichotomy algorithm to solve the equation
        while np.any(np.abs(b-a) > tol):
            new = (a+b)/2
            # if f(a)*f(new) <0 then f(new) < 0 --> store in b
            minus_bool = (np.sum(num/(a+denum),axis=0) -1)*(np.sum(num/(new+denum),axis=0) -1) < 0
            # if f(a)*f(new) > 0 then f(new) > 0 --> store in a
            plus_bool = (np.sum(num/(a+denum),axis=0) -1)*(np.sum(num/(new+denum),axis=0) -1) > 0
            b[minus_bool] = new[minus_bool]
            a[plus_bool] = new[plus_bool]

        return (a+b)/2

    
    def dichotomy_brstlg (self,u,v,tol) :
        """
        Function to solve the (u/(1+2x))**2 -4*v - 16*x = 0 equation. Here, x is the solution to the proximal operator which ensures positive bremsstrahlung values.
        In the future a vectorized version of Dekker Brent could be implemented.
        """
        # x value for which the function is always positive
        lambda_pos = - v / 4 
        # x value for which the function is always negative
        lambda_neg = 1/16 + np.absolute(u)/2 + np.absolute(v)/4
        # Dichotomy algorithm to solve the equation
        while np.any(np.abs(lambda_pos-lambda_neg) > tol) :
            new = (lambda_neg+lambda_pos)/2
            expr = (np.power((u/(1+2*lambda_pos)),2) -4*v -16*lambda_pos)*(np.power((u/(1+2*new)),2) -4*v -16*new )
            lambda_pos[expr>0] = new[expr>0]
            lambda_neg[expr<0] = new[expr<0]
        return lambda_neg

    #######################
    # Steps in A, P and B #
    #######################

    def _make_step_a(self,mask=None):
        """
        Multiplicative step in A.
        The main terms are calculated first.
        With mu_sparse = 0, the steps in A are calculated once. For mu_sparse != 0, the steps in A are calculated first with particle regularization. Then only the entries allowed by the mask are calculaed, without particle regularization. 
        To calculate the regularized step, we make a linear approximation of the log.
        """
        # Update of the d_matr (maybe it should be done in a dedicated function for clarity)
        self.d_matr=self.g_matr@self.p_matr + self.b_matr
        # Multiplicative update numerator U and denominator V
        d_a=(self.d_matr@self.a_matr)
        U=self.d_matr.T @ (self.x_matr.clip(min=1e-150) / d_a)
        V=self.d_matr.sum(axis=0)
        # Reset of the Lagrangian multiplier (Maybe non-necessary .. ?)
        self.lambda_s=np.zeros((self.x_shape[1]*self.x_shape[0],))

        if self.mu_sparse !=0 :
            # Regularized part of the algorithm
            if mask is None : 
                # In the linear approximation, the slope is constant. We modifiy this slope to approximate the log every 10 iterations.
                # The number of iterations in between two slope changes is arbitrary.
                if self.num_iterations%10 == 0 :
                    self.fixed_a = self.a_matr.copy()
                # Vectorized version of the regularization
                vec_sparse = np.array([0]+(self.p_-1)*[self.mu_sparse])
                denum = V[:,np.newaxis]+vec_sparse[:,np.newaxis]/(self.fixed_a + self.eps_sparse)
                # Lagragian multiplier
                self.lambda_s = self.dichotomy_simplex(self.a_matr*U,denum,0.001)
                # A update (regularized)
                self.a_matr=self.a_matr/(V[:,np.newaxis]+ vec_sparse[:,np.newaxis]/(self.fixed_a + self.eps_sparse)+self.lambda_s)*U
            else :
                # Lagragian multiplier
                self.lambda_s = self.dichotomy_simplex(self.a_matr*U,V[:,np.newaxis],0.001)
                # Update the entry that did not meet the sparsity requirements
                n_mask = np.invert(mask)
                # A update (masked)
                self.a_matr[n_mask]=self.a_matr[n_mask]/(V[:,np.newaxis]+self.lambda_s)[n_mask]*U[n_mask]
        
        else :
            # Lagragian multiplier
            self.lambda_s = self.dichotomy_simplex(self.a_matr*U,V[:,np.newaxis],0.001)
            # A update (not regularized)
            self.a_matr=self.a_matr/(V[:,np.newaxis]+self.lambda_s)*U
        

    def _make_step_p(self):
        """
        Multiplicative step in P.
        """
        
        d_a=(self.d_matr@self.a_matr)
        term1=self.g_matr.T @ (self.x_matr.clip(min=1e-150) / (d_a)) @ self.a_matr.T #The data are clipped to avoid issues during computation
        term2=self.g_matr.sum(axis=0)[:,np.newaxis]@self.a_matr.sum(axis=1)[:,np.newaxis].T
        # P update
        self.p_matr=self.p_matr/term2*term1
        # D update
        self.d_matr=self.g_matr@self.p_matr + self.b_matr

    def  _make_step_b(self) :
        """
        1) Gradient step in c0, c1 and c2
        2) Gradient step in b1 and b2
        The gradient step sizes are determined through line search.
        The positivity of the model is ensured by the use of the proximal method
        """
        ################
        #  c0, c1 & c2 #
        ################

        # Gradients
        dLdB = self.dLdB()
        grad_c0 = np.einsum("i...,i...",dLdB,self.calc_dc0())
        grad_c1 = np.einsum("i...,i...",dLdB,self.calc_dc1())
        grad_c2 = np.einsum("i...,i...",dLdB,self.calc_dc2())
        # Current estimate of the loss function, cached for efficiency reasons
        cur_loss = self._eval_function()
        # Initialisation of the new values of the parameters
        # The parameters are clipped to 1e-8 to ensure positivity
        c0_tilde = (self.c0 - 1/self.eta_c0*grad_c0).clip(min=1e-8) 
        c1_tilde = (self.c1 - 1/self.eta_c1*grad_c1).clip(min=1e-8)
        c2_tilde = (self.c2 - 1/self.eta_c2*grad_c2).clip(min=1e-8)
        d_tilde = self.g_matr@self.p_matr + self.calc_b(c0=c0_tilde,c1 = c1_tilde,c2 = c2_tilde)
        func_c = self._eval_function(d_matr=d_tilde)
        # Line search algorithm to determine gradient step size
        while func_c > (cur_loss + self.f_c_quadr(c0_tilde,c1_tilde,c2_tilde,grad_c0,grad_c1,grad_c2)) :
            # At each step the eta increase, which reduces the step size
            self.eta_c0*=self.alpha
            self.eta_c1*=self.alpha
            self.eta_c2*=self.alpha
            c0_tilde = (self.c0 - 1/self.eta_c0*grad_c0).clip(min=1e-8)
            c1_tilde = (self.c1 - 1/self.eta_c1*grad_c1).clip(min=1e-8)
            c2_tilde = (self.c2 - 1/self.eta_c2*grad_c2).clip(min=1e-8)
            d_tilde = self.g_matr@self.p_matr + self.calc_b(c0=c0_tilde,c1=c1_tilde,c2=c2_tilde)
            func_c = self._eval_function(d_matr=d_tilde)

        # To avoid being stuck with a too small step size 
        self.eta_c0/= self.beta
        self.eta_c1/= self.beta
        self.eta_c2/= self.beta
        # Parameters and matrices updates
        self.c0 = c0_tilde
        self.c1 = c1_tilde
        self.c2 = c2_tilde
        self.b_matr = self.calc_b()
        self.d_matr=self.g_matr@self.p_matr + self.b_matr

        #############
        #  b1 & b2  #
        #############

        # Gradients
        dLdB= self.dLdB()
        grad_b1 = np.einsum("i...,i...",dLdB,self.calc_db1())
        grad_b2 = np.einsum("i...,i...",dLdB,self.calc_db2())
        # Current estimate of the loss function, cached for efficiency reasons
        cur_loss = self._eval_function()
        
        # Initialisation of the new values of the parameters
        self.lambda_b = 0.01*np.ones((self.p_,))
        b1 = self.b1 - 1/self.eta_b1*grad_b1
        b2 = self.b2 - 1/self.eta_b2*grad_b2
        # The positivity of the model is ensured when poly_const < 0
        # If all the new values follow the constraint, the new values are accepted.
        # Otherwise the equation leading to acceptable new values is solved using dichotomy
        poly_const = b1**2 -4*b2
        if np.all(poly_const <= 0 ) :
            b1_tilde = b1.copy()
            b2_tilde = b2.copy()
        else : 
            self.lambda_b = self.dichotomy_brstlg(b1,b2,0.001)
            b1_tilde = b1/(1+2*self.lambda_b)
            b2_tilde = b2 + 4*self.lambda_b
        d_tilde = self.g_matr@self.p_matr + self.calc_b(b1=b1_tilde,b2=b2_tilde)
        func_b1 = self._eval_function(d_matr=d_tilde)

        # Line search algorithm to determine gradient step size
        while func_b1 > (cur_loss + self.f_b_quadr(b1_tilde,b2_tilde,grad_b1,grad_b2)) :
            # At each step the eta increase, which reduces the step size
            self.eta_b1*=self.alpha
            self.eta_b2*=self.alpha
            b1 = self.b1 - 1/self.eta_b1*grad_b1
            b2 = self.b2 - 1/self.eta_b2*grad_b2
            poly_const = b1**2 -4*b2
            if np.all(poly_const <= 0 ) :
                b1_tilde = b1.copy()
                b2_tilde = b2.copy()
            else : 
                self.lambda_b = self.dichotomy_brstlg(b1,b2,0.001)
                b1_tilde = b1/(1+2*self.lambda_b)
                b2_tilde = (b2 + 4*self.lambda_b).clip(min=1e-8)
            
            d_tilde = self.g_matr@self.p_matr + self.calc_b(b1=b1_tilde,b2=b2_tilde)
            func_b1 = self._eval_function(d_matr=d_tilde)
        
        # To avoid being stuck with a too small step size 
        self.eta_b1/= self.beta
        self.eta_b2/= self.beta
        # Parameters and matrices updates
        self.b1 = b1_tilde
        self.b2 = b2_tilde
        self.b_matr = self.calc_b()
        self.d_matr=self.g_matr@self.p_matr + self.b_matr

    #########################
    # B modelling functions #
    #########################

    def calc_b (self,b1=None,b2=None,c0=None,c1 = None, c2 = None) :
        """
        Function to calculate the values of B according to the model. 
        This function takes arguments to allow calculation with temporary values of the parameters.
        """
        if b1 is None : 
            b1 = self.b1
        if b2 is None : 
            b2 = self.b2
        if c0 is None : 
            c0 = self.c0
        if c1 is None :
            c1 = self.c1
        if c2 is None :
            c2 = self.c2
        return self.chapman_brstlg(b1,b2)*self.detector(c1,c2)*self.self_abs(c0)

    def chapman_brstlg ( self, b1 = None,b2 = None) :
        """
        Bremsstrahlung modelling function.
        This function takes arguments to allow calculation with temporary values of the parameters.
        See Chapman et al., 1984, J. of microscopy, vol. 136, pp. 171
        """
        if b1 is None : 
            b1 = self.b1
        if b2 is None : 
            b2 = self.b2
        return (1.0 / self.E[:,np.newaxis] + b1 + b2*self.E[:,np.newaxis])

    def detector (self,c1 = None, c2= None) :
        """
        Detector modelling function.
        This function takes arguments to allow calculation with temporary values of the parameters.
        Absorption in the dead layer * Photons not absorbed in the detector
        """
        if c1 is None :
            c1 = self.c1
        if c2 is None :
            c2 = self.c2
        return np.exp(-c2/np.power(self.E[:,np.newaxis],3))*(1 - np.exp(-c1/np.power(self.E[:,np.newaxis],3)))

    def self_abs (self,c0 = None) :
        """
        self-absorption modelling function.
        This function takes arguments to allow calculation with temporary values of the parameters.
        Phi rho z model with a constant Phi rho z function
        """
        if c0 is None : 
            c0 = self.c0
        return (np.power(self.E[:,np.newaxis],3)*(1-np.exp(-c0/np.power(self.E[:,np.newaxis],3)))/c0)

    def calc_db1 (self) :
        """
        Partial derivative of B with respect to b1
        """
        return self.detector()*self.self_abs()

    def calc_db2 (self) :
        """
        Partial derivative of B with respect to b2
        """
        return self.E[:,np.newaxis]*self.detector()*self.self_abs()

    def calc_dc0(self) :
        """
        Partial derivative of B with respect to c0
        """
        return self.chapman_brstlg()*self.detector()*np.power(self.E[:,np.newaxis],3)/np.power(self.c0,2)*(np.exp(-self.c0/np.power(self.E[:,np.newaxis],3)) - 1 + np.exp(-self.c0/np.power(self.E[:,np.newaxis],3))*self.c0/np.power(self.E[:,np.newaxis],3))
    
    def calc_dc1(self) :
        """
        Partial derivative of B with respect to c1
        """
        return self.chapman_brstlg()*self.self_abs()*np.exp(-self.c2/np.power(self.E[:,np.newaxis],3))*np.exp(-self.c1/np.power(self.E[:,np.newaxis],3))/np.power(self.E[:,np.newaxis],3)

    def calc_dc2(self) :
        """
        Partial derivative of B with respect to c2
        """
        return - self.chapman_brstlg()*self.self_abs()*np.exp(-self.c2/np.power(self.E[:,np.newaxis],3))*(1 - np.exp(-self.c1/np.power(self.E[:,np.newaxis],3)))/np.power(self.E[:,np.newaxis],3)

    def dLdB (self) :
        """
        Partial derivative of L with respect to B
        """
        return - (self.x_matr.clip(min=1e-150) / (self.d_matr@self.a_matr)) @ self.a_matr.T + np.sum(self.a_matr,axis=1)

    def f_b_quadr(self,b1_tilde,b2_tilde,grad_b1,grad_b2) :
        """
        Calculates the proximal approximation with respect to the b1 and b2 parameters of the loss function
        """
        return grad_b1@(b1_tilde-self.b1) + grad_b2@(b2_tilde - self.b2) + grad_b1@(b1_tilde - self.b1) + self.eta_b1/2 * np.linalg.norm(b1_tilde - self.b1)**2 + self.eta_b2/2 * np.linalg.norm(b2_tilde - self.b2)**2

    def f_c_quadr(self,c0_tilde,c1_tilde,c2_tilde,grad_c0,grad_c1,grad_c2) :
        """
        Calculates the proximal approximation with respect to the c0, c1 and c2 parameters of the loss function
        """
        c0_part = grad_c0@(c0_tilde-self.c0) + self.eta_c0/2 * np.linalg.norm(c0_tilde - self.c0)**2
        c1_part = grad_c1@(c1_tilde-self.c1) + self.eta_c1/2 * np.linalg.norm(c1_tilde - self.c1)**2
        c2_part = grad_c2@(c2_tilde-self.c2) + self.eta_c2/2 * np.linalg.norm(c2_tilde - self.c2)**2
        return c0_part + c1_part + c2_part

    ##################
    # Initialization #
    ##################
        
    def _initialize(self, x_matr):
        """
        Initialization of the data, matrices and parameters
        The data are flattened if necessary. The x-matr of SNMF has to be ExN, i.e. (number of energy channels) x (number of pixels).
        The a-matr is initialized at random unless init_a is specified.
        If a bremsstrahlung spectrum is specified, the b-matr update is deactivated (b_tol is set to 0) and B is set to 0.
        Otherwise, the b_matr is initialized through the values of c0, c1, c2, b1 and b2.
        The p-matr entries of the main phase are set through linear regression on the init_spectrum (if specified) or on the average spectrum.
        The other entries of the p_matr are initialized at random. 
        """
        # Data pre-processing
        # store the original shape of the input data X
        self.x_shape = x_matr.shape
        # If necessary, flattens X to a Ex(NM) matrix, such that the columns hold the raw spectra
        if x_matr.ndim == 3 :
            x_matr = x_matr.reshape((self.x_shape[0] * self.x_shape[1], self.x_shape[2])).T
            self.x_matr = x_matr.astype(np.float)
        else :
            self.x_matr = x_matr.astype(np.float)

        # Initialization of A
        if self.init_a is None :
            self.a_matr = np.random.rand(self.p_, self.x_matr.shape[1])            
        else : 
            self.a_matr = self.init_a

        # Initialization of B
        if self.bremsstrahlung :
            self.b_matr = np.zeros((self.g_matr.shape[0],self.p_))
            self.b_tol = 0
        else :
            # B is removed from the model

            self.b_matr = self.calc_b()
        
        #Initialization of p-matr
        # If the regularization is activated (mu_sparse != 0) it is important to correctly set the first phase so that the main phase is not penalized
        if self.init_p is None :
            # All phases are initialized at random and the first phase will be overwritten
            self.p_matr= np.random.rand(self.g_matr.shape[1],self.p_)
            # If a bremsstrahlung is specified, it is added to the g_matr and therefore should not be subtracted for the linear regression
            if self.bremsstrahlung :
                 # Average spectrum initialization without b_matr model 
                if self.init_spectrum is None :
                    avg_sp = np.average(self.x_matr,axis=1)
                    self.p_matr[:,0] = (np.linalg.inv(self.g_matr.T@self.g_matr)@self.g_matr.T@avg_sp).clip(min=1e-8)
                # init_spectrum initialization without b_matr model
                else : 
                    self.p_matr[:,0] = (np.linalg.inv(self.g_matr.T@self.g_matr)@self.g_matr.T@self.init_spectrum).clip(min=1e-8)
            # If no bremsstrahlung spectrum is specified the b-matr is substracted from the linear regression to get correct values for p_matr (Linear regression on the gaussians only)
            else :
               # Average spectrum initialization with b_matr model
                if self.init_spectrum is None :
                    avg_sp = np.average(self.x_matr,axis=1)
                    self.p_matr[:,0] = (np.linalg.inv(self.g_matr.T@self.g_matr)@self.g_matr.T@(avg_sp-self.b_matr[:,0])).clip(min=1e-8)
                # init_spectrum initialization with b_matr model
                else : 
                    self.p_matr[:,0] = (np.linalg.inv(self.g_matr.T@self.g_matr)@self.g_matr.T@(self.init_spectrum - self.b_matr[:,0])).clip(min=1e-8)
        else : 
            # If specified the p_matr is used
            self.p_matr = self.init_p
        # The linear regression of p_matr are clipped to avoid negative values
        
        # Initalization of other internal variables.
        self.d_matr = self.g_matr @ self.p_matr + self.b_matr
        self.num_iterations = 0
        self.lambda_s = np.ones((self.x_matr.shape[1],))

    #######################
    # Algorithm execution #
    #######################

    def _iteration(self):
        """
        Execute 1 iteration of the algorithm
        """
        # take step in A
        if self.step_a:
            self._make_step_a()

        # take step in P
        if self.step_p:
            self._make_step_p()

        # take step in B (only possible once b_tol was reached)
        if self.step_b : 
            self._make_step_b()

        self.num_iterations += 1


    def fit(self, x_matr):
        """
        Run the optimization algorithm to fit the input matrix, until either a maximum number of iterations is reached (max_iter) or the loss function has decreased by less than tol.
        If the particles regularization is activated, the algorithm runs a first time with a regularization (regularized step) and a second time with a mask on a_matr entries without regularization (data fitting step)
        Otherwise, the data fitting step is performed once.
        """

        # Initalization
        self._initialize(x_matr)
        # Loss function value initialization
        eval_after_p = self._eval_function()

        algo_start = time.time()
        # If mu_sparse != 0, this is the regularized step of the algorithm
        # Otherwise this is directly the data fitting step
        while True :
            start = time.time()
            # Take one step in A, P and B
            self._iteration()
            # For efficiency reasons, the convergence is checked only once every 10 iterations
            if self.num_iterations%10==1 :
                eval_before = eval_after_p
                eval_after_p = self._eval_function()

            # store some information for assessing the convergence
            # for debugging purposes
            if self.debug :
                self.eta_list.append((self.eta_b1,self.eta_b2,self.eta_c0,self.eta_c1,self.eta_c2))
                self.base_losses.append(self._base_loss())
                self.losses.append(self._eval_function())
                self.a_norm.append(np.linalg.norm(self.a_matr))
                self.p_norm.append(np.linalg.norm(self.p_matr))
                self.b_norm.append(np.linalg.norm(self.b_matr))

            # Activates the b_matr learning (Peaks in p_matr are learned first with the initial model)
            if abs(eval_before - eval_after_p) < self.b_tol :
                self.step_b = True

            # check convergence criterions
            if self.num_iterations >= self.max_iter:
                print('\nexits because max_iteration was reached')
                break

            # If there is no regularization the algorithm stops with this criterion
            # Otherwise it goes to the data fitting step
            elif abs(eval_before - eval_after_p) < self.tol:
                print('\nold function: {}, new_function: {}'.format(str(eval_before), eval_after_p))
                print('exit because of tol_function')
                if self.mu_sparse!=0 :
                    # Keep the high values of the first phase
                    main_phase_mask = self.a_matr[0,:] > 0.99
                    # Keep low values of the other phases
                    phases_mask = self.a_matr[1:,:] < 0.01
                    # Mask on a_matr entries
                    self.sparse_mask = np.vstack((main_phase_mask,phases_mask))
                    # Switch to the data fitting step
                    self.sparse_stop = True
                break
            
            elif np.isnan(eval_before - eval_after_p) :
                print("\nexit because of the presence of NaN")
                self.p_matr=self.p_matr.fill(np.nan)
                self.a_matr = self.a_matr.fill(np.nan)
                break

            elif (eval_before-eval_after_p) < 0 :
                print("\nexit because of negative decrease")
                break

            print(f"\rFinished iteration {self.num_iterations} of maximal {self.max_iter} function value "
                  f"decreased by: {eval_before-eval_after_p} taking: {time.time()-start} seconds",
                  end="", flush=True)

        # The loss function changes so it needs to be reevaluated to avoid pseudo negative decrease
        if self.sparse_stop :
            eval_before = self._eval_function()
            self._iteration()
            eval_after_p = self._eval_function()

        # Data fitting step after regularization
        while self.sparse_stop : 
            start = time.time()

            self._iteration()

            if self.num_iterations%10==1 :
                eval_before = eval_after_p
                eval_after_p = self._eval_function()

            if self.debug :
                self.eta_list.append((self.eta_b1,self.eta_b2,self.eta_c0,self.eta_c1,self.eta_c2))
                self.base_losses.append(self._base_loss())
                self.losses.append(self._eval_function())
                self.a_norm.append(np.linalg.norm(self.a_matr))
                self.p_norm.append(np.linalg.norm(self.p_matr))
                self.b_norm.append(np.linalg.norm(self.b_matr))

            if self.num_iterations >= self.max_iter:
                print('\nexits because max_iteration was reached')
                break

            elif abs(eval_before - eval_after_p) < self.tol:
                print('\nold function: {}, new_function: {}'.format(str(eval_before), eval_after_p))
                print('exit because of tol_function')
                break

            elif np.isnan(eval_before - eval_after_p) :
                print("\nexit because of the presence of NaN")
                self.p_matr=self.p_matr.fill(np.nan)
                self.a_matr = self.a_matr.fill(np.nan)
                break

            elif (eval_before-eval_after_p) < 0 :
                print("\nexit because of negative decrease")
                break

            print(f"\rFinished iteration {self.num_iterations} of maximal {self.max_iter} function value "
                  f"decreased by: {eval_before-eval_after_p} taking: {time.time()-start} seconds",
                  end="", flush=True)

        algo_time = time.time() - algo_start
        print(f"\nStopped after {self.num_iterations} iterations in {algo_time//60} minutes "
              f"and {np.round(algo_time) % 60} seconds.")

        return self.p_matr, self.a_matr

    ###########
    # Getters #
    ###########
    
    def get_phase_spectrum (self, phase_id) :
        """
        Returns the endmember of one phase
        :phase_id: index of the phase (int)
        """
        return self.d_matr[:,phase_id]

    def get_phase_bremsstrahlung (self, phase_id) :
        """
        Returns the bremsstrahlung modelling of one phase
        :phase_id: index of the phase (int)
        """
        return self.b_matr[:,phase_id]

    def get_phase_map (self, phase_id) :
        """
        Returns the abundances of one phase
        :phase_id: index of the phase (int)
        """
        return self.a_matr.T.reshape(*self.x_shape[:2], self.p_)[:, :,phase_id]

    ##############################
    # Currently unused functions #
    ##############################

    # def plot_convergence(self):
    #     """
    #     Helper function to plot the convergence curves of the algorithm
    #     """
    #     fig1 = plt.figure(figsize=(15, 3))

    #     ax1 = fig1.add_subplot(1, 6, 1)
    #     ax1.plot(self.losses)
    #     ax1.set_title('F(A,D)')
    #     ax2 = fig1.add_subplot(1, 6, 2)
    #     ax2.plot(np.maximum(np.array(self.losses[:-2]) - self.losses[-1], 0.01))
    #     ax2.set_yscale('log')
    #     ax2.set_title('F(A,D)_t - F(A,D)_T')
    #     ax3 = fig1.add_subplot(1, 6, 3)
    #     ax3.plot(self.a_update)
    #     ax3.set_title('A update')
    #     ax4 = fig1.add_subplot(1, 6, 4)
    #     ax4.plot(self.p_update)
    #     ax4.set_title('D update')
    #     ax6 = fig1.add_subplot(1, 6, 5)
    #     ax6.plot(self.a_norm)
    #     ax6.set_title('A_Norm')
    #     ax7 = fig1.add_subplot(1, 6, 6)
    #     ax7.plot(self.p_norm)
    #     ax7.set_title('D_Norm')
    #     fig1.tight_layout()
    #     plt.show()



######################
# Laplacian function #
######################

    # def _create_laplacian_a(self, n, t=0):
    #     """
    #     Helper method to create the laplacian matrix for the laplacian regularization
    #     :param n: width of the original image
    #     :return:the n x n laplacian matrix
    #     """
    #     if t==0 :
    #         #Blocks corresponding to the corner of the image (linking row elements)
    #         top_block=lil_matrix((n,n),dtype=np.float32)
    #         top_block.setdiag([4]+[6]*(n-2)+[4])
    #         top_block.setdiag(-2,k=1)
    #         top_block.setdiag(-2,k=-1)
    #         #Blocks corresponding to the middle of the image (linking row elements)
    #         mid_block=lil_matrix((n,n),dtype=np.float32)
    #         mid_block.setdiag([6]+[8]*(n-2)+[6])
    #         mid_block.setdiag(-2,k=1)
    #         mid_block.setdiag(-2,k=-1)
    #         #Construction of the diagonal of blocks
    #         list_blocks=[top_block]+[mid_block]*(n-2)+[top_block]
    #         blocks=block_diag(list_blocks)
    #         #Diagonals linking different rows
    #         blocks.setdiag(-2,k=n)
    #         blocks.setdiag(-2,k=-n)
    #     else :
            
    #         mid_blocks=[]
    #         for j in range(0,n**2,n) :
    #             mid_blocks+=[lil_matrix((n,n),dtype=np.float32)]
    #             wm=0
    #             diag_m=[]
    #             for i in range(n-1) :
    #                 wm=np.exp(-np.linalg.norm(self.x_matr[:,i+j]-self.x_matr[:,i+j+1])/t)
    #                 #wm=t*np.sin(MetricsUtils.spectral_angle(self.x_matr[:,i+j],self.x_matr[:,i+j+1]))
    #                 diag_m+=[-2*wm]
    #             mid_blocks[-1].setdiag(diag_m,k=1)
    #             mid_blocks[-1].setdiag(diag_m,k=-1)
            
    #         blocks=block_diag(mid_blocks)
            
    #         ws=0
    #         diag_s=[]
    #         for i in range(n**2-n) :
    #             ws=np.exp(-np.linalg.norm(self.x_matr[:,i]-self.x_matr[:,i+n])/t)
    #             #ws=t*np.sin(MetricsUtils.spectral_angle(self.x_matr[:,i],self.x_matr[:,i+n]))
    #             diag_s+=[-2*ws]
    #         blocks.setdiag(diag_s,k=n)
    #         blocks.setdiag(diag_s,k=-n)

    #         sum_diag=-1*blocks.sum(axis=0)
    #         blocks.setdiag(sum_diag.tolist()[0])

    #     return blocks

######################
# Laplacian function #
######################