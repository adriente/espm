import time
import csv
import os
import os.path
from tracemalloc import start

from sympy import true
start_time = time.time()
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
from esmpy.datasets.base import generate_spim
from esmpy.estimators import SmoothNMF
from esmpy.measures import find_min_angle, find_min_MSE, ordered_mse, ordered_mae, ordered_r2


C = 15
L = 200
P = 100**2
seed = 0

n_poisson = 300 # Average poisson number per pixel (this number will be splitted on the L dimension)

def syntheticG(L=200, C=15, seed=None):

    np.random.seed(seed=seed)
    n_el = 45
    n_gauss = np.random.randint(2, 5,[C])
    l = np.arange(0, 1, 1/L)
    mu_gauss = np.random.rand(n_el)
    sigma_gauss = 1/n_el + np.abs(np.random.randn(n_el))/n_el/5

    G = np.zeros([L,C])

    def gauss(x, mu, sigma):
        # return np.exp(-(x-mu)**2/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
        return np.exp(-(x-mu)**2/(2*sigma**2))

    for i, c in enumerate(n_gauss):
        inds = np.random.choice(n_el, size=[c] , replace=False)
        for ind in inds:
            w = 0.1+0.9*np.random.rand()
            G[:,i] += w * gauss(l, mu_gauss[ind], sigma_gauss[ind])
    return G
 
def load_toy_images():
    im1 = plt.imread("../esmpy/datasets/toy-problem/phase1.png")
    im1 = (1-np.mean(im1, axis=2)) *0.5

    im2 = plt.imread("../esmpy/datasets/toy-problem/phase2.png")
    im2 = (1-np.mean(im2, axis=2)) *0.5

    im0 = 1 - im1 - im2 

    Hdot = np.array([im0, im1, im2])

    return Hdot


def create_toy_problem(L, C, n_poisson, seed=None):
    np.random.seed(seed=seed)
    G = syntheticG(L,C, seed=seed) 
    Hdot = load_toy_images()
    K = len(Hdot)
    Hdotflat = Hdot.reshape(K, -1)
    Wdot = np.abs(np.random.laplace(size=[C, K]))
    Wdot = Wdot / np.mean(Wdot)/L
    Ddot = G @ Wdot
    Ydot = Ddot @ Hdotflat

    Y = 1/n_poisson * np.random.poisson(n_poisson * Ydot)
    shape_2d = Hdot.shape[1:]
    return G, Wdot, Ddot, Hdot, Hdotflat, Ydot, Y, shape_2d, K

from matplotlib.widgets import Slider
def plot_results(Ddot, D, Hdotflat, Hflat, n_pixel_side, Hs, Ws, G, losses):

    # avg_r2_losses = []
    # avg_ang_losses = []
    # for i in range(len(losses)):
    #     H_it = Hs[i]
    #     W_it = Ws[i]
    #     D_it = G @ W_it
    #     angles, true_inds = find_min_angle(Ddot.T, D_it.T, unique=True, get_ind=True)
    #     r2 = ordered_r2(Hdotflat, H_it, true_inds)
    #     avg_ang_losses.append(sum(angles)/len(angles))
    #     avg_r2_losses.append(sum(r2)/len(r2))
        

    fontsize = 30
    scale = 15
    aspect_ratio = 1.4
    marker_list = ["-o","-s","->","-<","-^","-v","-d"]
    mark_space = 20
    # cmap = plt.cm.hot_r    
    cmap = plt.cm.gray_r
    vmax = 1
    vmin = 0
    K = Hflat.shape[0]
    L = D.shape[0]

    angles, true_inds = find_min_angle(Ddot.T, D.T, unique=True, get_ind=True)

  

    mse = ordered_mse(Hdotflat, Hflat, true_inds)
    mae = ordered_mae(Hdotflat, Hflat, true_inds)
    r2 = ordered_r2(Hdotflat, Hflat, true_inds)
    Hflat = Hflat[true_inds,:]
    D = D[:,true_inds]
    

    fig = plt.figure(figsize = (scale/K * 3 * aspect_ratio,scale))
    gs = fig.add_gridspec(3,3)
    plt.subplots_adjust(bottom=0.15)
    ax_slide = plt.axes([0.15, 0.1, 0.65, 0.03])
    s_factor = Slider(ax_slide, 'N. Iteration',
                  0, len(Ws)-1, valinit=len(Ws)//2, valstep=1)
    x = np.linspace(0,1, num = L)
    rows = ["True maps","Reconstructed maps","Spectra"]

    for i in range(K): 

        
        ax1 = fig.add_subplot(gs[0, i])
        ax2 = fig.add_subplot(gs[1, i])
        ax3 = fig.add_subplot(gs[2, i])

        if i == 0:                
            ax1.set_ylabel('True maps', rotation=90, fontsize=fontsize)
            ax2.set_ylabel('Reconstructed maps', rotation=90, fontsize=fontsize)
            ax3.set_ylabel('Spectra', rotation=90, fontsize=fontsize)

        
        ax3.plot(x,Ddot.T[i,:],'bo',label='truth',linewidth=4)
        ax3.plot(x,D[:,i],'r-',label='reconstructed',markersize=3.5)
        ax3.set_title("{:.2f} deg".format(angles[true_inds[i]]),fontsize = fontsize-2)
        ax3.set_xlim(0,1)

        ax2.imshow((Hflat[i]).reshape(n_pixel_side,n_pixel_side),vmin = vmin, vmax = vmax , cmap=cmap)
        ax2.set_title("R2: {:.2f}".format(r2[i]),fontsize = fontsize-2)
        # axes[i,1].set_ylim(0.0,1.0)
        ax2.tick_params(axis = "both",labelleft = False, labelbottom = False,left = False, bottom = False)

        im = ax1.imshow(Hdotflat[i].reshape(n_pixel_side,n_pixel_side),vmin = vmin, vmax = vmax, cmap=cmap)
        ax1.set_title("Phase {}".format(i),fontsize = fontsize)
        ax1.tick_params(axis = "both",labelleft = False, labelbottom = False,left = False, bottom = False)
        ax3.legend()

       

    # Merge the bottom cells of the figure and plot the losses
    # ax4 = fig.add_subplot(gs[3, :])
    # ax4.set_ylabel('R2 Losses')
    # ax4.set_xlabel('Iteration')
    # ax4.plot(avg_r2_losses)

    # ax5 = fig.add_subplot(gs[4, :])
    # ax5.set_ylabel('Angle lesses')
    # ax5.set_xlabel('Iteration')
    # ax5.plot(avg_ang_losses)

    # ax6 = fig.add_subplot(gs[5, :])
    # ax6.set_ylabel('Losses')
    # ax6.set_xlabel('Iteration')
    # ax6.plot(losses)

    # for ax, row in zip(axes[:,0], rows):
    #     ax.set_ylabel(row, rotation=90, fontsize=fontsize)


    fig.subplots_adjust(right=0.84)
    # put colorbar at desire position
    cbar_ax = fig.add_axes([0.85, 0.5, 0.01, 0.3])
    fig.colorbar(im,cax=cbar_ax)
    fig.savefig('../results/figure.png')



  


    
        
np.random.seed(1)
n_poisson = 300 # Average poisson number per pixel (this number will be splitted on the L dimension)
G, Wdot, Ddot, Hdot, Hdotflat, Ydot, Y, shape_2d, K = create_toy_problem(L, C, n_poisson, seed=1)
l = np.arange(0, 1, 1/L)
# plt.plot(l, G[:,:3])
# plt.title("Spectral response of each elements")
# plt.draw()

pixels_side = 100
a = np.asarray(Y)
np.savetxt("../results/inputX.csv", a, delimiter=",")

a = np.asarray(Ydot)
np.savetxt("../results/inputDotX.csv", a, delimiter=",")
Ximg = Y.reshape((-1, pixels_side,pixels_side))
XimgDot = Ydot.reshape((-1, pixels_side,pixels_side))



num_iterations = 200


def downsample(X,n):
    b = X.shape[2]//n
    return X.reshape(X.shape[0], -1, n, b, n).sum((-1, -3)) /(n*n)

def downsample_vertical(X,n):
    b = X.shape[0]//n
    return X.reshape( -1, b,  X.shape[1], X.shape[2]).sum((-4)) /(n)


def downsample_flat(X,n, pixels_side):
    X = X.reshape((-1, pixels_side,pixels_side))
    X = downsample(X,n)
    return X.reshape((-1, pixels_side*pixels_side//n//n))

def downsample_flat_vertical(X,n):
    b = X.shape[0]//n
    return X.reshape( -1, b, X.shape[1]).sum((-3)) /(n)
print(Ximg.shape)

print(downsample_vertical(Ximg, 2).shape)

XimgDot = Ydot.reshape((-1, pixels_side,pixels_side))



fig = plt.figure()

X = downsample(Ximg, 1)
Xdot_ = downsample(XimgDot, 1)
plt.plot(X[:,0,0])
plt.plot(Xdot_[:,0,0])
plt.plot()
plt.xlabel('Energy channel [-]')
plt.ylabel('Intensity [-]')
plt.legend(['Measured', 'Ground Truth'])
plt.savefig("../results/df1_spectral.png",bbox_inches='tight')




fig = plt.figure()

X = downsample(Ximg,4)
Xdot_ = downsample(XimgDot, 4)
plt.plot(X[:,0,0],  label='Sine')
plt.plot(Xdot_[:,0,0],label='cos')
plt.xlabel('Energy channel [-]')
plt.ylabel('Intensity [-]')
plt.legend(['Measured', 'Ground Truth'])
plt.savefig("../results/df4_spectral.png",bbox_inches='tight')

fig = plt.figure()

X = downsample(Ximg,10)
Xdot_ = downsample(XimgDot, 10)
plt.plot(X[:,0,0])
plt.plot(Xdot_[:,0,0])
plt.xlabel('Energy channel [-]')
plt.ylabel('Intensity [-]')
plt.legend(['Measured', 'Ground Truth'])
plt.savefig("../results/df10_spectral.png",bbox_inches='tight')

X = downsample(Ximg,1)
fig = plt.figure()
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
# ax.set_title('Intensities [-]')
ax.imshow(X[0,:,:],  cmap='Greys')
plt.savefig("../results/df1_spatial.png",bbox_inches='tight')

X = downsample(Ximg,4)
fig = plt.figure()
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
# ax.set_title('Intensities [-]')
ax.imshow(X[0,:,:],  cmap='Greys')
plt.savefig("../results/df4_spatial.png",bbox_inches='tight')

X = downsample(Ximg,10)
fig = plt.figure()
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
# ax.set_title('Intensities [-]')
ax.imshow(X[0,:,:],  cmap='Greys')
plt.savefig("../results/df10_spatial.png",bbox_inches='tight')






def runProblem(lambda_L, mu, df, use_pinv, smooth_input, num_iterations):
    pixels_side = 100

    # Baseline
    downsampling_factors = [df]
    Hflat =None
    W = None

    for i in range(len(downsampling_factors)):
        start_time = time.time()
        n = df
        X = downsample(Ximg,n)
        # Xvert = downsample_vertical(Ximg, 10)
        if (smooth_input == 1):
            X = ndimage.median_filter(X, size=(1,3,3))

        Xflat = X.reshape((X.shape[0], -1))
        est = SmoothNMF( n_components = 3,tol=0.0001, max_iter = num_iterations, G = None, lambda_L=lambda_L, mu=mu, force_simplex=False)
        D , Hs, Ws, G, losses = est.fit_transform(Xflat, W=W, H=Hflat, update_W=True)
        Hflat = est.H_
        W = est.W_

        if use_pinv == 1:
            n = 1
            X = downsample(Ximg,n)
            if (smooth_input == 1):
                X = ndimage.median_filter(X, size=(1,3,3))
            Xflat = X.reshape((X.shape[0], -1))
            Hflat = np.linalg.pinv(W)@Xflat
        
        Hdotflat_down = downsample_flat(Hdotflat, n, pixels_side)


        employee_info = ['id', 'value']

    

        angle, true_inds = find_min_angle(Ddot.T, D.T, unique=True, get_ind=True)
        r2 = ordered_r2(Hdotflat_down, Hflat, true_inds)

        data = [['time', time.time()-start_time], ['angle', sum(angle)/len(angle)], ['r2', sum(r2)/len(r2)]]

        with open('../results/outputParams.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(data)
        plot_results(Ddot, D, Hdotflat_down, Hflat, pixels_side//n, Hs, Ws, G, losses)

        print(D.shape)
        print(Hflat.shape)
        a = np.asarray(D@Hflat)
        np.savetxt("../results/outputX.csv", a, delimiter=",")

        a = np.asarray(Xflat-D@Hflat)
        np.savetxt("../results/outputDiffX.csv", a, delimiter=",")

        

    

while (True):
    time.sleep(0.1)
    if (os.path.exists("../results/queries.csv")):

        # Read queries.csv file queries in results folder
        with open('../results/queries.csv', newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            os.remove("../results/queries.csv")
            spamreader = list(spamreader)
            _mu = float(spamreader[1][1])
            _lambda = float(spamreader[2][1])
            _df = int(spamreader[3][1])
            _use_pinv = int(spamreader[4][1])
            _smooth_input = int(spamreader[5][1])
            _num_iterations = int(spamreader[6][1])

            for row in spamreader:
                print(', '.join(row))

            runProblem(_lambda, _mu, _df, _use_pinv, _smooth_input, _num_iterations)



