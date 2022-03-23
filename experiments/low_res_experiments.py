# %load_ext autoreload
# %autoreload 2
# %matplotlib qt

import hyperspy.api as hs
from esmpy.estimators import SmoothNMF
import esmpy.datasets as ds

import matplotlib.pyplot as plt
import numpy as np

from esmpy.models.EDXS_function import print_concentrations_from_W

ds.generate_built_in_datasets(seeds_range=5)
spim = ds.load_particules(sample = 0)

spim = hs.load("../generated_datasets/FpBrgCaPv_N293_paper/sample_5.hspy")

# spim.axes_manager[-1].offset = 12000
spim.set_signal_type("EDS_ESMPY")
spim.set_analysis_parameters(beam_energy = 200,
azimuth_angle = 0.0,
elevation_angle = 22.0,
tilt_stage = 0.0,
elements = ["Si","Mg","Fe"],
thickness = 200e-7,
density = 3.5,
detector_type = "SDD_efficiency.txt",
width_slope = 0.01,
width_intercept = 0.065,
xray_db = "default_xrays.json")
G = spim.build_G("bremsstrahlung", norm = True)
fW = spim.set_fixed_W({"p0" : {"Si" : 0.0},"p1" : {"Fe" : 0.0}, "p2" : {"Mg" : 0.0}})

from esmpy.measures import find_min_angle, find_min_MSE, ordered_mse, ordered_r2, ordered_mae
from matplotlib.widgets import Slider


    


def plot_results(Ddot, D, Hdotflat, Hflat, n_pixel_side, Hs, Ws):
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


    fig, axes = plt.subplots(K,3,figsize = (scale/K * 3 * aspect_ratio,scale))
    plt.subplots_adjust(bottom=0.25)
    ax_slide = plt.axes([0.25, 0.1, 0.65, 0.03])
    s_factor = Slider(ax_slide, 'Smoothing factor',
                  0, len(Ws), valinit=6, valstep=1)
    x = np.linspace(0,1, num = L)
    for i in range(K): 
        axes[2,i].plot(x,Ddot.T[i,:],'bo',label='truth',linewidth=4)
        axes[2,i].plot(x,D[:,true_inds[i]],'r-',label='reconstructed',markersize=3.5)
        axes[2,i].set_title("{:.2f} deg".format(angles[i]),fontsize = fontsize-2)
        axes[2,i].set_xlim(0,1)

        axes[1,i].imshow((Hflat[true_inds[i],:]).reshape(n_pixel_side,n_pixel_side),vmin = vmin, vmax = vmax , cmap=cmap)
        axes[1,i].set_title("R2: {:.2f}".format(r2[true_inds[i]]),fontsize = fontsize-2)
        # axes[i,1].set_ylim(0.0,1.0)
        axes[1,i].tick_params(axis = "both",labelleft = False, labelbottom = False,left = False, bottom = False)

        im = axes[0,i].imshow(Hdotflat[i].reshape(n_pixel_side,n_pixel_side),vmin = vmin, vmax = vmax, cmap=cmap)
        axes[0,i].set_title("Phase {}".format(i),fontsize = fontsize)
        axes[0,i].tick_params(axis = "both",labelleft = False, labelbottom = False,left = False, bottom = False)
        axes[2,0].legend()
    rows = ["True maps","Reconstructed maps","Spectra"]

    for ax, row in zip(axes[:,0], rows):
        ax.set_ylabel(row, rotation=90, fontsize=fontsize)


    fig.subplots_adjust(right=0.84)
    # put colorbar at desire position
    cbar_ax = fig.add_axes([0.85, 0.5, 0.01, 0.3])
    fig.colorbar(im,cax=cbar_ax)

    # Updating the plot
    def update(val):
        current_v = s_factor.val
        # spline = UnivariateSpline(x, y, s = current_v)
        # p.set_ydata()
        angles, true_inds = find_min_angle(Ddot.T, Ws[int(val)].T, unique=True, get_ind=True)
       
        r2 = ordered_r2(Hdotflat, Hs[int(val)], true_inds)
        for i in range(K): 
            axes[2,i].plot(x,Ddot.T[i,:],'bo',label='truth',linewidth=4)
            axes[2,i].plot(x,Ws[int(val)][:,true_inds[i]],'r-',label='reconstructed',markersize=3.5)
            axes[2,i].set_title("{:.2f} deg".format(angles[i]),fontsize = fontsize-2)

            axes[1,i].imshow((Hs[int(val)][true_inds[i],:]).reshape(n_pixel_side,n_pixel_side),vmin = vmin, vmax = vmax , cmap=cmap)
            axes[1,i].set_title("R2: {:.2f}".format(r2[true_inds[i]]),fontsize = fontsize-2)

        #redrawing the figure
        print(1)
        fig.canvas.draw()
    # fig.tight_layout()
    # Calling the function "update" when the value of the slider is changed
    s_factor.on_changed(update)
    plt.show()

def downsample(X,n):
    b = X.shape[2]//n
    return X.reshape(X.shape[0], -1, n, b, n).sum((-1, -3)) /(n*n)

def downsample_flat(X,n, pixels_side):
    X = X.reshape((-1, pixels_side,pixels_side))
    X = downsample(X,n)
    return X.reshape((-1, pixels_side*pixels_side//n//n))

pixels_side = 64

G = spim.build_G("bremsstrahlung")
shape_2d = spim.shape_2d
# phases, weights = spim.phases, spim.weights
X = spim.X
est = SmoothNMF( n_components = 3,tol=0.000001, max_iter = 1000, G = None, lambda_L= 0., mu=0., force_simplex=False, init="nndsvdar")
Ximg = X.reshape((-1, pixels_side,pixels_side))
print(Ximg.shape)

downsampling_factors = [16,8,4]
results_0 = []
results_1 = []
results_2 = []
results_max = []

Hflat =None
W = None
for i in range(len(downsampling_factors)):
    downsample_factor_results_0 = []
    downsample_factor_results_1 = []
    downsample_factor_results_2 = []


    n = downsampling_factors[i]
    X_ = downsample(Ximg,n)
    Xflat = X_.reshape((X_.shape[0], -1))
    if (i > 0):
        print(Hflat.shape)
        factor = downsampling_factors[i-1]//downsampling_factors[i]
        H = Hflat.reshape((-1, pixels_side//n//factor, pixels_side//n//factor))
        H = H.repeat(factor, axis = 1).repeat(factor, axis = 2)
        Hflat = H.reshape((-1, pixels_side*pixels_side//n//n)) +0.1
        print(H.shape)
        print(Xflat.shape, W.shape, Hflat.shape)
        W =  W 
    
    # est = SmoothNMF( n_components = 3,tol=0.000001, max_iter = 1000, G = None, lambda_L= 0., mu=0., force_simplex=False, init="nndsvdar", fixed_W=W, random_state=False)
    D , Hs, Ws = est.fit_transform(Xflat, W=W, H=Hflat, n_pixel_side=pixels_side//n)
    Hflat = est.H_
    W = est.W_

    Wdot = spim.phases
    Hflatdot = downsample_flat(spim.maps,n, pixels_side)

    angles, _ = find_min_angle(Wdot.T, W.T, unique=True, get_ind=True)

     
    results_0.append(angles[0])
    results_1.append(angles[1])
    results_2.append(angles[2])
    results_max.append(max(angles))

   


plot_results(Wdot, W, Hflatdot, Hflat, pixels_side//n, Hs, Ws)

plt.plot([pixels_side//downsampling_factors[i] for i in range(len(downsampling_factors))], results_0)
plt.show()

plt.plot([pixels_side//downsampling_factors[i] for i in range(len(downsampling_factors))], results_1)

plt.show()

plt.plot([pixels_side//downsampling_factors[i] for i in range(len(downsampling_factors))], results_2)
plt.show()

plt.plot([pixels_side//downsampling_factors[i] for i in range(len(downsampling_factors))], results_max)
plt.show()