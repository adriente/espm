# %load_ext autoreload
# %autoreload 2
# %matplotlib qt

from curses.panel import bottom_panel
import hyperspy.api as hs
from esmpy.estimators import SmoothNMF
import esmpy.datasets as ds
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

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


    fig, axes = plt.subplots(3, 3,figsize = (scale/K * 3 * aspect_ratio,scale))
    plt.subplots_adjust(bottom=0.25)
    ax_slide = plt.axes([0.25, 0.1, 0.65, 0.03])
    s_factor = Slider(ax_slide, 'N. Iteration',
                  0, len(Ws)-1, valinit=len(Ws)//2, valstep=1)
    x = np.linspace(0,1, num = L)
    for i in range(K): 
        axes[2,i].plot(x,Ddot.T[i,:],'bo',label='truth',linewidth=4)
        axes[2,i].plot(x,D[:,true_inds[i]],'r-',label='reconstructed',markersize=3.5)
        axes[2,i].set_title("{:.2f} deg".format(angles[i]),fontsize = fontsize-2)
        axes[2,i].set_xlim(0,1)

        axes[1,i].imshow((Hflat[i]).reshape(n_pixel_side,n_pixel_side),vmin = vmin, vmax = vmax , cmap=cmap)
        axes[1,i].set_title("R2: {:.2f}".format(r2[true_inds[i]]),fontsize = fontsize-2)
        # axes[i,1].set_ylim(0.0,1.0)
        axes[1,i].tick_params(axis = "both",labelleft = False, labelbottom = False,left = False, bottom = False)

        im = axes[0,i].imshow(Hdotflat[i].reshape(n_pixel_side,n_pixel_side),vmin = vmin, vmax = vmax, cmap=cmap)
        axes[0,i].set_title("Phase {}".format(i),fontsize = fontsize)
        axes[0,i].tick_params(axis = "both",labelleft = False, labelbottom = False,left = False, bottom = False)
        axes[2,0].legend()

        # numerator = ((Hflatdot[i] - Hflat[true_inds[i],:]) ** 2)
        # denominator = ((Hflatdot[i] - np.average(Hflatdot[i], axis=0)) ** 2)
        # im = axes[3,i].imshow(numerator.reshape(n_pixel_side,n_pixel_side),vmin = vmin, vmax = vmax, cmap=cmap)
        # axes[3,i].set_title("Phase {}".format(i),fontsize = fontsize)
        # axes[3,i].tick_params(axis = "both",labelleft = False, labelbottom = False,left = False, bottom = False)

        # im = axes[4,i].imshow(denominator.reshape(n_pixel_side,n_pixel_side),vmin = vmin, vmax = vmax, cmap=cmap)
        # axes[4,i].set_title("Phase {}".format(i),fontsize = fontsize)
        # axes[4,i].tick_params(axis = "both",labelleft = False, labelbottom = False,left = False, bottom = False)

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

            axes[1,i].imshow((Hs[int(val)][i]).reshape(n_pixel_side,n_pixel_side),vmin = vmin, vmax = vmax , cmap=cmap)
            axes[1,i].set_title("R2: {:.2f}".format(r2[true_inds[i]]),fontsize = fontsize-2)

            # numerator = ((Hflatdot[i] - Hs[int(val)][true_inds[i]]) ** 2)
            # denominator = ((Hflatdot[i] - np.average(Hflatdot[i], axis=0)) ** 2)
            # im = axes[3,i].imshow(numerator.reshape(n_pixel_side,n_pixel_side),vmin = vmin, vmax = vmax, cmap=cmap)
            # axes[3,i].set_title("Phase {}".format(i),fontsize = fontsize)
            # axes[3,i].tick_params(axis = "both",labelleft = False, labelbottom = False,left = False, bottom = False)

            # im = axes[4,i].imshow(denominator.reshape(n_pixel_side,n_pixel_side),vmin = vmin, vmax = vmax, cmap=cmap)
            # axes[4,i].set_title("Phase {}".format(i),fontsize = fontsize)
            # axes[4,i].tick_params(axis = "both",labelleft = False, labelbottom = False,left = False, bottom = False)
        print(len(Hs), len(Ws))

        #redrawing the figure
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


downsampling_factors = [16,8,4,2]
results_0 = []
results_1 = []
results_2 = []
results_max = []
r2_max = []
angles = []
r2s = []



for j in range(1):
    downsample_factor_results_0 = []
    downsample_factor_results_1 = []
    downsample_factor_results_2 = []
    angle_max_1 = []
    r2_max_1 = []
    times_1 = []
 

    X = spim.X
    est = SmoothNMF( n_components = 3,tol=0.0001, max_iter = 1000, G = None, lambda_L= 0., mu=0., force_simplex=False, init="nndsvdar")
    Ximg = X.reshape((-1, pixels_side,pixels_side))
    Hflat =None
    W = None

    for i in range(len(downsampling_factors)):
        start_time = time.time()
        n = downsampling_factors[i]
        X_ = downsample(Ximg,n)
        Xflat = X_.reshape((X_.shape[0], -1))

        if (i > 0):
            print(Hflat.shape)
            factor = downsampling_factors[i-1]//downsampling_factors[i]
            H = Hflat.reshape((-1, pixels_side//n//factor, pixels_side//n//factor))
            H = H.repeat(factor, axis = 1).repeat(factor, axis = 2)
            Hflat = H.reshape((-1, pixels_side*pixels_side//n//n)) +0.01
            print(H.shape)
            print(Xflat.shape, W.shape, Hflat.shape)
            W =  W 
        
        est = SmoothNMF( n_components = 3,tol=0.0001, max_iter = 1000, G = None, lambda_L= 0., mu=0., force_simplex=False, init="nndsvdar")
        if i > 2:
            D , Hs, Ws = est.fit_transform(Xflat, H=Hflat, W=W, update_W=False)
        else:
            D , Hs, Ws = est.fit_transform(Xflat, H=Hflat, W=W, update_W=True)

        Hflat = est.H_
        W = est.W_

        Wdot = spim.phases
        Hflatdot = downsample_flat(spim.maps,n, pixels_side)
        times_1.append(time.time()-start_time)

        angle, true_inds = find_min_angle(Wdot.T, W.T, unique=True, get_ind=True)
        r2 = ordered_r2(Hflatdot, Hflat, true_inds)
        angles = angles + [angle[0], angle[1], angle[2]]
        r2s = r2s + [r2[0], r2[1], r2[2]]
        downsample_factor_results_0.append(angle[0])
        downsample_factor_results_1.append(angle[1])
        downsample_factor_results_2.append(angle[2])
        angle_max_1.append(max(angle))
        r2_max_1.append(min(r2))
        
        
    results_0.append(downsample_factor_results_0)
    results_1.append(downsample_factor_results_1)
    results_2.append(downsample_factor_results_2)
    results_max.append(angle_max_1)
    r2_max.append(r2_max_1)

G = spim.build_G("bremsstrahlung")
shape_2d = spim.shape_2d
# phases, weights = spim.phases, spim.weights



results_max_2 = []
for j in range(1):
    times_2 = []
    angle_max_2 = []
    r2_max_2 =  []
    X = spim.X
    est = SmoothNMF( n_components = 3,tol=0.0001, max_iter = 1000, G = None, lambda_L= 0., mu=0., force_simplex=False, init="nndsvdar")
    Ximg = X.reshape((-1, pixels_side,pixels_side))
    Hflat =None
    W = None

    for i in range(len(downsampling_factors)):
        

        start_time = time.time()
        n = downsampling_factors[i]
        X_ = downsample(Ximg,n)
        Xflat = X_.reshape((X_.shape[0], -1))
        if (i > 0):
            factor = downsampling_factors[i-1]//downsampling_factors[i]
            H = Hflat.reshape((-1, pixels_side//n//factor, pixels_side//n//factor))
            H = H.repeat(factor, axis = 1).repeat(factor, axis = 2)
            Hflat = H.reshape((-1, pixels_side*pixels_side//n//n)) +0.01
            W =  W 
        
        est = SmoothNMF( n_components = 3,tol=0.0001, max_iter = 1000, G = None, lambda_L= 0., mu=0., force_simplex=False, init="nndsvdar")
        D , Hs, Ws = est.fit_transform(Xflat, H=Hflat, W=W, update_W=True)
       
        Hflat = est.H_
        W = est.W_

        Wdot = spim.phases
        Hflatdot = downsample_flat(spim.maps,n, pixels_side)
        times_2.append(time.time()-start_time)

        angle, true_inds = find_min_angle(Wdot.T, W.T, unique=True, get_ind=True)
        r2 = ordered_r2(Hflatdot, Hflat, true_inds)
        angles = angles + [angle[0], angle[1], angle[2]]
        r2s = r2s + [r2[0], r2[1], r2[2]]
 
        angle_max_2.append(max(angle))
        r2_max_2.append(min(r2))

        
        plot_results(Wdot, W, Hflatdot, Hflat, pixels_side//n, Hs, Ws)

    results_max_2.append(angle_max_2)
    r2_max.append(r2_max_2)




results_max_3 = []
for j in range(1):

    times_3 = []
    angle_max_3 = []
    r2_max_3 = []
    
    X = spim.X
    est = SmoothNMF( n_components = 3,tol=0.0001, max_iter = 1000, G = None, lambda_L= 0., mu=0., force_simplex=False, init="nndsvdar")
    Ximg = X.reshape((-1, pixels_side,pixels_side))
    
    Hflat =None
    W = None

    for i in range(len(downsampling_factors)):
    
        start_time = time.time()
        n = downsampling_factors[i]
        X_ = downsample(Ximg,n)
        Xflat = X_.reshape((X_.shape[0], -1))

        est = SmoothNMF( n_components = 3,tol=0.0001, max_iter = 1000, G = None, lambda_L= 0., mu=0., force_simplex=False, init="nndsvdar")
        D , Hs, Ws = est.fit_transform(Xflat, update_W=True)
       
        Hflat = est.H_
        W = est.W_

        Wdot = spim.phases
        Hflatdot = downsample_flat(spim.maps,n, pixels_side)
        times_3.append(time.time()-start_time)

        angle, true_inds = find_min_angle(Wdot.T, W.T, unique=True, get_ind=True)
        r2 = ordered_r2(Hflatdot, Hflat, true_inds)
        angles = angles + [angle[0], angle[1], angle[2]]
        r2s = r2s + [r2[0], r2[1], r2[2]]
 
        angle_max_3.append(max(angle))
        r2_max_3.append(min(r2))


    results_max_3.append(angle_max_3)
    r2_max.append(r2_max_3)



results_max_4 = []
for j in range(1):
    times_4 = []
    angle_max_4 = []
    r2_max_4 =  []
    X = spim.X
    est = SmoothNMF( n_components = 3,tol=0.0001, max_iter = 1000, G = None, lambda_L= 0., mu=0., force_simplex=False, init="nndsvdar")
    Ximg = X.reshape((-1, pixels_side,pixels_side))
    Hflat =None
    W = None

    for i in range(len(downsampling_factors)):
        

        start_time = time.time()
        n = downsampling_factors[i]
        X_ = downsample(Ximg,n)
        X_ = ndimage.median_filter(X_, size=(1,5,5))
        Xflat = X_.reshape((X_.shape[0], -1))
        if (i > 0):
            factor = downsampling_factors[i-1]//downsampling_factors[i]
            H = Hflat.reshape((-1, pixels_side//n//factor, pixels_side//n//factor))
            H = H.repeat(factor, axis = 1).repeat(factor, axis = 2)
            Hflat = H.reshape((-1, pixels_side*pixels_side//n//n)) +0.01
            W =  W 
        
        est = SmoothNMF( n_components = 3,tol=0.0001, max_iter = 1000, G = None, lambda_L= 0., mu=0., force_simplex=False, init="nndsvdar")
        if i > 2:
            D , Hs, Ws = est.fit_transform(Xflat, H=Hflat, W=W, update_W=False)
        else:
            D , Hs, Ws = est.fit_transform(Xflat, H=Hflat, W=W, update_W=True)
       
        Hflat = est.H_
        W = est.W_

        Wdot = spim.phases
        Hflatdot = downsample_flat(spim.maps,n, pixels_side)
        times_4.append(time.time()-start_time)

        angle, true_inds = find_min_angle(Wdot.T, W.T, unique=True, get_ind=True)
        r2 = ordered_r2(Hflatdot, Hflat, true_inds)
        print(angles, r2s)
        angles = angles + [angle[0], angle[1], angle[2]]
        r2s = r2s + [r2[0], r2[1], r2[2]]
 
        angle_max_4.append(max(angle))
        r2_max_4.append(min(r2))

        plot_results(Wdot, W, Hflatdot, Hflat, pixels_side//n, Hs, Ws)


    results_max_2.append(angle_max_2)
    r2_max.append(r2_max_2)

results_max_5 = []
for j in range(1):

    times_5 = []
    angle_max_5 = []
    r2_max_5 = []
    
    X = spim.X
    est = SmoothNMF( n_components = 3,tol=0.00001, max_iter = 1000, G = None, lambda_L= 0., mu=0., force_simplex=False, init="nndsvdar")
    Ximg = X.reshape((-1, pixels_side,pixels_side))

    Hflat =None
    W = None

    for i in range(len(downsampling_factors)):
    
        start_time = time.time()
        n = downsampling_factors[i]
        X_ = downsample(Ximg,n)
        Xflat = X_.reshape((X_.shape[0], -1))

        est = SmoothNMF( n_components = 3,tol=0.00001, max_iter = 1000, G = None, lambda_L= 0., mu=0., force_simplex=False, init="nndsvdar")
        D , Hs, Ws = est.fit_transform(Xflat, update_W=True)
       
        Hflat = est.H_
        W = est.W_

        Wdot = spim.phases
        Hflatdot = downsample_flat(spim.maps,n, pixels_side)
        times_5.append(time.time()-start_time)

        angle, true_inds = find_min_angle(Wdot.T, W.T, unique=True, get_ind=True)
        r2 = ordered_r2(Hflatdot, Hflat, true_inds)
        angles = angles + [angle[0], angle[1], angle[2]]
        r2s = r2s + [r2[0], r2[1], r2[2]]
 
        angle_max_5.append(max(angle))
        r2_max_5.append(min(r2))


    results_max_5.append(angle_max_5)
    r2_max.append(r2_max_5)



plt.plot(downsampling_factors, angle_max_5)
plt.plot(downsampling_factors, angle_max_3)
plt.plot(downsampling_factors, angle_max_2)
plt.plot(downsampling_factors, angle_max_1)
plt.plot(downsampling_factors, angle_max_4)

plt.yscale("log")
plt.grid()
plt.ylim(bottom=0)
plt.title("Angle errors vs Downsampling factor")
plt.xlabel("N. Pixels/side")
plt.ylabel("Worst phase angle error [deg]")
plt.legend(["Arbitrary initialization after every upsampling iteration", "+ Lower tolerance from 0.00001 to 0.0001", "+ Carry upsampled H after every upsampling iteration", "+ freeze W after first 2 upsampling iterations", "+ apply median filder to X in pixel dimensions"])
plt.xlim(max(downsampling_factors), 0)  # decreasing time
plt.xticks(downsampling_factors)
plt.show()

plt.plot(downsampling_factors, times_5)
plt.plot(downsampling_factors, times_3)
plt.plot(downsampling_factors, times_2)
plt.plot(downsampling_factors, times_1)
plt.plot(downsampling_factors, times_4)
plt.grid()
plt.ylim(bottom=0)
plt.title("Computation time vs Dowsnampling factor")
plt.xlabel("Downsampling factor")
plt.ylabel("Time to convergence [s]")
plt.legend(["Arbitrary initialization after every upsampling iteration", "+ Lower tolerance from 0.00001 to 0.0001", "+ Carry upsampled H after every upsampling iteration", "+ freeze W after first 2 upsampling iterations", "+ apply median filder to X in pixel dimensions"])
plt.xlim(max(downsampling_factors), 0)  # decreasing time
plt.xticks(downsampling_factors)
plt.show()

plt.plot(downsampling_factors, r2_max_5)
plt.plot(downsampling_factors, r2_max_3)
plt.plot(downsampling_factors, r2_max_2)
plt.plot(downsampling_factors, r2_max_1)
plt.plot(downsampling_factors, r2_max_4)
plt.grid()
plt.ylim(top=1.2)
plt.title("R2 vs Dowsnampling factor")
plt.xlabel("Downsampling factor")
plt.ylabel("Worst phase R2 [-]")
plt.legend(["Arbitrary initialization after every upsampling iteration", "+ Lower tolerance from 0.00001 to 0.0001", "+ Carry upsampled H after every upsampling iteration", "+ freeze W after first 2 upsampling iterations", "+ apply median filder to X in pixel dimensions"])
plt.xlim(max(downsampling_factors), 0)  # decreasing time
plt.xticks(downsampling_factors)
plt.show()



plt.scatter(angles, r2s)
plt.title("R2 vs Angle error (all experiments, all upsampling iterations, all phases)")
plt.xlabel("Angle error [deg]")
plt.xlabel("Phase R2 [-]")
plt.grid()
plt.show()




