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

def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)
print(Z.shape)
print(X.shape)


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


    
def kl(H_true, H_pred):
    # Normalize matrices along the rows
    H_true = H_true / H_true.sum(axis=1, keepdims=True) + 0.00000001
    H_pred = H_pred / H_pred.sum(axis=1, keepdims=True) + 0.00000001

    # Print sum of H_true along the rows
    print(H_true.sum(axis=1))

    # Print sum of H_pred along the rows
    print(H_pred.sum(axis=1))

    print(H_pred)

    # Compute KL divergence along the rows where H_pred is positive
    return np.sum(np.where(H_true > 0, H_true * np.log(H_true / H_pred), 0), axis=1)


def plot_results(Ddot, D, Hdotflat, Hflat, n_pixel_side, Hs, Ws, losses):
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

    print(Hflat.shape)
    Hflat = Hflat[true_inds,:]
    print(Hflat.shape)
    D = D[:,true_inds]

    mse = ordered_mse(Hdotflat, Hflat, true_inds)
    mae = ordered_mae(Hdotflat, Hflat, true_inds)
    r2 = ordered_r2(Hdotflat, Hflat, true_inds)
    kls = kl(Hdotflat, Hflat[true_inds, :])
    
    

    fig = plt.figure(figsize = (scale/K * 3 * aspect_ratio,scale))
    gs = fig.add_gridspec(4,3)
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
        ax3.set_title("{:.2f} deg".format(angles[i]),fontsize = fontsize-2)
        ax3.set_xlim(0,1)

        ax2.imshow((Hflat[i]).reshape(n_pixel_side,n_pixel_side),vmin = vmin, vmax = vmax , cmap=cmap)
        ax2.set_title("KL: {:.2f}".format(kls[i]),fontsize = fontsize-2)
        # axes[i,1].set_ylim(0.0,1.0)
        ax2.tick_params(axis = "both",labelleft = False, labelbottom = False,left = False, bottom = False)

        im = ax1.imshow(Hdotflat[i].reshape(n_pixel_side,n_pixel_side),vmin = vmin, vmax = vmax, cmap=cmap)
        ax1.set_title("Phase {}".format(i),fontsize = fontsize)
        ax1.tick_params(axis = "both",labelleft = False, labelbottom = False,left = False, bottom = False)
        ax3.legend()

       


        # axes[3,:].plot(losses)
        # axes[3,:].set_title("Loss",fontsize = fontsize-2)

        # numerator = ((Hflatdot[i] - Hflat[true_inds[i],:]) ** 2)
        # denominator = ((Hflatdot[i] - np.average(Hflatdot[i], axis=0)) ** 2)
        # im = axes[3,i].imshow(numerator.reshape(n_pixel_side,n_pixel_side),vmin = vmin, vmax = vmax, cmap=cmap)
        # axes[3,i].set_title("Phase {}".format(i),fontsize = fontsize)
        # axes[3,i].tick_params(axis = "both",labelleft = False, labelbottom = False,left = False, bottom = False)

        # im = axes[4,i].imshow(denominator.reshape(n_pixel_side,n_pixel_side),vmin = vmin, vmax = vmax, cmap=cmap)
        # axes[4,i].set_title("Phase {}".format(i),fontsize = fontsize)
        # axes[4,i].tick_params(axis = "both",labelleft = False, labelbottom = False,left = False, bottom = False)

    # Merge the bottom cells of the figure and plot the losses
    ax4 = fig.add_subplot(gs[3, :])
    ax4.set_ylabel('Losses')
    ax4.set_xlabel('Iteration')
    ax4.plot(losses)

    # for ax, row in zip(axes[:,0], rows):
    #     ax.set_ylabel(row, rotation=90, fontsize=fontsize)


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
        kls = kl(Hdotflat, Hs[int(val)])
        print(kls)
        print(r2)

        for i in range(K): 
            axes[2,i].plot(x,Ddot.T[i,:],'bo',label='truth',linewidth=4)
            axes[2,i].plot(x,Ws[int(val)][:,i],'r-',label='reconstructed',markersize=3.5)
            axes[2,i].set_title("{:.2f} deg".format(angles[i]),fontsize = fontsize-2)

            axes[1,i].imshow((Hs[int(val)][i]).reshape(n_pixel_side,n_pixel_side),vmin = vmin, vmax = vmax , cmap=cmap)
            axes[1,i].set_title("R2: {:.2f}".format(r2[i]),fontsize = fontsize-2)

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


downsampling_factors = [1]

# results_max = []
# r2_max = []
angles = []
r2s = []

angle_max_exp = []
r2_max_exp = []
times_exp = []




X = spim.X
est = SmoothNMF( n_components = 3,tol=0.00001, max_iter = 1000, G = None, lambda_L=0, mu=0, force_simplex=False, init="nndsvdar")
Ximg = X.reshape((-1, pixels_side,pixels_side))
Hflat =None
W = None

for i in range(len(downsampling_factors)):
    start_time = time.time()
    n = downsampling_factors[i]
    X_ = downsample(Ximg,n)
    Xflat = X_.reshape((X_.shape[0], -1))
    
    est = SmoothNMF( n_components = 3,tol=0.00001, max_iter = 1000, G = None, lambda_L=0, mu=0, force_simplex=False, init="nndsvdar")

    D , Hs, Ws, losses = est.fit_transform(Xflat)

    Hflat = est.H_
    W = est.W_

    Wdot = spim.phases
    Hflatdot = downsample_flat(spim.maps,n, pixels_side)

    angle, true_inds = find_min_angle(Wdot.T, W.T, unique=True, get_ind=True)
    r2 = ordered_r2(Hflatdot, Hflat[true_inds,:], true_inds)
    kls = kl(Hflatdot, Hflat[true_inds,:])


    print('baseline angle', sum(angle)/len(angle))
    print('baseline r2', sum(r2)/len(angle))

plot_results(Wdot, W, Hflatdot, Hflat, pixels_side//n, Hs, Ws, losses)
r2_max_exp.append(sum(r2)/len(r2))
angle_max_exp.append(sum(angle)/len(angle))

X = spim.X
est = SmoothNMF( n_components = 3,tol=0.0001, max_iter = 1000, G = None, lambda_L=10, mu=10, force_simplex=False, init="nndsvdar")
Ximg = X.reshape((-1, pixels_side,pixels_side))
Hflat =None
W = None

for i in range(len(downsampling_factors)):
    start_time = time.time()
    n = downsampling_factors[i]
    X_ = downsample(Ximg,n)
    Xflat = X_.reshape((X_.shape[0], -1))
    
    est = SmoothNMF( n_components = 3,tol=0.0001, max_iter = 1000, G = None, lambda_L=10, mu=10, force_simplex=False, init="nndsvdar")

    D , Hs, Ws, losses = est.fit_transform(Xflat)

    Hflat = est.H_
    W = est.W_

    Wdot = spim.phases
    Hflatdot = downsample_flat(spim.maps,n, pixels_side)

    angle, true_inds = find_min_angle(Wdot.T, W.T, unique=True, get_ind=True)
    r2 = ordered_r2(Hflatdot, Hflat[true_inds,:], true_inds)
    kls = kl(Hflatdot, Hflat[true_inds,:])


    print('baseline angle', sum(angle)/len(angle))
    print('baseline r2', sum(r2)/len(angle))

plot_results(Wdot, W, Hflatdot, Hflat, pixels_side//n, Hs, Ws, losses)
r2_max_exp.append(sum(r2)/len(r2))
angle_max_exp.append(sum(angle)/len(angle))

n = 1
m = 1


X = spim.X
est = SmoothNMF( n_components = 3,tol=0.0001, max_iter = 1000, G = None, lambda_L=0, mu=0, force_simplex=False, init="nndsvdar")
Ximg = X.reshape((-1, pixels_side,pixels_side))
Hflat =None
W = None

for i in range(len(downsampling_factors)):
    start_time = time.time()
    n = downsampling_factors[i]
    X_ = downsample(Ximg,n)
    X_ = ndimage.median_filter(X_, size=(1,10,10))
    Xflat = X_.reshape((X_.shape[0], -1))
    
    est = SmoothNMF( n_components = 3,tol=0.0001, max_iter = 1000, G = None, lambda_L=0, mu=0, force_simplex=False, init="nndsvdar")

    D , Hs, Ws, losses = est.fit_transform(Xflat)

    Hflat = est.H_
    W = est.W_

    Wdot = spim.phases
    Hflatdot = downsample_flat(spim.maps,n, pixels_side)

    angle, true_inds = find_min_angle(Wdot.T, W.T, unique=True, get_ind=True)
    r2 = ordered_r2(Hflatdot, Hflat[true_inds,:], true_inds)
    kls = kl(Hflatdot, Hflat[true_inds,:])

    print('smoothing angle', sum(angle)/len(angle))
    print('smoothing r2', sum(r2)/len(angle))

plot_results(Wdot, W, Hflatdot, Hflat, pixels_side//n, Hs, Ws, losses)
r2_max_exp.append(sum(r2)/len(r2))
angle_max_exp.append(sum(angle)/len(angle))


downsampling_factors = [16,8,4,2,1]
X = spim.X
est = SmoothNMF( n_components = 3,tol=0.0001, max_iter = 1000, G = None, lambda_L=0, mu=0, force_simplex=False, init="nndsvdar")
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
    if i > 2:
        D , Hs, Ws, losses = est.fit_transform(Xflat, H=Hflat, W=W, update_W=False)
    else:
        D , Hs, Ws, losses = est.fit_transform(Xflat, H=Hflat, W=W, update_W=True)
   

    Hflat = est.H_
    W = est.W_

    Wdot = spim.phases
    Hflatdot = downsample_flat(spim.maps,n, pixels_side)

    angle, true_inds = find_min_angle(Wdot.T, W.T, unique=True, get_ind=True)
    r2 = ordered_r2(Hflatdot, Hflat[true_inds,:], true_inds)
    kls = kl(Hflatdot, Hflat[true_inds,:])


    print('baseline angle', sum(angle)/len(angle))
    print('baseline r2', sum(r2)/len(angle))
plot_results(Wdot, W, Hflatdot, Hflat, pixels_side//n, Hs, Ws, losses)
r2_max_exp.append(sum(r2)/len(r2))
angle_max_exp.append(sum(angle)/len(angle))

downsampling_factors = [1,1]
X = spim.X
est = SmoothNMF( n_components = 3,tol=0.0001, max_iter = 1000, G = None, lambda_L=0, mu=0, force_simplex=False, init="nndsvdar")
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
        H = ndimage.median_filter(H, size=(1,5,5))
  

        print('smoothing')

        Hflat = H.reshape((-1, pixels_side*pixels_side//n//n)) +0.01
        W =  W 
    
    est = SmoothNMF( n_components = 3,tol=0.0001, max_iter = 1000, G = None, lambda_L= 0., mu=0., force_simplex=False, init="nndsvdar")
    D , Hs, Ws, losses = est.fit_transform(Xflat, H=Hflat, update_H=False)

   

    Hflat = est.H_
    W = est.W_

    Wdot = spim.phases
    Hflatdot = downsample_flat(spim.maps,n, pixels_side)

    angle, true_inds = find_min_angle(Wdot.T, W.T, unique=True, get_ind=True)
    r2 = ordered_r2(Hflatdot, Hflat[true_inds,:], true_inds)
    kls = kl(Hflatdot, Hflat[true_inds,:])


    print('baseline angle', sum(angle)/len(angle))
    print('baseline r2', sum(r2)/len(angle))

H = Hflat.reshape((-1, pixels_side//n//factor, pixels_side//n//factor))
H = ndimage.median_filter(H, size=(1,5,5))
Hflat = H.reshape((-1, pixels_side*pixels_side//n//n)) 

plot_results(Wdot, W, Hflatdot, Hflat, pixels_side//n, Hs, Ws, losses)
r2_max_exp.append(sum(r2)/len(r2))
angle_max_exp.append(sum(angle)/len(angle))


plt.scatter(r2_max_exp, angle_max_exp)
plt.title("Avg R2 and Angle error (all experiments)")
plt.xlabel("Avg R2 error [-]")
plt.ylabel("Avg angle error [deg]")
plt.xlim(1, min(r2_max_exp))  
plt.ylim(0, max(angle_max_exp)) 

annotations = ['baseline', 'regulatization ($\lambda$ = 10, $\mu$ = 10)', 'smoothing median filter of size (1,10,10)', 'downsampling (16->8->4->2->1)', 'Intermediate smoothing']

# Add annotations to each data point
for i in range(len(r2_max_exp)):
    plt.annotate(annotations[i], xy=(r2_max_exp[i], angle_max_exp[i]), ha='center')


plt.grid()
plt.show()




n = 5
m = 5





lambdas = [10**((i//n)-1) for i in range(n*m)]
mus = [10**((i%n)-1) for i in range(n*m)]
for j in range(n*m):
    angle_max_exp.append([])
    r2_max_exp.append([])
    times_exp.append([])

    X = spim.X
    est = SmoothNMF( n_components = 3,tol=0.0001, max_iter = 1000, G = None, lambda_L= lambdas[j], mu=mus[j], force_simplex=False, init="nndsvdar")
    Ximg = X.reshape((-1, pixels_side,pixels_side))
    Hflat =None
    W = None

    for i in range(len(downsampling_factors)):
        start_time = time.time()
        n = downsampling_factors[i]
        X_ = downsample(Ximg,n)
        Xflat = X_.reshape((X_.shape[0], -1))

        
        est = SmoothNMF( n_components = 3,tol=0.0001, max_iter = 1000, G = None, lambda_L=lambdas[j], mu=mus[j], force_simplex=False, init="nndsvdar")

        D , Hs, Ws, losses = est.fit_transform(Xflat)

        Hflat = est.H_
        W = est.W_

        Wdot = spim.phases
        Hflatdot = downsample_flat(spim.maps,n, pixels_side)
        times_exp[-1].append(time.time()-start_time)

        angle, true_inds = find_min_angle(Wdot.T, W.T, unique=True, get_ind=True)
        r2 = ordered_r2(Hflatdot, Hflat[true_inds,:], true_inds)

        angle_max_exp[-1].append(sum(angle)/len(angle))
        r2_max_exp[-1].append(sum(r2)/len(r2))

        # plot_results(Wdot, W, Hflatdot, Hflat, pixels_side//n, Hs, Ws, losses)

    angles.append(sum(angle)/len(angle))
    r2s.append(sum(r2)/len(r2))
    

  
n = 5
m = 5
labels = ["$\mu$ ="+str(mus[i])+"$\lambda$ ="+str(lambdas[i]) for i in range(n*m)]

from matplotlib.colors import LinearSegmentedColormap

cmap_red_green = LinearSegmentedColormap.from_list("gradients", [(1,0,0),(0,1,0)], N=100)
cmap_green_red = LinearSegmentedColormap.from_list("gradients", [(0,1,0),(1,0,0)], N=100)

X = np.array(mus)
X = np.reshape(mus, (n,m))
Y = np.array(lambdas)
Y = np.reshape(lambdas, (n,m))
Z = [angle_max_exp[i][-1] for i in range(n*m)]
Z = np.array(Z)
Z = np.reshape(Z, (n,m))
ax = plt.axes(projection='3d')
ax.plot_surface(np.log10(X), np.log10(Y), Z )
ax.set_xlabel('$\mu$ (log)')
ax.set_ylabel('$\lambda$ (log)')
ax.set_zlabel('Angle error [deg]')
ax.set_title('Angle errors')
plt.show()

plt.imshow(Z, cmap=cmap_green_red,origin='lower')
plt.yticks([i for i in range(n)], [str(10**((i)-1)) for i in range(n)])
plt.xticks([i for i in range(m)], [str(10**((i)-1)) for i in range(m)])
plt.ylabel("$\lambda$")
plt.xlabel("$\mu$")
cbar = plt.colorbar()
plt.title("Avg. angle error [deg]")
plt.draw()
plt.show()

X = np.array(mus)
X = np.reshape(mus, (n,m))
Y = np.array(lambdas)
Y = np.reshape(lambdas, (n,m))
Z = [r2_max_exp[i][-1] for i in range(n*m)]
Z = np.array(Z)
Z = np.reshape(Z, (n,m))
ax = plt.axes(projection='3d')
ax.plot_surface(np.log10(X), np.log10(Y), Z )
ax.set_xlabel('$\mu$ (log)')
ax.set_ylabel('$\lambda$ (log)')
ax.set_zlabel('R2 error [-]');
ax.set_title('R2 errors')
plt.show()


plt.imshow(Z, cmap=cmap_red_green,origin='lower')
plt.yticks([i for i in range(n)], [str(10**((i%n)-1)) for i in range(n)])
plt.xticks([i for i in range(m)], [str(10**((i%n)-1)) for i in range(m)])
plt.ylabel("$\lambda$")
plt.xlabel("$\mu$")
cbar = plt.colorbar()
plt.title("Avg. R2 error [-]")
plt.draw()
plt.show()

for i in range(len(angle_max_exp)):
    plt.plot(downsampling_factors, angle_max_exp[i])

plt.yscale("log")
plt.grid()
plt.ylim(bottom=0)
plt.title("Angle errors vs Downsampling factor")
plt.xlabel("N. Pixels/side")
plt.ylabel("Worst phase angle error [deg]")
plt.legend(labels)
plt.xlim(max(downsampling_factors), 0)  # decreasing time
plt.xticks(downsampling_factors)
plt.show()

for i in range(len(angle_max_exp)):
    plt.plot(downsampling_factors, times_exp[i])
plt.grid()
plt.ylim(bottom=0)
plt.title("Computation time vs Dowsnampling factor")
plt.xlabel("Downsampling factor")
plt.ylabel("Time to convergence [s]")
plt.legend(labels)
plt.xlim(max(downsampling_factors), 0)  # decreasing time
plt.xticks(downsampling_factors)
plt.show()

for i in range(len(angle_max_exp)):
    plt.plot(downsampling_factors, r2_max_exp[i])
plt.grid()
plt.ylim(top=1.2)
plt.title("R2 vs Dowsnampling factor")
plt.xlabel("Downsampling factor")
plt.ylabel("Worst phase R2 [-]")
plt.legend(labels)
plt.xlim(max(downsampling_factors), 0)  # decreasing time
plt.xticks(downsampling_factors)
plt.show()



plt.scatter(angles, r2s)
plt.title("R2 vs Angle error (all experiments, all upsampling iterations, all phases)")
plt.xlabel("Angle error [deg]")
plt.xlabel("Phase R2 [-]")
plt.grid()
plt.show()




