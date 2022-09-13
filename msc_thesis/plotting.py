import numpy as np
import matplotlib.pyplot as plt
from esmpy.measures import find_min_angle,  ordered_r2

plt.rcParams.update({'font.family':'serif'})
plt.rcParams.update({'font.size': 18})

def plot_synthetic_dataset(Ddot, Hdot, xaxis, P1, P2):

    cmap = plt.cm.gray_r
    K = Hdot.shape[0]
    L = Ddot.shape[0]
  
    print(";....................................")

    max_intensity = np.max(Ddot)

    fig = plt.figure(figsize = (10,6))
    gs = fig.add_gridspec(3, 3, width_ratios=[1,.4,3], height_ratios=[1,1,1]) 

    for i in range(K): 
         
        ax1 = fig.add_subplot(gs[i, 0])
        ax3 = fig.add_subplot(gs[i, 2])

        if i == 0:                
            ax1.set_title('True maps')
            ax3.set_title('True spectra')

        if i==K-1:
            ax3.set_xlabel('Energy [keV]')

        else: 
            ax3.set_xticks([])

        im = ax1.imshow(Hdot[i].reshape(P1,P2), cmap=cmap, vmin=0, vmax=1)
        ax1.tick_params(axis = "both",labelleft = False, labelbottom = False,left = False, bottom = False)
        ax3.plot(xaxis,Ddot.T[i,:],'r-',markersize=3.5)
        ax3.set_ylabel("Intensity [-]")
        ax3.set_ylim(0,max_intensity)

        if i == 0:
            ax3.legend()

    plt.subplots_adjust(wspace=0.0, hspace=0.08)
    plt.subplots_adjust(bottom=0.05)

    cbar_ax = fig.add_axes([0.15, 0.0, 0.15, 0.01])
    fig.colorbar(im,cax=cbar_ax, orientation='horizontal')



def plot_results_report(Ddot, D, Hdot, H, xaxis, P1, P2):


    fontsize = 15
    cmap = plt.cm.gray_r
    K = H.shape[0]
    L = D.shape[0]

    angles, true_inds = find_min_angle(Ddot.T, D.T, unique=True, get_ind=True)


    r2 = ordered_r2(Hdot, H, true_inds)

    print(";....................................")

    max_intensity = max(np.max(D), np.max(Ddot))

    fig = plt.figure(figsize = (21,11))
    gs = fig.add_gridspec(3, 4, width_ratios=[1,1,.2,3], height_ratios=[1,1,1]) 


    for i in range(K): 
        j= true_inds.index(i)
         
        ax1 = fig.add_subplot(gs[i, 0])
        ax2 = fig.add_subplot(gs[i, 1])
        ax3 = fig.add_subplot(gs[i, 3])

        if i == 0:                
            ax1.set_title('True maps', fontsize=fontsize)
            ax2.set_title('Reconstructed maps',  fontsize=fontsize)
            ax3.set_title('Spectra (true vs reconstructed)',  fontsize=fontsize)


        if i==K-1:
            ax3.set_xlabel('Energy [keV]', fontsize=fontsize)
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.65)

        ax2.imshow((H[j]).reshape(P1,P2), cmap=cmap)
        ax2.text(0.05, 0.95, "R2: {:.2f}".format(r2[j]), transform=ax2.transAxes, fontsize=fontsize,
        verticalalignment='top', bbox=props)
        im = ax1.imshow(Hdot[i].reshape(P1,P2), cmap=cmap)
        ax1.tick_params(axis = "both",labelleft = False, labelbottom = False,left = False, bottom = False)
        ax2.tick_params(axis = "both",labelleft = False, labelbottom = False,left = False, bottom = False)
        ax3.plot(xaxis,D[:,j],'bo',label='reconstructed',linewidth=4)
        ax3.plot(xaxis,Ddot.T[i,:],'r-',label='truth',markersize=3.5)
        ax3.text(0.02, 0.95, "Angle error: {:.2f} deg".format(angles[j]), transform=ax3.transAxes, fontsize=fontsize,
        verticalalignment='top', bbox=props)
        ax3.set_ylabel("Intensity [-]", fontsize=fontsize)
        ax3.set_ylim(0,max_intensity)

        if i == 0:
            ax3.legend()

    plt.subplots_adjust(wspace=0.0, hspace=0.08)
    plt.subplots_adjust(bottom=0.05)

    cbar_ax = fig.add_axes([0.15, 0.0, 0.3, 0.01])
    fig.colorbar(im,cax=cbar_ax, orientation='horizontal')


