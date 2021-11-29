# Point selector in PCA score plot

import numpy as np

from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import matplotlib.pyplot as plt
from time import ctime
from mpl_toolkits.mplot3d import Axes3D

class SelectFromCollection:
    """Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : :class:`~matplotlib.axes.Axes`
        Axes to interact with.

    collection : :class:`matplotlib.collections.Collection` subclass
        Collection you want to select from.

    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to `alpha_other`.
    """


    def __init__(self, Spca, Shier, S_H_mean, S_H_std , alpha_other=0.3):
        
        
        #Spectral attributes
        self.Spca = Spca
        self.Shier = Shier
        self.elevel = len(self.Shier[0,:])
        self.selectedSpectra = []
        self.accepted = False
        self.S_H_mean = S_H_mean
        self.S_H_std = S_H_std
        
        self.title = 'Circle the desired spectra with your mouse'
        
        
        # User defined input : shown components
        print('\n####################')
        print('\nSelect PCA components to project (1,2,...n)')
        
        self.PCX = int(input('Horizontal component : '))-1
        self.PCY = int(input('Vertical component : '))-1
        
        # Plot data
        subplot_kw = dict()
        fig, ax = plt.subplots(subplot_kw=subplot_kw)
        self.canvas = ax.figure.canvas
        self.collection = ax.scatter(Spca[:, self.PCX], Spca[:, self.PCY], s=10)
        
        #Plot atributes
        self.alpha_other = alpha_other
        self.xys = self.collection.get_offsets()
        self.Npts = len(self.xys)
        
         

        # Ensure that we have separate colors for each object
        self.fc = self.collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

                
        def accept(event):
            if event.key == "enter":
                self.disconnect()
                self.accepted = True
                ax.set_title("")
                fig.canvas.draw()
                plt.close()
        
        fig.canvas.mpl_connect("key_press_event", accept)
          
        ax.set_title(('First circle the desired spectra with your mouse\nIf the mean spectra ploted is fine, click enter. You can also re-do the selection. ') , fontsize = 12)
        plt.xlabel('Principal component ' + str(self.PCX+1))
        plt.ylabel('Principal component ' + str(self.PCY+1))
        
        plt.show()
        
        
        
          
        

    def onselect(self, verts):
        
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        #print('\n \n self ind \n \n')
        print('\n')
        print(self.ind)
        

        
        self.computeSpectra()
    
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        
        self.canvas.draw_idle()

        #count = self.ind
        #np.savetxt('data_pure_'+str(np.round(100*np.random.rand(),-1))+'.txt',count)
        #np.savetxt('data_pure_'+str(count[0])+'.txt',count)
        #np.savetxt('data_pure_'+str(ctime())+'.txt',count)




    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()
        
    
    
    
    def computeSpectra(self):
        
        Si = np.mean(self.Shier[self.ind,:], axis  = 0)
        plt.figure()
        Si = Si*self.S_H_std + self.S_H_mean
        plt.plot(Si)
        plt.title('Close the window when done' , fontsize = 12)
        self.selectedSpectra = Si
        
        
        
    
    
    #return indices of selected points
    def getSelectedPoints(self):
        return self.ind
        



class Score_plot_3D:

          
    def __init__(self,Spca):
        
        self.a = int(input('PCA score for first axis (1,2,3...) : '))
        self.b = int(input('PCA score for second axis (1,2,3...) : '))
        self.c = int(input('PCA score for third axis (1,2,3...) : '))
        
        self.x = Spca[:,self.a-1]
        self.y = Spca[:,self.b-1]
        self.z = Spca[:,self.c-1]
        self.accepted = False
        
        def accept(event):
        
            if event.key == "enter":
                self.accepted = True
                ax.set_title("")
                fig2.canvas.draw()
                plt.close()
        
        
    
        fig2 = plt.figure()
        fig2.suptitle('Press enter to continue \n(close plot before)' , fontsize = 20)
        fig2.canvas.mpl_connect("key_press_event", accept)
        
        
        fig = plt.figure()
        ax = Axes3D(fig)
        
        ax.scatter(self.x  , self.y , self.z)
        
        ax.set_xlabel(('t' + str(self.a)) , fontsize = 15)
        ax.set_ylabel(('t' + str(self.b)) , fontsize = 15)
        ax.set_zlabel(('t' + str(self.c)) , fontsize = 15) 
        ax.set_title(('Score plot ' + 't' + str(self.a) + '/' + 't' + str(self.b) + '/' + 't' + str(self.c) + '  (you can drag the image and play with the graph)' + '\nClose window when you are done') , fontsize = 12)
        
        plt.show()
    