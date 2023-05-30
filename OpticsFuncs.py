# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 13:49:41 2022

@author: joche
"""

from LightPipes import *
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
from matplotlib import cm

def beam_width(Field, GridDimension, GridSize):
    I = Intensity(Field)
    half_index = int(GridDimension/2)
    ref_intensity = I[half_index,half_index]/np.exp(2)
    w = np.abs((half_index - np.argmin(np.abs(I[half_index]-ref_intensity))))*GridSize/GridDimension
    return w

def norm_gaussian_width(norm_z):
    return np.sqrt(1+(norm_z)**2)

def chi_squared(model_params, model, x_data, y_data):
    return np.sum((y_data - model(x_data, model_params))**2)



def xy_Intensity_plotter(Field, GridDimension, GridSize, title):
    I = Intensity(Field)
    fig = plt.figure()
    image = plt.imshow(I,cmap='plasma',vmin=0,vmax=0.35)
    # image = plt.imshow(I,cmap='plasma',vmin=0,vmax=1)
    fig.colorbar(mappable=image,label='Normalised intensity')
    gridticks = np.linspace(0,GridDimension,7)
    spaceticks = np.round(np.linspace(-GridSize/2/mm,GridSize/2/mm,7),2)
    plt.xticks(gridticks, spaceticks)
    plt.yticks(gridticks, spaceticks)
    plt.title(title)
    plt.xlabel("x-axis (mm)")
    plt.ylabel("y-axis (mm)")
    plt.show()
    return

def waterfall_plotter_mesh(Field, GridDimension, GridSize, title, del_z, steps, z):
    I_matrix = np.empty((GridDimension,steps+1))
    z_step = del_z/steps
    z_values = np.empty((1,steps+1))
    z_values[0,0] = z
    
    middle_index = int(GridDimension/2)
    I_matrix[:,0] = Intensity(Field)[:,middle_index]
    for i in range(1,steps+1):
        Field = Forvard(Field, z_step)
        z_values[0,i] = z + z_step*i
        I_matrix[:,i] = Intensity(Field)[:,middle_index]
    x_values = list(np.linspace(-GridSize/2,GridSize/2,GridDimension))
    X = np.dstack(([x_values]*(steps+1)))[0]
    Y = np.vstack(([z_values]*GridDimension))

    #Generate plot
    colours = cm.viridis(I_matrix)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, I_matrix, rcount=50, ccount=50,
                           facecolors=colours, shade=False)
    surf.set_facecolor((0,0,0,0))
    fig.colorbar(surf)
    plt.show()
    
    return Field, z_values[0,-1]

def waterfall_plotter_lines(Field, GridDimension, GridSize, title, del_z, steps, z, zoomfactor=1):
    zoomed_gridDim = int(GridDimension/zoomfactor)
    first_gridIndex = int((GridDimension - zoomed_gridDim)/2)
    rest_factor = GridDimension - (first_gridIndex*2 + zoomed_gridDim)
    I_matrix = np.empty((zoomed_gridDim,steps+1))
    z_step = del_z/steps
    z_values = np.empty((1,steps+1))
    z_values[0,0] = z
    
    middle_index = int(GridDimension/2)
    if first_gridIndex == 0:
        I_matrix[:,0] = Intensity(Field)[:,middle_index]
        for i in range(1,steps+1):
            Field = Forvard(Field, z_step)
            I_matrix[:,i] = Intensity(Field)[:,middle_index]
            xy_Intensity_plotter(Field, GridDimension, GridSize, "Field at step: {}".format(i))
    else:
        I_matrix[:,0] = Intensity(Field)[first_gridIndex+rest_factor:-first_gridIndex,middle_index]
        for i in range(1,steps+1):
            Field = Forvard(Field, z_step)
            I_matrix[:,i] = Intensity(Field)[first_gridIndex+rest_factor:-first_gridIndex,middle_index]
            xy_Intensity_plotter(Field, GridDimension, GridSize, "Field at step: {}".format(i))
            
    x_values = list(np.linspace(-GridSize/2/zoomfactor,GridSize/2/zoomfactor,zoomed_gridDim)*1000)
    # print(x_values)
    X = np.dstack(([x_values]*(steps+1)))[0]
    z_values = np.linspace(z,z+del_z,steps+1)
    Y = np.vstack(([z_values]*zoomed_gridDim))
    #Generate plot
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # waterfall_plot(fig,ax,X,Y,I_matrix,linewidth=2,alpha=0.5) 
    # ax.set_xlabel('x-axis (mm)'); ax.set_ylabel('z-axis (m)')
    # plt.title(title)
    # fig.tight_layout()   
    
    # ax2 = fig.add_axes((0.07,0.5,0.3,0.4))
    image = plt.imshow(I_matrix,interpolation=None,aspect='auto',vmin=0,vmax=1,cmap='plasma')
    fig.colorbar(mappable=image, label="Normalised Intensity")
    gridticks = np.linspace(0,zoomed_gridDim,5)
    spaceticks = np.round(np.linspace(-GridSize/2/mm/zoomfactor,GridSize/2/mm/zoomfactor,5),2)
    z_grid_ticks = np.linspace(0,steps,5)[:-1]
    z_space_ticks = np.round(np.linspace(z_values[0],z_values[-1],5),2)#[:-1]
    plt.yticks(gridticks, spaceticks)
    plt.xticks(z_grid_ticks, z_space_ticks)
    plt.ylabel("Perpendicular distance (mm)")
    plt.xlabel("Propagation distance (m)")
    plt.show()
    
    return Field, z_values[-1]

# def colour_map(Field, GridDimension, GridSize, title, del_z, steps, z):

def waterfall_plotter_gaussian_test(Field, GridDimension, GridSize, title, del_z, steps, z, zoomfactor=1):
    zoomed_gridDim = int(GridDimension/zoomfactor)
    first_gridIndex = int((GridDimension - zoomed_gridDim)/2)
    rest_factor = GridDimension - (first_gridIndex*2 + zoomed_gridDim)
    I_matrix = np.empty((zoomed_gridDim,steps+1))
    z_step = del_z/steps
    z_values = np.empty((1,steps+1))
    z_values[0,0] = z
    
    w_old = beam_width(Field, GridDimension, GridSize)
    counter = 0        
    middle_index = int(GridDimension/2)
    if first_gridIndex == 0:
        I_matrix[:,0] = Intensity(Field)[:,middle_index]
        for i in range(1,steps+1):
            Field = Forvard(Field, z_step)
            I_matrix[:,i] = Intensity(Field)[:,middle_index]
            w_new = beam_width(Field, GridDimension, GridSize)
            print(w_new)
            if w_new == w_old:
                counter += 1
            elif w_new < w_old:
                counter = 0
            if w_new > w_old:
                z += z_step*(i-1-int((counter+1)/2))
                waist_index = i
                break
            w_old = w_new
        for i in range(i,steps+1):
            Field = Forvard(Field, z_step)
            I_matrix[:,i] = Intensity(Field)[:,middle_index]
    else:
        I_matrix[:,0] = Intensity(Field)[first_gridIndex+rest_factor:-first_gridIndex,middle_index]
        for i in range(1,steps+1):
            Field = Forvard(Field, z_step)
            I_matrix[:,i] = Intensity(Field)[first_gridIndex+rest_factor:-first_gridIndex,middle_index]
            w_new = beam_width(Field, GridDimension, GridSize)
            print(w_new)
            if w_new == w_old:
                counter += 1
            elif w_new < w_old:
                counter = 0
            if w_new > w_old:
                z += z_step*(i-1-int((counter+1)/2))
                waist_index = i
                break
            w_old = w_new
        for i in range(i,steps+1):
            Field = Forvard(Field, z_step)
            I_matrix[:,i] = Intensity(Field)[first_gridIndex+rest_factor:-first_gridIndex,middle_index]
    
    del_z_lens_waist = z - z_values[0,0]
    x_values = list(np.linspace(-GridSize/2/zoomfactor,GridSize/2/zoomfactor,zoomed_gridDim)*1000)
    X = np.dstack(([x_values]*(steps+1)))[0]
    z_values = np.linspace(z,z+del_z,steps+1)
    Y = np.vstack(([z_values]*zoomed_gridDim))
    #Generate plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    waterfall_plot(fig,ax,X,Y,I_matrix,linewidth=2,alpha=0.5) 
    ax.set_xlabel('x-axis (mm)'); ax.set_ylabel('z-axis (m)')
    plt.title(title)
    fig.tight_layout()   
    
    ax2 = fig.add_axes((0.07,0.5,0.3,0.4))
    image = plt.imshow(I_matrix,interpolation=None,cmap='plasma',aspect='auto')
    fig.colorbar(mappable=image)
    plt.plot([waist_index,waist_index],[0,zoomed_gridDim],linestyle='--',color='red')
    gridticks = np.linspace(0,zoomed_gridDim,5)
    spaceticks = np.round(np.linspace(-GridSize/2/mm/zoomfactor,GridSize/2/mm/zoomfactor,5),2)
    z_grid_ticks = np.linspace(0,steps,5)[:-1]
    z_space_ticks = np.round(np.linspace(z_values[0],z_values[-1],5),2)[:-1]
    plt.yticks(gridticks, spaceticks)
    plt.xticks(z_grid_ticks, z_space_ticks)
    plt.ylim([0,zoomed_gridDim])
    plt.xlim([0,steps])
    # plt.ylabel("x-axis (mm)")
    plt.xlabel("z-axis (m)")
    plt.show()
    
    return Field, z_values[-1], del_z_lens_waist   


def waterfall_plot(fig,ax,X,Y,Z,cb_label="Normalised intensity", **kwargs):
    '''
    Make a waterfall plot
    Input:
        fig,ax : matplotlib figure and axes to populate
        Z : n,m numpy array. Must be a 2d array even if only one line should be plotted
        X,Y : n,m array
        kwargs : kwargs are directly passed to the LineCollection object
    '''
    # Set normalization to the same values for all plots
    norm = plt.Normalize(Z.min().min(), Z.max().max())
    # Check sizes to loop always over the smallest dimension
    n,m = Z.shape
    if n>m:
        X=X.T; Y=Y.T; Z=Z.T
        m,n = n,m

    for j in range(n):
        # reshape the X,Z into pairs 
        points = np.array([X[j,:], Z[j,:]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)  
        # The values used by the colormap are the input to the array parameter
        lc = LineCollection(segments, cmap='plasma', norm=norm, array=(Z[j,1:]+Z[j,:-1])/2, **kwargs)
        line = ax.add_collection3d(lc,zs=(Y[j,1:]+Y[j,:-1])/2, zdir='y') # add line to axes

    fig.colorbar(lc, label=cb_label) # add colorbar, as the normalization is the same for all
    # it doesent matter which of the lc objects we use
    ax.auto_scale_xyz(X,Y,Z) # set axis limits   
    return



# X, Y, Z = axes3d.get_test_data(0.2)

# #Example of mesh waterfall plot:
# # Normalize to [0,1]
# norm = plt.Normalize(Z.min(), Z.max())
# colors = cm.viridis(norm(Z))
# rcount, ccount, _ = colors.shape

# fig = plt.figure()

# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(X, Y, Z, rcount=rcount, ccount=ccount,
#                         facecolors=colors, shade=False)
# # surf = ax.plot_surface(X, Y, Z, facecolors=colors, shade=False)
# surf.set_facecolor((0,0,0,0))
# fig.colorbar(surf)
# plt.show()

# #Example of lines waterfall plot:
# # Generate waterfall plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# waterfall_plot(fig,ax,X,Y,Z,linewidth=1.5,alpha=0.5) 
# ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z') 
# fig.tight_layout()   
 