# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 22:10:58 2022

@author: joche
"""

from LightPipes import Field
import numpy as _np
from OpticsFuncs import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def new_Axicon(Fin, phi, n1 = 1.5, x_shift = 0.0, y_shift = 0.0 ):
    """
    *Propagates the field through an axicon.*
    
    :param Fin: input field
    :type Fin: Field
    :param phi: top angle of the axicon in radiants
    :type phi: int, float
    :param n1: refractive index of the axicon material (default = 1.5)
    :type phi: int, float
    :param x_shift: shift in x direction (default = 0.0)
    :type x_shift: int, float
    :param y_shift: shift in y direction (default = 0.0)
    :type y_shift: int, float
    :return: output field (N x N square array of complex numbers).
    :rtype: `LightPipes.field.Field`
    :Example:
    
    >>> phi=179.7/180*3.1415
    >>> F = Axicon(F, phi) # axicon with top angle phi, refractive index = 1.5, centered in grid
    >>> F = Axicon(F,phi, n1 = 1.23, y_shift = 2*mm) # Idem, refractive index = 1.23, shifted 2 mm in y direction
    >>> F = Axicon(F, phi, 1.23, 2*mm, 0.0) # Idem
     
    .. seealso::
    
        * :ref:`Example: Bessel beam with axicon <Generation of a Bessel beam with an axicon.>`
    """
    Fout = Field.copy(Fin) #Creates copy of input field = will become output
    k = 2*_np.pi/Fout.lam # k = 2 pi / lambda
    theta = _np.tan(_np.pi-phi)*(n1-1) #Changed phase shift
    Ktheta = k * theta
    yy, xx = Fout.mgrid_cartesian
    xx -= x_shift
    yy -= y_shift
    fi = -Ktheta*_np.sqrt(xx**2+yy**2)
    Fout.field *= _np.exp(1j*fi)
    Fout._IsGauss=False
    return Fout

def real_Axicon_old(Fin, phi, n1 = 1.5, R = 0, d = 0, w=0.1, x_shift = 0.0, y_shift = 0.0):
    """
    *Propagates the field through an axicon with non-negligible thickness and rounded tip.*
    
    :param Fin: input field
    :type Fin: Field
    :param phi: top angle of the axicon in radiants
    :type phi: int, float
    :param n1: refractive index of the axicon material (default = 1.5)
    :type n1: int, float
    :param R: radius of the imperfect tip in m (default = 0, i.e. perfect sharp tip)
    :type R: int, float
    :param d: thickness part of the axicon before the conical surface in m -also called edge thickness- (default = 0)
    :type d: int, float
    :param w: radius of the axicon in m (default = 10 cm), the function assumes the field outside of the axicon to be neglible
    :type w: int, float
    :param x_shift: shift in x direction (default = 0.0)
    :type x_shift: int, float
    :param y_shift: shift in y direction (default = 0.0)
    :type y_shift: int, float
    :return: output field (N x N square array of complex numbers).
    :rtype: `LightPipes.field.Field`

    """
    Fout = Field.copy(Fin) #Creates copy of input field = will become output
    k = 2*_np.pi/Fout.lam # k = 2 pi / lambda
    alpha = _np.pi - phi #Angle conical surface with vertical
    yy, xx = Fout.mgrid_cartesian
    xx -= x_shift #Shifting for the position of axicon
    yy -= y_shift
    r_sq = xx**2 + yy**2 #Radius squared
    
    #Phase for conical section:
    t_conical = d + w*_np.tan(alpha) #Global phase, used for a non-thin axicon
    
    tip_r = R*_np.sin(alpha) #Radius where the conical and circular surface intersect
    tip_r_sq = tip_r**2 
    
    phase = _np.empty_like(r_sq) 
    phase[r_sq > tip_r_sq] = (1-n1)*_np.sqrt(r_sq[r_sq > tip_r_sq])*_np.tan(alpha) + n1*t_conical
    #Curvature + global phase for this section
    #(1-n)*(|r|*tan(alpha)+n*thickness)
    
    #Phase for circular section:
    #Global phase, used for a non-thin axicon
    circ_tip_diff = tip_r*_np.tan(alpha) - R*(1 - _np.cos(alpha))
    #^From that intersection point: length to tip of conical surface - length to edge of circle
    t_circ = t_conical - circ_tip_diff 
    phase[r_sq <= tip_r_sq] = (1-n1)*(R-_np.sqrt(R**2-r_sq[r_sq <= tip_r_sq])) + n1*t_circ
    #(1-n)*(R-sqrt(R^2+r^2) + n*thickness)
    
    Fout.field *= _np.exp(1j*k*phase)
    
    Fout._IsGauss=False
    return Fout

def real_Axicon(Fin, phi, n1 = 1.5, R = 0, d = 0, w=0.1, x_shift = 0.0, y_shift = 0.0, geo_adjust=False):
    """
    *Propagates the field through an axicon with non-negligible thickness and rounded tip.*
    
    :param Fin: input field
    :type Fin: Field
    :param phi: top angle of the axicon in radiants
    :type phi: int, float
    :param n1: refractive index of the axicon material (default = 1.5)
    :type n1: int, float
    :param R: radius of the imperfect tip in m (default = 0, i.e. perfect sharp tip)
    :type R: int, float
    :param d: thickness part of the axicon before the conical surface in m -also called edge thickness- (default = 0)
    :type d: int, float
    :param w: radius of the axicon in m (default = 10 cm), the function assumes the field outside of the axicon to be neglible
    :type w: int, float
    :param x_shift: shift in x direction (default = 0.0)
    :type x_shift: int, float
    :param y_shift: shift in y direction (default = 0.0)
    :type y_shift: int, float
    :param geo_adjust: Whether or not to use the geometric adjustment of the points in the grid (default = False)
    :type geo_adjust: bool
    :return: output field (N x N square array of complex numbers).
    :rtype: `LightPipes.field.Field`

    """
    
    Fout = Field.copy(Fin) #Creates copy of input field = will become output
    k = 2*_np.pi/Fout.lam # k = 2 pi / lambda
    alpha = _np.pi - phi #Angle conical surface with vertical
    yy, xx = Fout.mgrid_cartesian
    xx -= x_shift #Shifting for the position of axicon
    yy -= y_shift
    r_sq = xx**2 + yy**2 #Radius squared
    abs_r = _np.sqrt(r_sq)
    
    #Dimensions of axicon:
    t_conical = d + w*_np.tan(alpha) #Global phase, used for a non-thin axicon
    
    tip_r = R*_np.sin(alpha) #Radius where the conical and circular surface intersect
    tip_r_sq = tip_r**2 
    
    #Adjustment of path length through air:
        #For conical surface:
    gamma = _np.empty_like(r_sq)
    gamma[r_sq > tip_r_sq] = _np.arcsin(n1*_np.sin(alpha))-alpha
    
        #For rounded tip:
    circ_ang = _np.empty_like(r_sq)
    circ_ang[r_sq <= tip_r_sq] = _np.arcsin(abs_r[r_sq <= tip_r_sq] / R)
    gamma[r_sq <= tip_r_sq] = _np.arcsin(n1*_np.sin(circ_ang[r_sq <= tip_r_sq]))-circ_ang[r_sq <= tip_r_sq]
    inv_cos_gamma = 1/_np.cos(gamma)
    
    #Conical surface phase shift:
    phase = _np.empty_like(r_sq) 
    phase[r_sq > tip_r_sq] = (inv_cos_gamma[r_sq > tip_r_sq]-n1)*abs_r[r_sq > tip_r_sq]*_np.tan(alpha) + n1*t_conical
    #Curvature + global phase for this section
    #(1-n)*(|r|*tan(alpha)+n*thickness)
    
    #Circular tip phase shift:
    #Global phase, used for a non-thin axicon
    circ_tip_diff = tip_r*_np.tan(alpha) - R*(1 - _np.cos(alpha))
    #^From that intersection point: length to tip of conical surface - length to edge of circle    
    t_circ = t_conical - circ_tip_diff 
    phase[r_sq <= tip_r_sq] = (inv_cos_gamma[r_sq <= tip_r_sq]-n1)*(R-_np.sqrt(R**2-r_sq[r_sq <= tip_r_sq])) + n1*t_circ + inv_cos_gamma[r_sq <= tip_r_sq]*circ_tip_diff
    #(1-n)*(R-sqrt(R^2+r^2)) + n*thickness + thickness_difference
    Fout.field *= _np.exp(1j*k*phase)
    
    if geo_adjust == True:
        del_r = _np.empty_like(r_sq) 
        del_r[r_sq > tip_r_sq] = -(abs_r[r_sq > tip_r_sq]*_np.tan(alpha))*_np.tan(gamma[r_sq > tip_r_sq])
        del_r[r_sq <= tip_r_sq] = -(R-_np.sqrt(R**2-r_sq[r_sq <= tip_r_sq]) + circ_tip_diff)*_np.tan(gamma[r_sq <= tip_r_sq])
        GridSize = _np.max(abs(xx[:,0]), axis=0)*2
        GridDim = len(xx)
        pointsize = GridSize / GridDim
        print(pointsize)
        del_x_index = _np.zeros_like(r_sq,dtype=int) 
        del_y_index = _np.zeros_like(r_sq,dtype=int) 
        del_x_index[abs_r > 0] = ((xx[abs_r > 0]/abs_r[abs_r > 0]*del_r[abs_r > 0])/pointsize).astype(int)
        del_y_index[abs_r > 0] = ((yy[abs_r > 0]/abs_r[abs_r > 0]*del_r[abs_r > 0])/pointsize).astype(int)
        print(del_x_index)
        # print(del_r)
        # print(abs_r)
        # print(xx)
        fieldgrid = _np.copy(Fout.field)
        xy_Intensity_plotter(Fout, GridDim, GridSize, "old")
        Fout.field = _np.zeros_like(fieldgrid)
        for i in range(GridDim):
            for j in range(GridDim):
                Fout.field[i+del_x_index[i,j],j+del_y_index[i,j]] += fieldgrid[i,j]
        xy_Intensity_plotter(Fout, GridDim, GridSize, "adjusted")
        print(Fout.field[Fout.field>0])
    Fout._IsGauss=False
    return Fout

def plot_real_Axicon(Fin, phi, n1 = 1.5, R = 0, d = 0, w=0.1, x_shift = 0.0, y_shift = 0.0, n_lines = 10):
    """
    *Propagates the field through an axicon with non-negligible thickness and rounded tip.*
    
    :param Fin: input field
    :type Fin: Field
    :param phi: top angle of the axicon in radiants
    :type phi: int, float
    :param n1: refractive index of the axicon material (default = 1.5)
    :type n1: int, float
    :param R: radius of the imperfect tip in m (default = 0, i.e. perfect sharp tip)
    :type R: int, float
    :param d: thickness part of the axicon before the conical surface in m -also called edge thickness- (default = 0)
    :type d: int, float
    :param w: radius of the axicon in m (default = 10 cm), the function assumes the field outside of the axicon to be neglible
    :type w: int, float
    :param x_shift: shift in x direction (default = 0.0)
    :type x_shift: int, float
    :param y_shift: shift in y direction (default = 0.0)
    :type y_shift: int, float
    :param n_lines: number of lines to be plotted
    :type n_lines: int

    """
    Fout = Field.copy(Fin) #Creates copy of input field = will become output
    k = 2*_np.pi/Fout.lam # k = 2 pi / lambda
    alpha = _np.pi - phi #Angle conical surface with vertical
    yy, xx = Fout.mgrid_cartesian
    X = _np.copy(xx)*1000
    Y = _np.copy(yy)*1000
    xx -= x_shift #Shifting for the position of axicon
    yy -= y_shift
    r_sq = xx**2 + yy**2 #Radius squared
    
    #Thickness for conical section:
    t_conical = d + w*_np.tan(alpha) #Global thickness of perfect tip axicon, used for a non-thin axicon
    
    tip_r = R*_np.sin(alpha) #Radius where the conical and circular surface intersect
    tip_r_sq = tip_r**2 
    
    thickness = _np.empty_like(r_sq) 
    thickness[r_sq > tip_r_sq] =  t_conical - _np.sqrt(r_sq[r_sq > tip_r_sq])*_np.tan(alpha)
    #max thickness of conical section - |r|*tan(alpha)
    
    #Phase for circular section:
    #Global thickness of actual axicon with rounded tip, used for a non-thin axicon
    circ_tip_diff = tip_r*_np.tan(alpha) - R*(1 - _np.cos(alpha))
    #^From that intersection point: length to tip of conical surface - length to edge of circle
    t_circ = t_conical - circ_tip_diff 
    thickness[r_sq <= tip_r_sq] = _np.sqrt(R**2-r_sq[r_sq <= tip_r_sq]) - R + t_circ
    #sqrt(R^2+r^2) - R + max thickness of circ. section
    thickness *= 1000
    
    #Generate twoD projection data:
    twoD_x = X[0,:]
    twoD_y = Y[:,0]
    twoD_x_thickness = _np.max(thickness, axis=0)
    twoD_y_thickness = _np.max(thickness, axis=1)
    
    print("The total thickness of the Axicon is: {} cm".format(_np.max(twoD_x_thickness)/10))
    print("The radius of the tip of the Axicon is: {} cm".format(R*100))
    print("The angle of conical surface to the vertical is: {} degrees".format(alpha/_np.pi*180))
    print("The diameter of the Axicon is: {} cm".format(2*w*100))
    print("The refractive index of the Axicon is: {}".format(n1))
    
    #Reducing number of lines:
    # GridDimension = len(X)
    # averaging_n = int(GridDimension/n_lines)
    # print(X,Y,thickness)
    # thickness = _np.mean(thickness.reshape(GridDimension,n_lines,averaging_n), axis=2)
    # X = _np.mean(X.reshape(GridDimension,n_lines,averaging_n), axis=2)
    # Y = Y.reshape(GridDimension,n_lines,averaging_n)[:,:,0]
    # print(X,Y,thickness)
    
    #Reducing number of lines:
    GridDimension = len(X)
    averaging_n = int(GridDimension/n_lines)
    thickness = _np.mean(thickness.reshape(n_lines,averaging_n,GridDimension), axis=1)
    Y = _np.mean(Y.reshape(n_lines,averaging_n,GridDimension), axis=1)
    X = X.reshape(n_lines,averaging_n,GridDimension)[:,0,:]
    
    #Generate plot
    print("Generating plot")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    waterfall_plot(fig,ax,X,Y,thickness,linewidth=2,alpha=0.8,cb_label='Thickness (mm)') 
    ax.plot(twoD_x, twoD_x_thickness, 'gray', zdir='y', zs=twoD_y[-1])
    ax.plot(twoD_y, twoD_y_thickness, 'gray', zdir='x', zs=twoD_x[0])
    ax.set_xlabel('x-axis (mm)'); ax.set_ylabel('y-axis (mm)');
    fig.tight_layout()
    
    # plt.figure()
    # plt.plot(twoD_x, twoD_x_thickness, 'gray')
    
    
    return