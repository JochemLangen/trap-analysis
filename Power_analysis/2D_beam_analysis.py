# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 18:03:05 2022

@author: joche
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2
from Beam_analysis import Gaussian_density, Gaussian_radius, chi_sqrd

Excel_file = pd.read_excel("Beam-profile_measurement.xlsx")

#The arrays corresponding to the first point along the z-axis:
pos_1 = np.array(Excel_file["Unnamed: 7"][1:59], dtype=float)
pos_err_1 = np.array(Excel_file["Unnamed: 8"][1:59], dtype=float)
power_d_1 = np.array(Excel_file["Unnamed: 9"][1:59], dtype=float)
power_d_err_1 = np.array(Excel_file["Unnamed: 10"][1:59], dtype=float)
#The arrays corresponding to the second point along the z-axis:
pos_2 = np.array(Excel_file["Unnamed: 17"][5:63], dtype=float)
pos_err_2 = np.array(Excel_file["Unnamed: 18"][5:63], dtype=float)
power_d_2 = np.array(Excel_file["Unnamed: 19"][5:63], dtype=float)
power_d_err_2 = np.array(Excel_file["Unnamed: 20"][5:63], dtype=float)

#The arrays corresponding to the third point along the z-axis:
pos_3 = np.array(Excel_file["Unnamed: 27"][15:73], dtype=float)
pos_err_3 = np.array(Excel_file["Unnamed: 28"][15:73], dtype=float)
power_d_3 = np.array(Excel_file["Unnamed: 29"][15:73], dtype=float)
power_d_err_3 = np.array(Excel_file["Unnamed: 30"][15:73], dtype=float)

z = np.array([13,463,1065])
z_err = np.array([0.5,0.5,np.sqrt(2)])

pos = np.dstack([pos_1,pos_2,pos_3])[0]
pos_err = np.dstack([pos_err_1,pos_err_2,pos_err_3])[0]
power = np.dstack([power_d_1,power_d_2,power_d_3])[0]
power_err = np.dstack([power_d_err_1,power_d_err_2,power_d_err_3])[0]

def Gaussian_propagation(pos,waist,A):
    w_array = Gaussian_radius(z,waist) #Note: this uses the global variable z
    return np.ravel(Gaussian_density(pos,A,w_array))

def plot_resid(norm_resid, norm_resid_err, x, plot, D, xmin, xmax, font_size, resid_y_lim):
    plot.plot([xmin,xmax],[D, D],[xmin,xmax],[-D,-D],[xmin,xmax],[0,0], color =(0.1,0.1,0.1), linestyle = '--', linewidth = 1, zorder = 1)
    plt.fill_between([xmin,xmax],[D, D],[-D, -D], facecolor=(0.9,0.9,0.9), interpolate=True, zorder = 0)
    plot.errorbar(x, norm_resid, yerr= norm_resid_err, capsize = 2, fmt='x', ms = 6, color = 'black', zorder = 3)
    
    plt.xlim([xmin,xmax])
    plt.ylim([-resid_y_lim,resid_y_lim])
    
    yticks = np.linspace(-resid_y_lim*0.6,resid_y_lim*0.6,3)
    # plt.yticks(yticks[:-1],fontsize = font_size)
    plt.yticks(yticks,fontsize = font_size)
    plt.xticks(fontsize = font_size)
    plot.tick_params(axis='both', which='major', width= 1.3, length= 5)
    plot.minorticks_on()
    plt.ylabel("Normalised Residuals", fontsize = font_size,labelpad=32) 
    return

def fitted_interval(smooth_coords, waist_fit, waist_err, A_fit, A_err):
    shape = np.shape(smooth_coords)
    smooth_y_fit = np.reshape(Gaussian_propagation(smooth_coords,waist_fit,A_fit),shape)
    smooth_y_thin_low = Gaussian_propagation(smooth_coords,waist_fit-waist_err,A_fit-A_err)
    smooth_y_thin_high = Gaussian_propagation(smooth_coords,waist_fit-waist_err,A_fit+A_err)
    smooth_y_wide_low = Gaussian_propagation(smooth_coords,waist_fit+waist_err,A_fit-A_err)
    smooth_y_wide_high = Gaussian_propagation(smooth_coords,waist_fit+waist_err,A_fit+A_err)
    fitted_array = np.dstack((smooth_y_thin_low,smooth_y_thin_high,smooth_y_wide_low,smooth_y_wide_high))[0]
    smooth_y_high = np.reshape(np.max(fitted_array, axis=1),shape)
    smooth_y_low = np.reshape(np.min(fitted_array, axis=1),shape)
    return smooth_y_fit, smooth_y_high, smooth_y_low

def perform_2D_gaussfit(coords,y,y_err,x_err, A_guess, waist_guess):
    
    raveled_y_err = np.ravel(y_err)
    raveled_y = np.ravel(y)
    popt, pcov = curve_fit(Gaussian_propagation,coords,raveled_y,p0=[waist_guess,A_guess],sigma=raveled_y_err,absolute_sigma=True)
    waist_fit, A_fit = popt
    waist_err = pcov[0,0]
    A_err = pcov[1,1]
    fitted_y = Gaussian_propagation(coords,waist_fit,A_fit)
    w_array = Gaussian_radius(z,waist_fit)
    chi = chi_sqrd(fitted_y,raveled_y,raveled_y_err)
    degrees_f = len(raveled_y)-2
    chi_red = chi / degrees_f
    P = 1-chi2.cdf(chi,degrees_f,scale=1)
    if P == 0.0:
        print("The cumulative distribution function P has a value of less than {}".format(1e-16))
    else:
        print("The cumulative distribution function P has a value of {}".format(P))
    print("The reduced chi squared value is: {}".format(chi_red))
    print("A = {} +/- {} W/mm".format(A_fit,A_err))
    print("W_0 = {} +/- {} mm".format(waist_fit,waist_err))
    
    #Plotting:
    x_spacing = 0.3
    x_ends = np.array([np.amin(coords)-x_spacing,np.amax(coords)+x_spacing])    
        
    smooth_coords = np.dstack([np.linspace(x_ends[0],x_ends[1],100)]*3)[0]
    smooth_y_fit, smooth_y_high, smooth_y_low = fitted_interval(smooth_coords, waist_fit, waist_err, A_fit, A_err)
    
    #Calculate residuals:
    raveled_norm_residuals = (raveled_y-fitted_y)/raveled_y_err
    norm_resid = np.reshape(raveled_norm_residuals,(58,3))
    y_fit, y_high, y_low = fitted_interval(coords, waist_fit, waist_err, A_fit, A_err)
    y_fit_err = (y_high-y_low)/2
    D=2
    D_percentages = np.array(norm_resid <= D).sum(axis=0)/58*100
    print(r'The percentages of the residuals within the {} sigma interval are: {}%, {}% & {}%'.format(D,D_percentages[0],D_percentages[1],D_percentages[2]))

    resid_y_lim=10
    font_size=18
    
    fig = plt.figure()
    figure_ratio = 2
    colours = ['red', 'blue', 'green']
    for i in range(3):
        ax = fig.add_axes((i/figure_ratio,0,1/figure_ratio,1))
        ax.errorbar(coords[:,i],y[:,i],xerr=x_err[:,i],yerr=y_err[:,i], fmt='xk',capsize=2)
        ax.plot(smooth_coords[:,i],smooth_y_fit[:,i], color = colours[i])
        dotted_colour = (0.5,0.1,0.1)

        plt.plot([-w_array[i],w_array[i]],Gaussian_density([-w_array[i],w_array[i]],A_fit,w_array[i]),linestyle='--', color=colours[i], alpha=0.8)
        ax.plot(smooth_coords[:,i],smooth_y_high[:,i], linestyle=':', color=dotted_colour)
        ax.plot(smooth_coords[:,i],smooth_y_low[:,i], linestyle=':', color=dotted_colour)
        ax.fill_between(smooth_coords[:,i],smooth_y_low[:,i],smooth_y_high[:,i], color=colours[i],alpha=0.2)

        plt.xlim([x_ends[0],x_ends[1]])
        plt.ylim([-10,500])

        ax.minorticks_on()
        plt.tick_params(axis='both', which='major', width= 1.3, length= 5)
        plt.xticks([])
        if i == 0:
            plt.ylabel("1D Power density (W/mm)", fontsize = font_size) 
            yticks = ax.get_yticks()
            plt.yticks(yticks[1:-1], fontsize = font_size) 
        else:
            plt.yticks([])
        
        ax_resid = fig.add_axes((i/figure_ratio,-0.2,1/figure_ratio,0.2))
        plot_resid(norm_resid[:,i], None, coords[:,i], ax_resid, D, x_ends[0],x_ends[1], font_size, resid_y_lim)
        xticks = ax_resid.get_xticks()
        plt.xticks(xticks[1:-1], fontsize = font_size) 
            
        if i == 1:
            plt.ylabel("")
            plt.yticks([])
            plt.xlabel("Distance from the centre of the beam (mm)", fontsize = font_size) 
        elif i == 2:
            plt.ylabel("")
            plt.yticks([])
        
    ax = fig.add_axes((0,1,3/figure_ratio,0.3))  

    
    inside_spacing = z[-1]-z[0]
    outside_spacing = 200
    endpoints = np.array([z[0]-outside_spacing,z[-1]+outside_spacing])
    y_spacing = 0.001
    smooth_z = np.linspace(endpoints[0],endpoints[-1],50)
    smooth_waist_fit = Gaussian_radius(smooth_z,waist_fit)
    ax.plot(smooth_z,smooth_waist_fit, color = 'gray')
    for i in range(3):
        ax.plot(z[i],w_array[i], marker='o',color=colours[i],linestyle='')
    plt.xlim([endpoints[0],endpoints[-1]])
    plt.ylim([w_array[0]-y_spacing,w_array[-1]+y_spacing])
    ax.minorticks_on()
    plt.tick_params(axis='both', which='major', width= 1.3, length= 5)
    ax.xaxis.tick_top()
    plt.xlabel("Propagation distance from the laser (mm)", fontsize = font_size,labelpad=12) 
    plt.ylabel("Beam radius (mm)", fontsize = font_size,labelpad=14) 
    ax.xaxis.set_label_position('top') 
    plt.xticks(fontsize = font_size) 
    plt.yticks(fontsize = font_size) 
    return A_fit, A_err, waist_fit, waist_err

#Initial parameter guesses:
A_guess = 700
waist_guess = 1.5

perform_2D_gaussfit(pos,power,power_err,pos_err, A_guess, waist_guess)
