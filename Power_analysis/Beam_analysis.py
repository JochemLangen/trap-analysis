# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 13:02:37 2022

@author: joche
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

Excel_file = pd.read_excel("Beam-profile_measurement.xlsx")
z = np.array([13,463,1065])
z_err = np.array([0.5,0.5,np.sqrt(2)])

#The arrays corresponding to the first point along the z-axis:
pos_1 = np.array(Excel_file["Unnamed: 7"][1:59], dtype=float)
pos_err_1 = np.array(Excel_file["Unnamed: 8"][1:59], dtype=float)
power_d_1 = np.array(Excel_file["Unnamed: 9"][1:59], dtype=float)
power_d_err_1 = np.array(Excel_file["Unnamed: 10"][1:59], dtype=float)
#The arrays corresponding to the second point along the z-axis:
pos_2 = np.array(Excel_file["Unnamed: 17"][1:67], dtype=float)
pos_err_2 = np.array(Excel_file["Unnamed: 18"][1:67], dtype=float)
power_d_2 = np.array(Excel_file["Unnamed: 19"][1:67], dtype=float)
power_d_err_2 = np.array(Excel_file["Unnamed: 20"][1:67], dtype=float)

#The arrays corresponding to the third point along the z-axis:
pos_3 = np.array(Excel_file["Unnamed: 27"][1:78], dtype=float)
pos_err_3 = np.array(Excel_file["Unnamed: 28"][1:78], dtype=float)
power_d_3 = np.array(Excel_file["Unnamed: 29"][1:78], dtype=float)
power_d_err_3 = np.array(Excel_file["Unnamed: 30"][1:78], dtype=float)

def Gaussian_density(x,A,W):
    return A/W*np.exp(-2*(x/W)**2)

def Gaussian_radius(z,waist):
    z_R = (np.pi*waist**2)/(0.5319e-3)
    return waist*np.sqrt(1+(z/z_R)**2)

def chi_sqrd(y_fit, y, y_err):
    return np.sum(((y-y_fit)/y_err)**2)

def perform_gaussfit(x,y,y_err,x_err, A_guess, W_guess):
    popt, pcov = curve_fit(Gaussian_density,x,y,p0=[A_guess,W_guess],sigma=y_err,absolute_sigma=True)
    A_fit, W_fit = popt
    A_err = pcov[0,0]
    W_err = pcov[1,1]
    fitted_y = Gaussian_density(x,A_fit,W_fit)
    chi = chi_sqrd(fitted_y,y,y_err)
    chi_red = chi / (len(x)-2)
    print("The reduced chi squared value is: {}".format(chi_red))
    print("A = {} +/- {} W/mm".format(A_fit,A_err))
    print("W = {} +/- {} mm".format(W_fit,W_err))
    
    #Plotting:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(x,y,xerr=x_err,yerr=y_err, fmt='xk',capsize=2)
    smooth_x = np.linspace(x[0],x[-1],100)
    smooth_y_fit = Gaussian_density(smooth_x,A_fit,W_fit)
    smooth_y_thin_low = Gaussian_density(smooth_x,A_fit-A_err,W_fit-W_err)
    smooth_y_thin_high = Gaussian_density(smooth_x,A_fit+A_err,W_fit-W_err)
    smooth_y_wide_low = Gaussian_density(smooth_x,A_fit-A_err,W_fit+W_err)
    smooth_y_wide_high = Gaussian_density(smooth_x,A_fit+A_err,W_fit+W_err)
    fitted_array = np.dstack((smooth_y_thin_low,smooth_y_thin_high,smooth_y_wide_low,smooth_y_wide_high))[0]
    smooth_y_high = np.max(fitted_array, axis=1)
    smooth_y_low = np.min(fitted_array, axis=1)
    ax.plot(smooth_x,smooth_y_fit, color = 'red')
    dotted_colour = (0.5,0.1,0.1)
    ax.plot(smooth_x,smooth_y_high, linestyle=':', color=dotted_colour)
    ax.plot(smooth_x,smooth_y_low, linestyle=':', color=dotted_colour)
    ax.fill_between(smooth_x,smooth_y_low,smooth_y_high, color='red',alpha=0.2)
    plt.xlim([-4,4])
    plt.ylim([-10,500])
    
    return A_fit, A_err, W_fit, W_err

#Initial parameter guesses:
A_guess_1 = 700
W_guess_1 = 1.5
A_fit = np.empty((3))
A_err = np.empty_like(A_fit)
W_fit = np.empty_like(A_fit)
W_err = np.empty_like(A_fit)

A_fit[0], A_err[0], W_fit[0], W_err[0] = perform_gaussfit(pos_1,power_d_1,power_d_err_1,pos_err_1, A_guess_1, W_guess_1)
A_fit[1], A_err[1], W_fit[1], W_err[1] = perform_gaussfit(pos_2,power_d_2,power_d_err_2,pos_err_2, A_guess_1, W_guess_1)
A_fit[2], A_err[2], W_fit[2], W_err[2] = perform_gaussfit(pos_3,power_d_3,power_d_err_3,pos_err_3, A_guess_1, W_guess_1)


popt, pcov = curve_fit(Gaussian_radius,z,W_fit,p0=[W_guess_1],sigma=W_err,absolute_sigma=True)
waist_fit = popt[0]
waist_err = pcov[0,0]
fitted_W = Gaussian_radius(z,waist_fit)
chi = chi_sqrd(fitted_W,W_fit,W_err)
chi_red = chi / (len(z)-2)
print("The reduced chi squared value is: {}".format(chi_red))
print("W_0 = {} +/- {} mm".format(waist_fit,waist_err))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.errorbar(z,W_fit,xerr=z_err,yerr=W_err, fmt='xk',capsize=2)
smooth_z = np.linspace(-100,2000,10)
smooth_waist_fit = Gaussian_radius(smooth_z,waist_fit)
ax.plot(smooth_z,smooth_waist_fit, color = 'red')
plt.xlim([-100,2000])
