# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 20:14:05 2023

@author: joche
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def power_law(x,m,a,b):
    return a*(x**m)+b

r_values = np.linspace(0,1,20)
m = 15
a = 2
b = 2

I_values = power_law(r_values,m,a,b)

m_guess = 10
a_guess =1
b_guess = 0

M = 1 #Magnification
R = 1 #Normalisation
del_R = 0.1 #Ring size increase

I_values /= M**2
r_values /= M*R
r_values += del_R

increase = 0.01
r_increases = np.arange(0,1,increase)
len_r = len(r_increases)
m_array = np.empty(len_r)
a_array = np.empty(len_r)
b_array = np.empty(len_r)
m_err_array = np.empty(len_r)
a_err_array = np.empty(len_r)
b_err_array = np.empty(len_r)

# factor = 0.0001
# factor_array = np.full(len_r,1.)

for i in range(len_r):
    # r_values += r_increases[i]
    r_values += increase
    
    popt, pcov = curve_fit(power_law,r_values,I_values,absolute_sigma=False,p0=[m_guess,a_guess,b_guess],bounds=(0,np.inf))
    #Absolute sigma has been set to false as this provided a larger and thus more realistic error
    m_array[i], a_array[i], b_array[i] = popt
    m_err_array[i], a_err_array[i], b_err_array[i] = np.sqrt(np.diagonal(pcov))
    # m_fit, a_fit, b_fit = popt
    # m_fit_err, a_fit_err, b_fit_err = np.sqrt(np.diagonal(pcov))
    # print(m_fit,a_fit,b_fit)
    # print(a*)

plt.figure()
plt.scatter(r_increases,m_array)
plt.show()

plt.figure()
plt.scatter(r_increases,a_array)
plt.show()

plt.figure()
plt.scatter(r_increases,b_array)
plt.show()
    

# plt.figure()
# plt.scatter(r_values,I_values)
# plt.show()