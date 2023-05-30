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

r_values = np.linspace(1,3,20)
m = 10
a = 1
b = 1

I_values = power_law(r_values,m,a,b)

m_guess = 2
a_guess =1
b_guess = 0.5

M = 1 #Magnification
R = 1 #Normalisation
del_R = 0.0 #Ring size increase

I_values /= M**2
r_values /= M*R
r_values += del_R

# popt, pcov = curve_fit(power_law,r_values,I_values,absolute_sigma=False,p0=[m_guess,a_guess,b_guess],bounds=(0,np.inf))
# m_fit, a_fit, b_fit = popt
# m_fit_err, a_fit_err, b_fit_err = np.sqrt(np.diagonal(pcov))
# print(m_fit,a_fit,b_fit)
# print(a*M**(m-2), b*M**(-2))

# plt.figure()

# plt.plot(r_values,power_law(r_values,m_fit,a_fit,b_fit))
# plt.scatter(r_values,I_values)
# plt.show()


increase = 0.1
r_increases = np.arange(0,2.2,increase)
r_limit =2
orig_r_limit = r_limit
r_min = 1
len_r = len(r_increases)
m_array = np.empty(len_r)
a_array = np.empty(len_r)
b_array = np.empty(len_r)
m_err_array = np.empty(len_r)
a_err_array = np.empty(len_r)
b_err_array = np.empty(len_r)


r_values = np.arange(r_min,r_limit,0.02)
I_values = power_law(r_values,m,a,b)
r_values_len = len(r_values)
# factor = 0.0001
# factor_array = np.full(len_r,1.)

for i in range(len_r):
    
    
    popt, pcov = curve_fit(power_law,r_values,I_values,absolute_sigma=False,p0=[m_guess,a_guess,b_guess],bounds=(0,np.inf))
    #Absolute sigma has been set to false as this provided a larger and thus more realistic error
    m_array[i], a_array[i], b_array[i] = popt
    m_err_array[i], a_err_array[i], b_err_array[i] = np.sqrt(np.diagonal(pcov))
    
    plt.figure()
    x_val = np.linspace(r_min,r_limit,100)
    
    plt.scatter(r_values,I_values)
    plt.plot(x_val,power_law(x_val,m_array[i], a_array[i], b_array[i]),color='orange')
    plt.xlim([r_min,r_increases[-1]+orig_r_limit])
    plt.show()
    # r_values += increase
    r_limit += increase
    
    r_values = np.arange(r_min,r_limit,0.02)
    I_values = np.insert(I_values,0,np.full(len(r_values)-r_values_len,b))
    r_values_len = len(r_values)
    
    
def linear(x,m,a):
    return m*x + a

popt, pcov = curve_fit(linear,r_increases/orig_r_limit,m_array/m,absolute_sigma=False,p0=[1,m],bounds=(0,np.inf))
#Absolute sigma has been set to false as this provided a larger and thus more realistic error
lin_val, const_val = popt
print(lin_val,const_val)
print(lin_val/m)
plt.figure()
plt.scatter(r_increases/orig_r_limit,m_array/m)
plt.plot(r_increases/orig_r_limit,linear(r_increases/orig_r_limit,lin_val,const_val))
plt.show()
print(np.amax(m_array),np.amax(r_increases))

plt.figure()
plt.scatter(r_increases,a_array)
plt.show()
print(m_array)
plt.figure()
# plt.plot(r_increases,b-a_array*(r_increases-2)**m_array)
plt.scatter(r_increases,b_array)
print(b_array)
plt.show()


