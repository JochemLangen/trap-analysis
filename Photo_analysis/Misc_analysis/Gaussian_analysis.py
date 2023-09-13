# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 20:40:22 2023

@author: joche
"""

from funcs_photos import *
import pandas as pd

# path = "Feb\\Gaussian_beam_2\\Gaussian_beam_02080us_6.pgm"
# path = "Third_beam\\Gaussian\\Gaussian_beam_3.bmp"
# path = "Feb\\Gaussian_beam\\Gaussian_beam_01037us.bmp"
# path = "Feb\\Gaussian_beam_2\\Mirror_Gaussian_beam_11456us_6_2.pgm"
# path = "Feb\\Gaussian_beam_3\\Gaussian_beam_04166us_4.pgm"
# path = "Feb\\Gaussian_beam_3\\Gaussian_beam_no_lens2_46873us.pgm"
# path = "Feb\\Gaussian_beam_3\\Gaussian_beam_no_lenses_01037us_2.pgm"
# path = "Feb\\Gaussian_beam_3\\Gaussian_beam_fibre_tip_101036us.pgm"
# path = "Feb\\Gaussian_beam_4\\Gaussian_beam_126040us_recentre.pgm"
path = "D:\\Jochem\\Documents\\Uni\\Physics\\Level 4 Project\\New_col\\Gaussian_beam\\Gaussian_beam_003123us_2.pgm"
# path = "New_col\\Ring_axicon_lens\\Hollow_beam_39577us.pgm"
# path = "New_col\\Ring_lens_axicon\\axicon_3\\Hollow_beam_39577us.pgm"
# path = "New_col\\Ring_axicon\\2_degrees\\Hollow_beam_60415us.pgm"
# path = "New_col\\Bessel_beam\\Bessel_beam_00000mm_999995us_3.pgm"
# path = "New_col\\Bessel_beam\\Bessel_beam_25000mm_999995us_2.pgm"
# path = "New_col\\Bessel_beam\\Bessel_beam_00000mm_60415us.pgm"
im = Image.open(path)
pixels = np.array(im).T
im_shape = np.shape(pixels)
max_intensity = 2**16-2**4
print(max_intensity)
# max_intensity= 2**8-1
print(max_intensity)
print(np.amax(np.amax(pixels)))
pixels = pixels/max_intensity
pixel_err = 2**6/max_intensity
r_radial_err = 0.5 #Pixel width
pixel_size = np.average([6.14/1280,4.9/1024]) #In mm
pi = np.pi

#Generate the cart. coord. system:
x = np.arange(im_shape[0])
y = np.arange(im_shape[1])
cart_coords = np.dstack([np.dstack([x]*im_shape[1])[0],np.vstack([y]*im_shape[0])])

#Remove the saturated pixels:
non_sat_indices = pixels < 1
pixelsa = pixels[non_sat_indices]
cart_coordsa = cart_coords[non_sat_indices]
no_sat_pix = np.size(non_sat_indices)-np.count_nonzero(non_sat_indices)
no_sat_pix_perc = np.round(no_sat_pix/np.size(non_sat_indices)*100,3)
print("There are {} saturated pixels ({}%). They are not included in the fit.".format(no_sat_pix,no_sat_pix_perc))

def TwoD_gaussian(xy,A,B,C,D,E):
    return A*np.exp(-((xy[:,:,0]-B)**2+(xy[:,:,1]-C)**2)/D)+E

def TwoD_gaussian_lin(xy,A,B,C,D,E):
    return A*np.exp(-((xy[:,0]-B)**2+(xy[:,1]-C)**2)/D)+E
    
def radial_gaussian(r,A,D,E):
    return A*np.exp(-(r**2)/D)+E

#Generate the initial guess
A_guess = 0.6
B_guess = im_shape[0]/2
C_guess = im_shape[1]/2
waist_guess = 500
D_guess = (waist_guess**2)/2
E_guess = 0
guess_I = TwoD_gaussian(cart_coords,A_guess,B_guess,C_guess,D_guess,E_guess)


xticks = np.array([0,1.2,2.4,3.6,4.8,6])/pixel_size
yticks = np.array([0,0.9,1.8,2.7,3.6,4.5])/pixel_size
fontsize = 13

#Plot the original image:
plt.figure()
pixel_im = plt.imshow(pixels.T,interpolation=None,cmap='plasma',aspect='auto',vmin=0,vmax=1)
plt.colorbar(mappable=pixel_im,label='Normalised intensity')
plt.xticks(xticks,labels=xticks*pixel_size,fontsize=fontsize)
plt.xlabel("Horizontal image axis (mm)",fontsize=fontsize)
plt.ylabel("Vertical image axis (mm)",fontsize=fontsize)
plt.yticks(yticks,labels=yticks*pixel_size,fontsize=fontsize)
plt.title('Original image')
plt.show()

#Plot the guess:
plt.figure()
sim_im = plt.imshow(guess_I.T,interpolation=None,cmap='plasma',aspect='auto',vmin=0,vmax=1)
plt.colorbar(mappable=sim_im,label='Normalised intensity')
plt.xticks(xticks,labels=xticks*pixel_size,fontsize=fontsize)
plt.xlabel("Horizontal image axis (mm)",fontsize=fontsize)
plt.ylabel("Vertical image axis (mm)",fontsize=fontsize)
plt.yticks(yticks,labels=yticks*pixel_size,fontsize=fontsize)
plt.title('Gaussian guess')
plt.show()

#Perform the fitting:
popt, pcov = curve_fit(TwoD_gaussian_lin,cart_coordsa,pixelsa,p0=[A_guess,B_guess,C_guess,D_guess,E_guess],bounds=(0,np.inf))
A_fit, B_fit, C_fit, D_fit, E_fit = popt
A_err, B_err, C_err, D_err, E_err = np.sqrt(np.diagonal(pcov))
w_0 = np.sqrt(2*D_fit)
w_0_err = D_err/w_0 #Calculus method
print("The fitted beam waist is: {} +/- {} mm".format(w_0*pixel_size,w_0_err*pixel_size))
fit_I = TwoD_gaussian(cart_coords,A_fit, B_fit, C_fit, D_fit, E_fit)
print(popt)

#Plot the fit:
plt.figure()
fit_im = plt.imshow(fit_I.T,interpolation=None,cmap='plasma',aspect='auto',vmin=0,vmax=1)
plt.colorbar(mappable=fit_im,label='Normalised intensity')
# plt.scatter(B_fit,C_fit,marker='x',color='black')
plt.xticks(xticks,labels=xticks*pixel_size,fontsize=fontsize)
plt.xlabel("Horizontal image axis (mm)",fontsize=fontsize)
plt.ylabel("Vertical image axis (mm)",fontsize=fontsize)
plt.yticks(yticks,labels=yticks*pixel_size,fontsize=fontsize)
plt.title('Fitted Gaussian')
plt.show()

#Calc. and plot the residuals:
resid_I = pixels-fit_I
resid_I[np.logical_not(non_sat_indices)] = 0 #The saturated pixel residuals are set to 0

plt.figure()
resid_im = plt.imshow(resid_I.T,interpolation=None,cmap='plasma',aspect='auto')
plt.colorbar(mappable=resid_im,label='Residuals on normalised intensity')
plt.xticks(xticks,labels=xticks*pixel_size,fontsize=fontsize)
plt.xlabel("Horizontal image axis (mm)",fontsize=fontsize)
plt.ylabel("Vertical image axis (mm)",fontsize=fontsize)
plt.yticks(yticks,labels=yticks*pixel_size,fontsize=fontsize)
plt.title('Gaussian fit residuals')
plt.show()



#Calc. and plot the fractional residuals w.r.t the pixel intensities of the orig. image
non_zero_fit = pixels > 0
frac_resid_I = np.zeros_like(resid_I)
frac_resid_I[non_zero_fit] = abs(resid_I[non_zero_fit]/pixels[non_zero_fit])

plt.figure()
fr_resid_im = plt.imshow(frac_resid_I.T,interpolation=None,cmap='plasma',aspect='auto')
plt.colorbar(mappable=fr_resid_im,label='Fractional absolute normalised intensity residuals')
plt.xticks(xticks,labels=xticks*pixel_size,fontsize=fontsize)
plt.xlabel("Horizontal image axis (mm)",fontsize=fontsize)
plt.ylabel("Vertical image axis (mm)",fontsize=fontsize)
plt.yticks(yticks,labels=yticks*pixel_size,fontsize=fontsize)
plt.title('Fractional gaussian fit residuals')
plt.show()

#Generate a polar coordinate system on the fitted gaussian
gauss_cart_coords = cart_coords.astype(float)
gauss_cart_coords[:,:,0] -= B_fit
gauss_cart_coords[:,:,1] -= C_fit
r, theta, del_theta = CartPolar2(gauss_cart_coords)
exp_decay_val = 5 #The exponent
r_decay_val = np.sqrt(exp_decay_val*D_fit)
min_length = min([im_shape[0]-B_fit,B_fit,im_shape[1],C_fit])
if r_decay_val > min_length:
    r_decay_val = min_length
    exp_decay_val = (r_decay_val**2)/D_fit
    print("The chosen range to investigate was too long and has been capped at: {} mm".format(r_decay_val*pixel_size))
print("The averages are taken over the region within r = {} mm where the intensity is {}% of the peak".format(r_decay_val*pixel_size,np.round(np.exp(-exp_decay_val)*100,3)))

no_theta_points = 100
del_theta = pi/(no_theta_points) # (2pi / (# of points)) / 2
theta_array = np.linspace(del_theta,2*pi-del_theta,no_theta_points)
# print(theta_array[1:]-theta_array[:-1])
# print(del_theta*2)
# print(theta_array)
avg_pixel_values = np.empty_like(theta_array)
avg_pixel_errors = np.empty_like(theta_array)
avg_fit_value = np.empty_like(theta_array)
avg_fit_err = np.empty_like(theta_array)

for i in range(no_theta_points):
    radials_bool = np.logical_and(np.logical_and(theta >= theta_array[i]-del_theta,
                                              theta < theta_array[i]+del_theta), r <= r_decay_val)
    
    pixel_radials = pixels[radials_bool]
    avg_pixel_values[i] = np.mean(pixel_radials)
    avg_pixel_errors[i] = np.std(pixel_radials)/np.sqrt(np.size(pixel_radials))
    #Calculate the reference value
    fit_radials = fit_I[radials_bool]
    avg_fit_value[i] = np.mean(fit_radials)
    avg_fit_err[i] = np.std(fit_radials)/np.sqrt(np.size(fit_radials))


# fig = plt.figure()
# ax = fig.add_axes((0,0,1,1))
# plt.errorbar(theta_array,avg_pixel_values,yerr=avg_pixel_errors,marker='x',color='black',capsize=2,linestyle='')
# xlim = ax.get_xlim()
# plt.errorbar(theta_array,avg_fit_value,yerr=avg_fit_err,color='red',capsize=2,linestyle='--',zorder=0)

# # plt.plot([xlim[0],xlim[1]],[avg_fit_value,avg_fit_value],linestyle='--',color='red',zorder=0)
# plt.xlim(xlim)
# plt.show()

# reduced_cart_indices = np.logical_and(cart_coords[:,:,0] <= B_fit - r_decay_val,cart_coords[:,:,0] >= B_fit + r_decay_val,
#                                       cart_coords[:,:,1] <= C_fit - r_decay_val,cart_coords[:,:,1] >= C_fit + r_decay_val)
# reduced_cart_coord = cart_coords[]


#Normalised for model
norm_pix_avg = avg_pixel_values/avg_fit_value
norm_pix_avg_err = norm_pix_avg*np.sqrt((avg_fit_err/avg_fit_value)**2+(avg_pixel_errors/avg_pixel_values)**2)
fig = plt.figure()
ax = fig.add_axes((0,0,1,1))
plt.errorbar(theta_array,norm_pix_avg,yerr=norm_pix_avg_err,marker='x',color='black',capsize=2,linestyle='')
plt.xticks(fontsize=fontsize)
plt.xlabel("Angle (radians)",fontsize=fontsize)
plt.ylabel("Average normalised intensity",fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.title('The average intensity normalised with the fit')
plt.show()
max_norm_pix = np.argmax(norm_pix_avg)
min_norm_pix = np.argmin(norm_pix_avg)
max_difference = norm_pix_avg[max_norm_pix] - norm_pix_avg[min_norm_pix]
max_difference_err =  np.sqrt(norm_pix_avg_err[max_norm_pix]**2+norm_pix_avg_err[min_norm_pix]**2)
print("The maximum difference between the averaged beam intensities is {} +/- {}".format(max_difference,max_difference_err))

#Plot averaged intensity profile against the Gaussian fit
ra = r[non_sat_indices]
thetaa = theta[non_sat_indices]
low_ra = ra[thetaa <= pi]
high_ra = ra[thetaa > pi]
low_pixelsa = pixelsa[thetaa <= pi]
high_pixelsa = pixelsa[thetaa > pi]

avg_int = 100
r_interval = r_decay_val / avg_int

r_avg = np.empty(avg_int*2)
pixel_avg = np.empty_like(r_avg)
r_avg_err = np.empty_like(r_avg)
pixel_avg_err = np.empty_like(r_avg)

for i in range(avg_int):
    r_int_indices = np.logical_and(low_ra >= i*r_interval, low_ra < (i+1)*r_interval)
    r_avg_range = low_ra[r_int_indices]
    if len(r_avg_range) == 0:
        r_avg[i] = np.nan
        pixel_avg[i] = np.nan
        r_avg_err[i] = np.nan
        pixel_avg_err[i] = np.nan
    else:
        r_avg[i] = np.mean(r_avg_range)
        pixel_avg_range = low_pixelsa[r_int_indices]
        pixel_avg[i] = np.mean(pixel_avg_range)
        
        r_avg_err[i] = np.std(r_avg_range)/np.sqrt(np.size(r_avg_range))
        pixel_avg_err[i] = np.std(pixel_avg_range)/np.sqrt(np.size(pixel_avg_range))
    
    r_int_indices = np.logical_and(high_ra >= i*r_interval, high_ra < (i+1)*r_interval)
    r_avg_range = high_ra[r_int_indices]
    if len(r_avg_range) == 0:
        r_avg[avg_int+i] = np.nan
        pixel_avg[avg_int+i] = np.nan
        r_avg_err[avg_int+i] = np.nan
        pixel_avg_err[avg_int+i] = np.nan
    else:
        r_avg[avg_int+i] = -np.mean(r_avg_range)
        pixel_avg_range = high_pixelsa[r_int_indices]
        pixel_avg[avg_int+i] = np.mean(pixel_avg_range)
        
        r_avg_err[avg_int+i] = np.std(r_avg_range)/np.sqrt(np.size(r_avg_range))
        pixel_avg_err[avg_int+i] = np.std(pixel_avg_range)/np.sqrt(np.size(pixel_avg_range))


fig = plt.figure()
ax = fig.add_axes((0,0,1,1))
plt.errorbar(r_avg,pixel_avg,yerr=pixel_avg_err,xerr=r_radial_err,marker='x',color='black',capsize=2,linestyle='')
xlim = ax.get_xlim()
x_values = np.linspace(xlim[0],xlim[1],100)
y_values = radial_gaussian(x_values,A_fit,D_fit,E_fit)
plt.plot(x_values,y_values,zorder=0)
plt.xlim(xlim)
xticks = np.array([-1,-0.5,0,0.5,1])/pixel_size
plt.xticks(xticks,labels=xticks*pixel_size,fontsize=fontsize)
plt.xlabel("Radius (mm)",fontsize=fontsize)
plt.ylabel("Normalised intensity",fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.title('The Gaussian fit against the model averaged over angles')

plt.show()



plt.figure()
theta_im = plt.imshow(theta.T,interpolation=None,cmap='plasma',aspect='auto')
plt.colorbar(mappable=theta_im,label='Radians')
plt.show()

