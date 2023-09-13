# -*- coding: utf-8 -*-
"""
Created on Tue May 23 11:59:25 2023

@author: joche
"""

from funcs_photos import *
import pandas as pd
from scipy.fft import rfft2, irfft2, rfft,fft2,ifft2,fft,fftn,fftshift,fftfreq,ifftshift
tickwidth = 2
ticklength = 4
rootdir = "D:\\Jochem\\Documents\\Uni\\Physics\\Level 4 Project\\New_col\\Gaussian_beam"
path = "D:\\Jochem\\Documents\\Uni\\Physics\\Level 4 Project\\New_col\\Gaussian_beam\\Gaussian_beam_003123us_2.pgm"

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
no_sat_pix = np.size(non_sat_indices)-np.count_nonzero(non_sat_indices)
no_sat_pix_perc = np.round(no_sat_pix/np.size(non_sat_indices)*100,3)
print("There are {} saturated pixels ({}%). Note, these are included in the fit!".format(no_sat_pix,no_sat_pix_perc))


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
fontsize = 11

#Plot the original image:
    # fig = plt.figure()
    # ax1 = fig.add_axes((0,0,1,1))
    # pixel_im = plt.imshow(pixels.T,interpolation=None,cmap='plasma',aspect='auto',origin='lower',vmin=0,vmax=1)
    # cbar = plt.colorbar(mappable=pixel_im)
    # cbar.ax.tick_params(labelsize=fontsize, width= tickwidth, length= ticklength)
    # # ticklabs = cbar.ax.get_yticklabels()
    # # cbar.ax.set_yticklabels(ticklabs,   fontsize=fontsize)
    # # plt.colorbar(mappable=pixel_im,label=r'Normalised intensity, $I\;/\;I_{max}$"')
    # plt.xticks(xticks,labels=xticks*pixel_size,fontsize=fontsize)
    # plt.xlabel("x (mm)",fontsize=fontsize)
    # plt.ylabel("y (mm)",fontsize=fontsize)
    # plt.gca().set_aspect('equal')
    # plt.yticks(yticks,labels=yticks*pixel_size,fontsize=fontsize)
    # # ax1.tick_params(width=4,length=3)
    # plt.tick_params(axis='both', which='major', width= tickwidth, length= ticklength)
    # plt.text(0.05, 0.9, '(b)',transform=ax1.transAxes,fontsize=fontsize,color='white')
fig =plt.figure()
fig.add_axes((0,0,0.4,0.4))
pixel_im = plt.imshow(pixels.T,interpolation=None,cmap='plasma',aspect='auto',vmin=0,vmax=1)
cbar = plt.colorbar(mappable=pixel_im,label='Normalised intensity')
cbar.ax.tick_params(labelsize=fontsize, width= tickwidth, length= ticklength)
# ticklabs = cbar.ax.get_yticklabels()
# cbar.ax.set_yticklabels(ticklabs, fontsize=fontsize)
plt.xticks(xticks,labels=xticks*pixel_size,fontsize=fontsize)
# pixel_im.figure.axes[0].tick_params(axis="both", labelsize=21)
plt.xlabel("Horizontal image axis (mm)",fontsize=fontsize)
plt.ylabel("Vertical image axis (mm)",fontsize=fontsize)
plt.yticks(yticks,labels=yticks*pixel_size,fontsize=fontsize)
plt.tick_params(axis='both', which='major', width= tickwidth, length= ticklength)
plt.savefig(rootdir+"\\gauss_og_img.svg",dpi=300,bbox_inches='tight')
# plt.title('Original image')
plt.show()
pixels_old = pixels.copy()


freq = rfft2(pixels)

f_shape = np.shape(freq)
xcentre = int(f_shape[0]/2)

fshift = freq.copy()
fshift[:xcentre] = freq[xcentre:]
fshift[xcentre:] = freq[:xcentre]

abs_fshift_old = np.log(abs(fshift))
cut_off_freq_x = 20
cut_off_freq_y = 20

fshift[:xcentre-cut_off_freq_x] = 0
fshift[xcentre+cut_off_freq_x:] = 0
fshift[:,cut_off_freq_y:] = 0

plt.figure()
pixel_im = plt.imshow(abs_fshift_old.T,interpolation=None,cmap='plasma',aspect='auto')
plt.plot([xcentre-cut_off_freq_x,xcentre-cut_off_freq_x,xcentre+cut_off_freq_x,xcentre+cut_off_freq_x],[0,cut_off_freq_y,cut_off_freq_y,0],color='black')
plt.colorbar(mappable=pixel_im,label='Amplitude')
freq_x_ticks = np.linspace(0,1280,5,dtype=int)
freq_x_labels = freq_x_ticks - int(1280/2)
plt.xticks(freq_x_ticks,labels=freq_x_labels,fontsize=fontsize)
freq_y_ticks = np.linspace(0,512,5)
plt.yticks(freq_y_ticks,fontsize=fontsize)
plt.xlabel("Horizontal spatial frequency (1/pixels)",fontsize=fontsize)
plt.ylabel("Vertical spatial frequency (1/pixels)",fontsize=fontsize)
plt.title('Fourier spectrum')
plt.show()

freq = fshift.copy()
freq[:xcentre] = fshift[xcentre:]
freq[xcentre:] = fshift[:xcentre]

pixels = irfft2(freq)
plt.figure()
pixel_im = plt.imshow(pixels.T,interpolation=None,cmap='plasma',aspect='auto',vmin=0,vmax=1)
plt.colorbar(mappable=pixel_im,label='Normalised intensity')
plt.xticks(xticks,labels=xticks*pixel_size,fontsize=fontsize)
plt.xlabel("Horizontal image axis (mm)",fontsize=fontsize)
plt.ylabel("Vertical image axis (mm)",fontsize=fontsize)
plt.yticks(yticks,labels=yticks*pixel_size,fontsize=fontsize)
plt.title('Adjusted image')
plt.show()

fig =plt.figure()
fig.add_axes((0,0,0.4,0.4))
pixel_im = plt.imshow(pixels.T,interpolation=None,cmap='plasma',aspect='auto',vmin=0,vmax=1)
cbar = plt.colorbar(mappable=pixel_im,label='Normalised intensity')
cbar.ax.tick_params(labelsize=fontsize, width= tickwidth, length= ticklength)
# ticklabs = cbar.ax.get_yticklabels()
# cbar.ax.set_yticklabels(ticklabs, fontsize=fontsize)
plt.xticks(xticks,labels=xticks*pixel_size,fontsize=fontsize)
# pixel_im.figure.axes[0].tick_params(axis="both", labelsize=21)
plt.xlabel("Horizontal image axis (mm)",fontsize=fontsize)
plt.ylabel("Vertical image axis (mm)",fontsize=fontsize)
plt.yticks(yticks,labels=yticks*pixel_size,fontsize=fontsize)
plt.tick_params(axis='both', which='major', width= tickwidth, length= ticklength)
plt.savefig(rootdir+"\\gauss_adj_img.svg",dpi=300,bbox_inches='tight')
# plt.title('Original image')
plt.show()
pixels_old = pixels.copy()


plot_index = 350
plt.figure()
plt.plot(pixels[plot_index,:])
plt.plot(pixels_old[plot_index,:])
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
cart_coords_lin = np.reshape(cart_coords,(im_shape[0]*im_shape[1],2))
pixels_lin = np.reshape(pixels,(im_shape[0]*im_shape[1]))
popt, pcov = curve_fit(TwoD_gaussian_lin,cart_coords_lin,pixels_lin,p0=[A_guess,B_guess,C_guess,D_guess,E_guess],bounds=(0,np.inf))
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


mean_intensity = np.average(pixels,axis=(0,1))
# mean_intensity_err = np.std(pixels)/np.sqrt(im_shape[0]*im_shape[1])
peak_intensity = np.amax(pixels)
rel_peak_intensity = peak_intensity/mean_intensity
# rel_peak_intensity_err = np.sqrt(pixel_err**2+(rel_peak_intensity*mean_intensity_err)**2)/mean_intensity
rel_peak_intensity_err = pixel_err/mean_intensity
print("Relative peak intensity: {} +/- {}".format(rel_peak_intensity,rel_peak_intensity_err))


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
fig =plt.figure()
fig.add_axes((0,0,0.5,0.5))
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
low_pixels_lin = pixels_lin[thetaa <= pi]
high_pixels_lin = pixels_lin[thetaa > pi]

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
        pixel_avg_range = low_pixels_lin[r_int_indices]
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
        pixel_avg_range = high_pixels_lin[r_int_indices]
        pixel_avg[avg_int+i] = np.mean(pixel_avg_range)
        
        r_avg_err[avg_int+i] = np.std(r_avg_range)/np.sqrt(np.size(r_avg_range))
        pixel_avg_err[avg_int+i] = np.std(pixel_avg_range)/np.sqrt(np.size(pixel_avg_range))

fontsize=14
fig =plt.figure()
ax = fig.add_axes((0,0,0.5,0.5))
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
# plt.title('The Gaussian fit against the model averaged over angles')
plt.savefig(rootdir+"\\gauss_fit_img.svg",dpi=300,bbox_inches='tight')
plt.show()


plt.figure()
theta_im = plt.imshow(theta.T,interpolation=None,cmap='plasma',aspect='auto')
plt.colorbar(mappable=theta_im,label='Radians')
plt.show()