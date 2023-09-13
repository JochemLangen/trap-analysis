# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 17:45:14 2023

@author: joche
"""
from funcs_photos import *
import pandas as pd

plt.rcParams['figure.dpi'] = 140

# path = "New_col\\Ring_axicon\\Hollow_beam_60415us.pgm"
# path = "New_col\\Bessel_beam\\Bessel_beam_04000mm_60415us.pgm"
# path = "New_col\\Bessel_beam\\Bessel_beam_25000mm_1037us_4.pgm"
# path = "New_col\\Ring_axicon_lens\\Hollow_beam_39577us.pgm"
# # path = "New_col\\Ring_axicon_lens\\Hollow_beam_60415us_2.pgm"
# # path = "New_col\\Ring_lens_axicon\\Hollow_beam_60415us.pgm"
# # path = "New_col\\Ring_lens_axicon\\Hollow_beam_39577us.pgm"
# # path = "New_col\\Ring_lens_axicon\\axicon_2\\Hollow_beam_84370us.pgm"
# # path = "New_col\\Ring_lens_axicon\\axicon_3\\Hollow_beam_84370us.pgm"
# # path = "New_col\\Ring_lens_axicon\\Hollow_beam_84370us.pgm"
# # path = "New_col\\Flipped_ring_axicon\\Hollow_beam_84370_2.pgm"
# path = "New_col\\Flip_focus_ring_axicon\\Hollow_beam_10413us_7000mm.pgm"
path = "New_col\\Flip_focus_ring_axicon\\Collimated\\moved_camera\\Hollow_beam_4166us_16000mm.pgm"
path = "New_col\\Flipped_ring_axicon\\Dist_range\\Hollow_beam_149995us_1000mm.pgm"
path = "New_col\\Flipped_ring_axicon\\Dist_range\\Hollow_beam_149995us_2000mm.pgm"
path = "New_col\\Flipped_ring_axicon\\Dist_range\\Hollow_beam_149995us_3000mm.pgm"
path = "New_col\\Flipped_ring_axicon\\Dist_range\\Hollow_beam_149995us_4000mm.pgm"
path = "New_col\\Flipped_ring_axicon\\Dist_range\\Hollow_beam_149995us_6000mm.pgm"
path = "New_col\\Flipped_ring_axicon\\Dist_range\\Hollow_beam_149995us_8000mm.pgm"
path = "New_col\\Flipped_ring_axicon\\Dist_range\\Hollow_beam_149995us_10000mm.pgm"
path = "New_col\\Flipped_ring_axicon\\Dist_range\\Hollow_beam_149995us_12000mm.pgm"
path = "New_col\\Flipped_ring_axicon\\Dist_range\\Hollow_beam_149995us_14000mm.pgm"
path = "New_col\\Flipped_ring_axicon\\Dist_range\\Hollow_beam_149995us_16000mm.pgm"
path = "New_col\\Flipped_ring_axicon\\Img_dist_range\\Hollow_beam_130206us_0000mm.pgm"
path = "New_col\\Flipped_ring_axicon\\Img_dist_range\\Hollow_beam_130206us_2000mm.pgm"
path = "New_col\\Flipped_ring_axicon\\Img_dist_range\\Hollow_beam_130206us_4000mm.pgm"
path = "New_col\\Flipped_ring_axicon\\Img_dist_range\\Hollow_beam_130206us_6000mm.pgm"
path = "New_col\\Flipped_ring_axicon\\Img_dist_range\\Hollow_beam_130206us_8000mm.pgm"
path = "New_col\\Flipped_ring_axicon\\Img_dist_range\\Hollow_beam_130206us_10000mm.pgm"
path = "New_col\\Flipped_ring_axicon\\Img_dist_range\\Hollow_beam_130206us_12000mm.pgm"
path = "New_col\\Flipped_ring_axicon\\Img_dist_range\\Hollow_beam_130206us_14000mm.pgm"
path = "New_col\\Flipped_ring_axicon\\Img_dist_range\\Hollow_beam_130206us_16000mm.pgm"
path = "New_col\\Flipped_ring_axicon\\Img_dist_range\\Hollow_beam_130206us_18000mm.pgm"
path = "New_col\\Flipped_ring_axicon\\Img_dist_range\\Hollow_beam_130206us_20000mm.pgm"
path="New_col\\Flip_focus_ring_axicon\\New_set\\Hollow_beam_999999us_bg.pgm"
path="New_col\\Flip_focus_ring_axicon\\New_set\\Hollow_beam_499996us.pgm"
path="New_col\\Flip_focus_ring_axicon\\Collimated\\New_set\\Hollow_beam_24998us_8000mm.pgm"

# path="New_col\\Flip_focus_ring_axicon\\Collimated\\New_set\\Hollow_beam_24998us_10000mm.pgm"
im_name = path[path.rfind("\\")+1:]
print(im_name)
im = Image.open(path)
pixels = np.array(im).T
non_sat_indices = pixels < 2**16-2**4
no_sat_pix = np.size(non_sat_indices)-np.count_nonzero(non_sat_indices)
no_sat_pix_perc = np.round(no_sat_pix/np.size(non_sat_indices)*100,3)
print("There are {} saturated pixels ({}%). They are not included in the fit.".format(no_sat_pix,no_sat_pix_perc))


im_shape = np.shape(pixels)
print(np.amax(pixels))
max_intensity = np.amax(pixels) # The reference parameters are w.r.t. pixels normalised to the max profile intensity
# max_intensity = 1
pixels = pixels/max_intensity
pixel_err = 2**6 / max_intensity #The description of the camera states 10-bits, the remaining 2-bits are considered the error
pixel_err=0.5 / max_intensity #For 8-bit images
pixel_size = np.average([6.14/1280,4.9/1024]) #In mm
print(im_shape)
fontsize = 13

#Plot the original image:
xticks = np.array([0,1.2,2.4,3.6,4.8,6])/pixel_size
yticks = np.array([0,0.9,1.8,2.7,3.6,4.5])/pixel_size
fontsize = 13
plt.figure()
orig_pixels = pixels#*max_intensity/(2**16-2**4)
pixel_im = plt.imshow(orig_pixels.T,interpolation=None,cmap='plasma',aspect='auto',vmax=1,vmin=0)
plt.colorbar(mappable=pixel_im,label='Normalised intensity')
plt.xticks(xticks,labels=xticks*pixel_size,fontsize=fontsize)
plt.xlabel("Horizontal image axis (mm)",fontsize=fontsize)
plt.ylabel("Vertical image axis (mm)",fontsize=fontsize)
plt.yticks(yticks,labels=yticks*pixel_size,fontsize=fontsize)
# plt.title('Axicon distance: 18.0 +/- 1.0 mm')
# plt.title('Axicon distance: 33.0 +/- 1.0 mm')
# plt.title('Axicon distance: 43.5 +/- 1.0 mm')
# plt.savefig(path[:-4]+'_orig'+'.svg',dpi=300,bbox_inches='tight')
plt.show()


#Define Cartesian Coordinates: We take them as the centre of each pixel
#These parameters are for a guess of the top right corner (prev. used 400 & 0 for x,y offsets 0,0 for corner coords)
corner_x = 0
corner_y = 0
corner_coords = [im_shape[0]/2+0.5+corner_x,im_shape[1]/2+0.5+corner_y] #Acting guess of centre to find the outer ring (you don't want it to be exactly on a pixel)
R_guess = 200
x_offset_guess = 0   #Offset from corner_coords - make these positive
y_offset_guess = 0      #Offset from corner_coords - make these positive
print(corner_coords[0]+x_offset_guess,corner_coords[1]+y_offset_guess)
theta_ind = np.array([[-1,512+corner_y],[-1,513+corner_y]])
# theta_ind = np.array([[0,-1],[-1,0]])
ring_type = "inner"
save_fig = False


if "bottom" in im_name:
    corner_coords[1] = im_shape[1]-0.5
    y_offset_guess *= -1
    if "right" in im_name:
        theta_ind[[0,1]] = theta_ind[[1,0]]
if "left" in im_name:
    corner_coords[0] = im_shape[0]-0.5
    x_offset_guess *= -1
    if "top" in im_name:
        theta_ind[[0,1]] = theta_ind[[1,0]]
if "centre" in im_name:
    corner_coords = [im_shape[0]/2+0.5,im_shape[1]/2+0.5]
    x_offset_guess = 0
    y_offset_guess = 0
    theta_ind = np.array([[0,-1],[0,-2]])

x = np.arange(im_shape[0])-corner_coords[0]
y = np.arange(im_shape[1])-corner_coords[1]
cart_coords = np.dstack([np.dstack([x]*im_shape[1])[0],np.vstack([y]*im_shape[0])])

#Convert to Polar coordinate system:
r, theta, del_theta = CartPolar2(cart_coords)

plt.figure()
theta_im = plt.imshow(theta.T,interpolation=None,cmap='plasma',aspect='auto')
plt.colorbar(mappable=theta_im,label='radians')
plt.show()
# print(r[np.argmax(pixels)],theta[np.argmax(pixels)])
# print('hello')
#Setting the parameters for the outer ring determination
no_theta_points = 100 #The number of angles in each fitting range
averaging_int_size = 10 #The number of points over which it is averaged to get a smooth curve
darkness_limit = 0.2 #The minimum value to be considered as the outer peak
peak_size = averaging_int_size #The +/- area around the first value above this limit in which the max value is taken as the peak

theta_array = np.linspace(theta[theta_ind[0,0],theta_ind[0,1]],theta[theta_ind[1,0],theta_ind[1,1]],no_theta_points)

R, R_err, x_offset, x_offset_err, y_offset, y_offset_err = polar_find_centre(pixels, theta_array, r, theta, del_theta, cart_coords,corner_coords,averaging_int_size,darkness_limit,peak_size,R_guess,x_offset_guess,y_offset_guess,ring_type,plot=True,subplot=False,fontsize=fontsize,path=path,save_figs=save_fig)

#Print the results:
print("The outer radius is {} +/- {} mm".format(R*pixel_size,R_err*pixel_size))
print("The outer radius is {} +/- {} pixels".format(R,R_err))
print("The x offset is {} +/- {} pixels".format(x_offset,x_offset_err))
print("The y offset is {} +/- {} pixels".format(y_offset,y_offset_err))


#Write results
dataframe = pd.DataFrame(np.array([[R, R_err, x_offset, x_offset_err, y_offset, y_offset_err]]),
                         columns=['Radius (pixels)','Radius error (pixels)','X offset (pixels)','X offset error (pixels)','Y offset (pixels)','Y offset error (pixels)'],
                         dtype=float)
dataframe.to_csv(path[:-4]+'.csv')
