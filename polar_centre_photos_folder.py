# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 23:38:52 2023

@author: joche
"""

import os
import sys
from funcs_photos import *
import pandas as pd

# rootdir = 'D:\\Jochem\\Documents\\Uni\\Physics\\Level 4 Project\\New_col\\Ring_axicon_lens\\New_set\\Adj_cam'



# rootdir = 'D:\\Jochem\\Documents\\Uni\\Physics\\Level 4 Project\\New_col\\Flip_focus_ring_axicon\\Collimated\\New_set\\125_mm_lens\\Adj_cam'
# rootdir = 'D:\\Jochem\\Documents\\Uni\\Physics\\Level 4 Project\\New_col\\Flip_focus_ring_axicon\\Collimated\\New_set\\200_mm_lens\\Ax_dist_range\\Adj_cam'
rootdir = 'D:\\Jochem\\Documents\\Uni\\Physics\\Level 4 Project\\New_col\\Flip_focus_ring_axicon\\Collimated\\New_set\\200_mm_lens\\Adj_cam'

# rootdir = 'D:\\Jochem\\Documents\\Uni\\Physics\\Level 4 Project\\New_col\\Flip_focus_ring_axicon\\Collimated\\New_set\\Dist_range\\Adj_cam'
# rootdir = 'D:\\Jochem\\Documents\\Uni\\Physics\\Level 4 Project\\New_col\\Flip_focus_ring_axicon\\New_set\\Dist_range\\Adj_cam'

# rootdir = 'New_col\\Flipped_ring_axicon\\Img_dist_range_2\\Adj_cam'
# rootdir = 'New_col\\Flipped_ring_axicon\\Img_dist_range\\Adj_cam'
# rootdir = 'New_col\\Flipped_ring_axicon\\Dist_range'#\\Adj_cam'
# rootdir = 'New_col\\Flipped_ring_axicon\\Dist_range_2\\Adj_cam'
 
#The below has generated backgrounds:
# rootdir = 'D:\\Jochem\\Documents\\Uni\\Physics\\Level 4 Project\\New_col\\Ring_lens_axicon\\Ax_dist_range'
# dist_float_factor = 10
# rootdir = 'D:\\Jochem\\Documents\\Uni\\Physics\\Level 4 Project\\New_col\\Flip_focus_ring_axicon\\New_set\\Iris_range'
# dist_float_factor = 100
#THe above has distances
# rootdir = 'D:\\Jochem\\Documents\\Uni\\Physics\\Level 4 Project\\New_col\\Ring_lens_axicon\\2_degrees' #not done yet -> ring is too small for meaningful comparison
# rootdir = 'D:\\Jochem\\Documents\\Uni\\Physics\\Level 4 Project\\New_col\\Ring_lens_axicon\\Obs_obj'
# rootdir = 'D:\\Jochem\\Documents\\Uni\\Physics\\Level 4 Project\\New_col\\Ring_axicon_lens\\Obs_obj'
# rootdir = 'D:\\Jochem\\Documents\\Uni\\Physics\\Level 4 Project\\New_col\\Ring_axicon_lens\\Obs_obj\\closer_obj'
# rootdir = 'D:\\Jochem\\Documents\\Uni\\Physics\\Level 4 Project\\New_col\\Ring_lens_axicon'


print(rootdir)

pixel_array = []
dist_array = []
exp_array = []
exp_float_factor = 100
exposure_err = 0.005 #In us
name_array = []

bg_pix_array = []
bg_pix_err_array = []
bg_exp_array = []
bg_dist_array = []
bg_dist_err_array = []
bg_exp_err_array = []
bg_exp_dist_array = []
single_dir = True

saturation_value = 2**16-2**4
pixel_err = 2**6

if rootdir.find('cam') == -1:
    bg_choosing_index = 0
else:
    bg_choosing_index = 1


#These parameters are for a guess of the offset from the image centre (prev. used 400 & 0 for x,y offsets 0,0 for corner coords)
corner_x = 0
corner_y = 0


# theta_ind = np.array([[0,-1],[-1,0]])
ring_type = "inner"
plots = True
subplots = False
save_fig = False
save_info = False
distances = True #Whether or not distances are used
if distances == True:
    dist_float_factor = 1000 
background_gen = False #Whether or not the background gen file has been used
pixel_size = np.average([6.14/1280,4.9/1024]) #In mm
xticks = np.array([0,1.2,2.4,3.6,4.8,6])/pixel_size
yticks = np.array([0,0.9,1.8,2.7,3.6,4.5])/pixel_size
fontsize = 13

#Setting the parameters for the outer ring determination
no_theta_points = 500 #The number of angles in each fitting range
averaging_int_size = 10 #The number of points over which it is averaged to get a smooth curve
darkness_limit = 0.05 #The minimum value to be considered as the peak
drop_limit = -0.05 #The value below which the difference between points must be to be the drop-off from the first peak
peak_size = averaging_int_size #The +/- area around the first value above this limit in which the max value is taken as the peak
R_jump = 40 #The jump in corner coordinates if the circle does not fall in the centre
safety_frac = 0.05 #The safety fraction on the used radius in combining the images



dir_index = 0
for subdir, dirs, files in os.walk(rootdir):
    if single_dir == True:
        if dir_index > 0:
            break
    dir_index += 1
    directory_name = subdir[subdir[:-1].rfind("\\")+1:]
    for file in files:
        path = os.path.join(subdir,file)
        if path[-3:] == 'pgm':
            im = Image.open(path)
            
            dist_last_index = path.rfind("mm")
            dreduced_path = path[:dist_last_index]
            exp_last_index = path.rfind("us")
            ereduced_path = path[:exp_last_index]
            if path.find('no') == -1: #Images with no in them will not be used
                if path.find('bg') == -1:
                    if distances == True:
                        dist_array += [float(dreduced_path[dreduced_path.rfind("_")+1:dist_last_index])/dist_float_factor]
                    exp_array += [float(ereduced_path[ereduced_path.rfind("_")+1:exp_last_index])/exp_float_factor]
                    name_array += [path[path.rfind("\\")+1:]]
                    
                    pixel_array += [np.array(im).T]
                elif background_gen == False:
                    if distances == True:
                        bg_dist_array += [float(dreduced_path[dreduced_path.rfind("_")+1:dist_last_index])/dist_float_factor]
                    bg_exp_array += [float(ereduced_path[ereduced_path.rfind("_")+1:exp_last_index])/exp_float_factor]
                    
                    bg_pix_array += [np.array(im).T]
        elif path[-3:] == 'csv' and background_gen == True:
            dist_last_index = path.rfind("mm")
            dreduced_path = path[:dist_last_index]
            exp_last_index = path.rfind("us")
            ereduced_path = path[:exp_last_index]
            if path.find('_bg_err') != -1:
                if distances == True:
                    bg_dist_err_array += [float(dreduced_path[dreduced_path.rfind("_")+1:dist_last_index])/dist_float_factor]
                bg_exp_err_array += [float(ereduced_path[ereduced_path.rfind("_")+1:exp_last_index])/exp_float_factor]
                
                bg_pix_err_array += [pd.read_csv(path).values[:,1:]]
            elif path.find('_bg') != -1: 
                if distances == True:
                    bg_dist_array += [float(dreduced_path[dreduced_path.rfind("_")+1:dist_last_index])/dist_float_factor]
                bg_exp_array += [float(ereduced_path[ereduced_path.rfind("_")+1:exp_last_index])/exp_float_factor]
                
                bg_pix_array += [pd.read_csv(path).values[:,1:]]
            # if path.find('background_data') != -1:
            #     background_gen = True
            #     pix_std_array, R_pix_SE_array, L_pix_SE_array, indices = pd.read_csv(path).values[:,1:]
                
                
            # pixel_im = plt.imshow(np.array(im),interpolation=None,cmap='plasma',aspect='auto',vmin=0,vmax=saturation_value)
            # plt.gca().set_aspect('equal')
            # plt.xlim([0,1279])
            # plt.ylim([0,1023])
            # plt.colorbar(mappable=pixel_im,label='Normalised intensity')
            # plt.title(file)
            # plt.show()
            

dist_array = np.asarray(dist_array)
exp_array = np.asarray(exp_array)
pixel_array = np.asarray(pixel_array,dtype=float)
name_array = np.asarray(name_array,dtype=str)

if distances == True:
    dist_len = len(dist_array)
    sorted_indices = np.argsort(dist_array)
    dist_array = dist_array[sorted_indices]
    exp_array = exp_array[sorted_indices]
    pixel_array = pixel_array[sorted_indices]
    pixel_err_array = np.full_like(pixel_array,pixel_err)
    name_array = name_array[sorted_indices]
else:
    dist_len = len(exp_array)

#Finding the index of the lowest exposure image within a set:
uniq_dist, dist_counts = np.unique(dist_array, return_counts=True)
cum_count = 0
if distances == True:
    no_images = len(uniq_dist)
    min_indices = np.empty(no_images,dtype=int)

    for i in range(no_images):
        current_range = cum_count+dist_counts[i]
        exp_range = exp_array[cum_count:current_range]
        sorted_indices = np.argsort(exp_range)
        
        dist_array[cum_count:current_range] = dist_array[cum_count:current_range][sorted_indices]
        exp_array[cum_count:current_range] = exp_array[cum_count:current_range][sorted_indices]
        pixel_array[cum_count:current_range] = pixel_array[cum_count:current_range][sorted_indices]
        name_array[cum_count:current_range] = name_array[cum_count:current_range][sorted_indices]
        
        
        min_indices[i] = cum_count
        cum_count += dist_counts[i]
    max_indices = np.append(min_indices[1:],dist_len)  
    
else:
    no_images = 1
    sorted_indices = np.argsort(exp_array)
    exp_array = exp_array[sorted_indices]
    pixel_array = pixel_array[sorted_indices]
    pixel_err_array = np.full_like(pixel_array,pixel_err)
    name_array = name_array[sorted_indices]
    
    min_indices = np.array([0])
    max_indices = np.array([dist_len])


#Define Cartesian Coordinates: We take them as the centre of each pixel
im_shape = np.shape(pixel_array[0])
corner_coords = [im_shape[0]/2+0.1+corner_x,im_shape[1]/2+0.1+corner_y] #Acting guess of centre to find the outer ring (you don't want it to be exactly on a pixel nor half-way (if the half pixel length is used for del_theta, rather than radius))
R_guess = 200
x_offset_guess = 0   #Offset from corner_coords
y_offset_guess = 0      #Offset from corner_coords


#Subtracting the background:
for i in range(dist_len):
    if distances == True:
        difference_arr = abs(dist_array[i]-bg_dist_array)
        bg_index = np.flatnonzero(difference_arr == difference_arr.min()) #This finds the appropriate background file
        if len(bg_index) != 1: #This checks whether there are not multiple bg files at the same distance, the bg with the closest exposure time is then chosen
            diff_exp_arr = abs(exp_array[i]-np.take(bg_exp_array,bg_index))
            diff_exp_index = np.flatnonzero(diff_exp_arr == diff_exp_arr.min())
            bg_index = np.take(bg_index,diff_exp_index)
            if len(bg_index) == 2:
                bg_index = bg_index[bg_choosing_index]
            elif len(bg_index) > 2:
                print("There are more than 2 options for the background after exp. and dist. being considered!")
                bg_index = bg_index[bg_choosing_index]
            else:
                bg_index = bg_index[0] #Turns the length-1 array into a scalar
        else:
            bg_index = bg_index[0]
    else:
        diff_exp_arr = abs(exp_array[i]-bg_exp_array)
        bg_index = np.flatnonzero(diff_exp_arr == diff_exp_arr.min())[0]
        
    if background_gen == True:
        if distances == True:
            difference_arr = abs(dist_array[i]-bg_dist_err_array)
            bg_err_index = np.flatnonzero(difference_arr == difference_arr.min()) #This finds the appropriate background file
            if len(bg_err_index) != 1: #This checks whether there are not multiple bg files at the same distance, the bg with the closest exposure time is then chosen
                diff_exp_arr = abs(exp_array[i]-np.take(bg_exp_err_array,bg_err_index))
                diff_exp_index = np.flatnonzero(diff_exp_arr == diff_exp_arr.min())
                bg_err_index = np.take(bg_err_index,diff_exp_index)
                if len(bg_err_index) == 2:
                    bg_err_index = bg_err_index[bg_choosing_index]
                elif len(bg_err_index) > 2:
                    print("There are more than 2 options for the background after exp. and dist. being considered!")
                    bg_err_index = bg_err_index[bg_choosing_index]
                else:
                    bg_err_index = bg_err_index[0] #Turns the length-1 array into a scalar
            else:
                bg_err_index = bg_err_index[0]
        else:
            diff_exp_arr = abs(exp_array[i]-bg_exp_err_array)
            bg_err_index = np.flatnonzero(diff_exp_arr == diff_exp_arr.min())[0]
        
    # print("The image has exp. {} and dist. {}".format(exp_array[i],dist_array[i]))
    # print('The bg image has exp. {} and dist. {}'.format(bg_exp_array[bg_index],bg_dist_array[bg_index]))
    bg_pixels = bg_pix_array[bg_index] #finds the appropriate background pixels measured at the closest distance (rounded down at half-distance)
    
    if background_gen == True:
        scaled_bg_pix = bg_pixels
        scaled_bg_pix_err = bg_pix_err_array[bg_err_index]
        #The bg values are determined for each specific image when this method is used
    else:
        scaled_bg_pix = bg_pixels/bg_exp_array[bg_index]*exp_array[i] #This scales the background to the right exposure used
        # scaled_bg_pix_err = scaled_bg_pix*np.sqrt((pixel_err/bg_pixels)**2 + (0.005/bg_exp_array[bg_index])**2 + (0.005/exp_array[i])**2) #Propagates error using calc. method
        scaled_bg_pix_err = np.sqrt(((bg_pixels+pixel_err)/bg_exp_array[bg_index]*exp_array[i] - scaled_bg_pix)**2 +
                                    (bg_pixels/(exposure_err + bg_exp_array[bg_index])*exp_array[i] - scaled_bg_pix)**2 +
                                    (bg_pixels/bg_exp_array[bg_index]*(exposure_err+exp_array[i]) - scaled_bg_pix)**2)
        
    
    plt.figure()
    image_x = int(corner_coords[0])
    image_y = int(corner_coords[1])
    image_x_values = np.linspace(0,1280,1280)
    image_ylim = 25000
    plt.errorbar(image_x_values,pixel_array[i][:,image_y],yerr=pixel_err_array[i][:,image_y])
    plt.errorbar(image_x_values,scaled_bg_pix[:,image_y],yerr=scaled_bg_pix_err[:,image_y])
    plt.xlim([image_x-300,image_x+300])
    plt.ylim([0,image_ylim])
    if distances == True:
        plt.title('Original image with exp: '+str(exp_array[i])+" at distance: "+str(dist_array[i]))
    else:
        plt.title('Original image with exp: '+str(exp_array[i]))
    
    plt.show()
    
    pixel_err_array[i][pixel_array[i] >= saturation_value] = np.inf #All saturated pixels get an infinite error, this means that they will be averaged out when combining images
    
    pixel_array[i] -= scaled_bg_pix #This subtracts the scaled background
    pixel_err_array[i] = np.sqrt(scaled_bg_pix_err**2 + pixel_err_array[i]**2) #Propagates final error using calc method
    
    excess_indices = pixel_array[i] < 0 #Determines all pixels for which the background ref value is higher than the measured background
    excess_pix = abs(pixel_array[i][excess_indices]) #Abs values of these excess background pixels
    pixel_err_array[i][excess_indices] = np.sqrt(excess_pix**2 + pixel_err_array[i][excess_indices]**2) #Is this the right method?
    
    # larger_err_indices = pixel_err_array[i][excess_indices] < excess_pix #The indices where the excess value is larger than the error
    # pixel_err_array[i][excess_indices][larger_err_indices] = excess_pix[larger_err_indices]
    
    pixel_array[i][excess_indices] = 0

    
    #NOTE: THE SATURATION VALUE SHOULD ALSO BE REDUCED BY THE BACKGROUND -> This will be necessary for the fitting
    #THOUGH, not when the fitted image won't have a notable number of saturated pixels
    
    # plt.figure()
    # pixel_im = plt.imshow(pixel_array[i].T,interpolation=None,cmap='plasma',aspect='auto',vmin=0,vmax=saturation_value)
    # plt.colorbar(mappable=pixel_im,label='Normalised intensity')
    # plt.title(dist_array[i])
    # plt.gca().set_aspect('equal')
    # plt.xlim([0,1279])
    # plt.ylim([0,1023])
    # plt.show()
    # plt.figure()
    # pixel_im = plt.imshow(pixel_err_array[i].T,interpolation=None,cmap='plasma',aspect='auto')
    # plt.colorbar(mappable=pixel_im,label='Normalised intensity')
    # plt.title(dist_array[i])
    # plt.gca().set_aspect('equal')
    # plt.xlim([0,1279])
    # plt.ylim([0,1023])
    # plt.show()
    # plt.figure()
    # pixel_im = plt.imshow((pixel_array[i]/pixel_err_array[i]).T,interpolation=None,cmap='plasma',aspect='auto')
    # plt.colorbar(mappable=pixel_im,label='Normalised intensity')
    # plt.title('SNR')
    # plt.gca().set_aspect('equal')
    # plt.xlim([0,1279])
    # plt.ylim([0,1023])
    # plt.show()
    # plt.figure()
    # image_x = int(corner_coords[0])
    # image_y = int(corner_coords[1])
    # image_x_values = np.linspace(0,1280,1280)
    # image_ylim = 0.02*40000
    # plt.errorbar(image_x_values,pixel_array[i][:,image_y],yerr=pixel_err_array[i][:,image_y])
    # plt.ylim([0,image_ylim])
    # plt.xlim([image_x-300,image_x+300])
    # plt.title('Original image with exp: '+str(exp_array[i])+" at distance: "+str(dist_array[i]))
    # plt.show()

# sys.exit()
#General image fitting param.
R_array = np.empty(no_images)
R_err_array = np.empty(no_images)
x_offset_array = np.empty(no_images)
x_offset_err_array = np.empty(no_images)
y_offset_array = np.empty(no_images)
y_offset_err_array = np.empty(no_images)
if distances == True:
    final_dist_array = dist_array[min_indices]
else:
    final_dist_array = np.array([0])
final_exp_array = exp_array[min_indices]




#Create the folder with the final processed images
new_path = rootdir+'\\Processed'
if not os.path.exists(new_path):
    os.makedirs(new_path)

for j,i in enumerate(min_indices):
    pixels = pixel_array[i]
    # non_sat_indices = pixels < saturation_value
    # no_sat_pix = np.size(non_sat_indices)-np.count_nonzero(non_sat_indices)
    # no_sat_pix_perc = np.round(no_sat_pix/np.size(non_sat_indices)*100,3)
    # print("There are {} saturated pixels ({}%). They are not included in the fit.".format(no_sat_pix,no_sat_pix_perc))
    
    max_intensity = np.amax(pixels) # The reference parameters are w.r.t. pixels normalised to the max profile intensity
    pixels = pixels/max_intensity
    # pixel_err = pixel_err_array[i] / max_intensity #The description of the camera states 10-bits, the remaining 2-bits are considered the error
    # pixel_err=0.5 / max_intensity #For 8-bit images

    if plots == True:
        #Plot the original image:
        plt.figure()
        pixel_im = plt.imshow(pixels.T,interpolation=None,cmap='plasma',aspect='auto',vmin=0,vmax=1)
        plt.colorbar(mappable=pixel_im,label='Normalised intensity')
        plt.scatter(corner_coords[0],corner_coords[1],marker='x',color='white')
        plt.gca().set_aspect('equal')
        plt.xlim([0,1279])
        plt.ylim([0,1023])
        plt.xticks(xticks,labels=xticks*pixel_size,fontsize=fontsize)
        plt.xlabel("Horizontal image axis (mm)",fontsize=fontsize)
        plt.ylabel("Vertical image axis (mm)",fontsize=fontsize)
        plt.yticks(yticks,labels=yticks*pixel_size,fontsize=fontsize)
        
        # plt.title('Axicon distance: 18.0 +/- 1.0 mm')
        # plt.title('Axicon distance: 33.0 +/- 1.0 mm')
        # plt.title('Axicon distance: 43.5 +/- 1.0 mm')
        # plt.savefig(path[:-4]+'_orig'+'.svg',dpi=300,bbox_inches='tight')
        plt.show()

    
    
    coord_jump_x = R_jump
    coord_jump_y = R_jump
    attempts = 0
    jump_index = 0 
    
    while attempts < 20:
        attempts += 1
        x = np.arange(im_shape[0])-corner_coords[0]
        y = np.arange(im_shape[1])-corner_coords[1]
        cart_coords = np.dstack([np.dstack([x]*im_shape[1])[0],np.vstack([y]*im_shape[0])])

        #Convert to Polar coordinate system:
        r, theta, del_theta = CartPolar2(cart_coords)
        
        theta_array = np.linspace(0+del_theta[-1,int(corner_coords[1])],2*pi-del_theta[-1,int(corner_coords[1])],no_theta_points+1)[:-1]
        
        
        try:
            R_array[j], R_err_array[j], x_offset_array[j], x_offset_err_array[j], y_offset_array[j], y_offset_err_array[j] = polar_find_centre(pixels, theta_array, r, theta, del_theta, cart_coords,corner_coords,averaging_int_size,darkness_limit,drop_limit,peak_size,R_guess,x_offset_guess,y_offset_guess,ring_type,plot=plots,subplot=subplots,fontsize=fontsize,path=path,save_figs=save_fig)
            
            
            x -= x_offset_array[j]-corner_coords[0]
            y -= y_offset_array[j]-corner_coords[1]
            cart_coords = np.dstack([np.dstack([x]*im_shape[1])[0],np.vstack([y]*im_shape[0])])

            #Convert to Polar coordinate system:
            r, theta, del_theta = CartPolar2(cart_coords)
            
            corner_coords[0] = x_offset_array[j]
            corner_coords[1] = y_offset_array[j]
            break
        except Exception as ex:
            #Plot the original image:
            plt.figure()
            pixel_im = plt.imshow(pixels.T,interpolation=None,cmap='plasma',aspect='auto',vmin=0,vmax=1)
            plt.colorbar(mappable=pixel_im,label='Normalised intensity')
            plt.scatter(corner_coords[0],corner_coords[1],marker='x',color='white')
            # plt.axis('equal')
            plt.gca().set_aspect('equal')
            plt.xlim([0,1279])
            plt.ylim([0,1023])
            plt.xticks(xticks,labels=xticks*pixel_size,fontsize=fontsize)
            plt.xlabel("Horizontal image axis (mm)",fontsize=fontsize)
            plt.ylabel("Vertical image axis (mm)",fontsize=fontsize)
            plt.yticks(yticks,labels=yticks*pixel_size,fontsize=fontsize)
            plt.title("Unsuccesful fitting attempt: {}".format(attempts))
            plt.show()
            
            
            #The method below cycles the starting position from which to find the circle in an octagon increasing in radius after each full loop
            jump_index = attempts % 8
            if attempts == 1:
                corner_coords[0] += coord_jump_x
            elif jump_index == 2:
                corner_coords[1] += coord_jump_y
            elif jump_index == 3:
                corner_coords[0] -= coord_jump_x
            elif jump_index == 4:
                # coord_jump_x = abs(coord_jump_x)+R_jump
                corner_coords[0] -= coord_jump_x
            elif jump_index == 5:
                corner_coords[1] -= coord_jump_y
            elif jump_index == 6:
                # coord_jump_y = abs(coord_jump_y)+R_jump
                corner_coords[1] -= coord_jump_y
            elif jump_index == 7:
                corner_coords[0] += coord_jump_x
            elif jump_index == 0:
                corner_coords[0] += coord_jump_x
            elif jump_index == 1:
                corner_coords[0] += R_jump
                corner_coords[1] += coord_jump_y
                coord_jump_x += R_jump
                coord_jump_y += R_jump
            
            if attempts == 20:
                print("The attempt limit of 20 was reached, please change some of the variables.")
                print("A likely change necessary is the value in R_jump or the corner_x and corner_y values.")
                print("For a small circle, the darkness limit may cause the usuable area for the corner coordinates to be small, change the above values (or the darkness_limit) appropriately.")
                print("The error is: ",ex)
                sys.exit()
            # corner_coords[0] += coord_jump_x
            
            # if jump_index == 0:
            #     coord_jump_x = -coord_jump_x
            #     jump_index += 1
            # else:
            #     coord_jump_x = abs(coord_jump_x)+R_jump
            #     jump_index = 0
    print("The fit took {} attempt(s)".format(attempts))
    #Print the results:
    print("The outer radius is {} +/- {} mm".format(R_array[j]*pixel_size,R_err_array[j]*pixel_size))
    print("The outer radius is {} +/- {} pixels".format(R_array[j],R_err_array[j]))
    print("The x offset is {} +/- {} pixels".format(x_offset_array[j],x_offset_err_array[j]))
    print("The y offset is {} +/- {} pixels".format(y_offset_array[j],y_offset_err_array[j]))
    
    pixels = pixel_array[i]
    pixels_err = pixel_err_array[i]
    exp_val = exp_array[i]
    # np.linspace(0,1280,1280)
    if plots == True:
        plt.figure()
        image_x = int(corner_coords[0])
        image_y = int(corner_coords[1])
        image_x_values = np.linspace(0,1280,1280)
        image_ylim = 0.02*40000
        plt.errorbar(image_x_values,pixels[:,image_y],yerr=pixels_err[:,image_y])
        plt.ylim([0,image_ylim])
        plt.xlim([image_x-300,image_x+300])
        if distances == True:
            plt.title('Original image with exp: '+str(exp_val)+" at distance: "+str(dist_array[i]))
        else:
            plt.title('Original image with exp: '+str(exp_val))
        plt.show()
    
    for k in range(i+1,max_indices[j]):
        high_arr = pixel_array[k]
        exp_val_high = exp_array[k]
        sc_high_arr = high_arr/exp_val_high*exp_val #This scales the array to the right exposure used
        
        sc_high_arr_err = np.sqrt(((high_arr + pixel_err_array[k])/exp_val_high*exp_val-sc_high_arr)**2 + (high_arr/(exp_val_high+0.005)*exp_val-sc_high_arr)**2 + (high_arr/exp_val_high*(exp_val+0.005)-sc_high_arr)**2)

        # print(np.amin(high_arr),exp_array[k],exp_array[i])
        # sc_high_arr_err = sc_high_arr*np.sqrt((pixel_err_array[k]/high_arr)**2 + (0.005/exp_array[k])**2 + (0.005/exp_array[i])**2) #Propagates error using calc. method

        pixels, pixels_err, inner_r = avg_arrays(pixels,pixels_err, sc_high_arr,sc_high_arr_err, r, corner_coords, safety_frac)
        
        if plots == True:
            # plt.figure()
            # plt.plot([corner_coords[0]-inner_r,corner_coords[0]-inner_r],[0,saturation_value],color='orange')
            # plt.plot([corner_coords[0]+inner_r,corner_coords[0]+inner_r],[0,saturation_value],color='orange')
            # plt.errorbar(np.linspace(0,1280,1280),sc_high_arr[:,512],yerr=sc_high_arr_err[:,512])
            # # plt.plot(orig_pixels[:,512])
            # plt.ylim([0,0.1*40000])
            # plt.xlim([corner_coords[0]-300,corner_coords[0]+300])
            # plt.title('The scaled image to be added with exp: '+str(exp_val_high)+" at distance: "+str(dist_array[k]))
            # plt.show()
            
            plt.figure()
            plt.plot([image_x-inner_r,image_x-inner_r],[0,saturation_value],color='orange')
            plt.plot([image_x+inner_r,image_x+inner_r],[0,saturation_value],color='orange')
            plt.errorbar(image_x_values,pixels[:,image_y],yerr=pixels_err[:,image_y])
            plt.ylim([0,image_ylim])
            plt.xlim([image_x-300,image_x+300])
            if distances == True:
                plt.title('Added image with exp: '+str(exp_val_high)+" at distance: "+str(dist_array[k]))
            else:
                plt.title('Added image with exp: '+str(exp_val_high))
            plt.show()

    if save_info == True:
        pixels_df = pd.DataFrame(pixels)
        pixels_df.to_csv(new_path+'\\'+name_array[i][:-4]+'.csv')
        
        pixels_err_df = pd.DataFrame(pixels_err)
        pixels_err_df.to_csv(new_path+'\\'+name_array[i][:-4]+'_err.csv')




# print(final_dist_array, final_exp_array,R_array, R_err_array, x_offset_array, x_offset_err_array, y_offset_array, y_offset_err_array)
# Write results
if save_info == True:
    d = {'Distance (mm)':final_dist_array,'Exposure time (us)':final_exp_array,'Radius (pixels)':R_array,'Radius error (pixels)':R_err_array,
         'X offset (pixels)':x_offset_array,'X offset error (pixels)':x_offset_err_array,'Y offset (pixels)':y_offset_array,
         'Y offset error (pixels)':y_offset_err_array}
    dataframe = pd.DataFrame(d)
    dataframe.to_csv(new_path+'\\'+directory_name+'.csv')



# dataframe = pd.DataFrame([[dist_array, exp_array,R_array, R_err_array, x_offset_array, x_offset_err_array, y_offset_array, y_offset_err_array],
#                           columns=['Distance (mm)','Exposure time (us)','Radius (pixels)','Radius error (pixels)','X offset (pixels)','X offset error (pixels)','Y offset (pixels)','Y offset error (pixels)'],
#                           dtype=float)
# dataframe.to_csv(directory_name+'.csv')

#Method for combining images:
#Take N lines in the polar coord system from the circular centre
#For each line take the points within the fitted radius
#Find the nearest value with a non-zero weighting
#Note, all the errors can be inverted to give an array of weights for the fitting
#After subtracting the backgrounds earlier the weightings of any saturated values can be turned to zero
#Further note: the indices of these points can be determined beforehand

#The closest nan value over all lines sets the inner radius of the section that's used
#Can add a safety percentage to make sure no bloomed pixels are used (bleeding), note: no streaking has been observed
#This percentage can be determined by comparing scaled images
#It can also be determined from the point were the value of the non-sat image scaled is higher than the saturated image value
#However, that method does not work at every angle -> not a consistent approach

#Can also add them together before subtracting the background
#Dediced to subtract the background first in case there are multiple better background files. Additionally, this way it reduces bad points by picking them out at each stage. Furthermore, by doing so it also counteracts the effect of bad linearity fitting of the exposure scaling
#Either way, it will need to be done after centre finding

#To combine them, can set all the weights to zero in the scaled image where the pixels fall outside of the ring
