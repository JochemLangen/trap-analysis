# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 12:06:06 2023

@author: joche
"""
import os
import sys
from funcs_photos import *
import pandas as pd
from scipy import interpolate

# rootdir = 'D:\\Jochem\\Documents\\Uni\\Physics\\Level 4 Project\\New_col\\Ring_axicon\\2_degrees'

# rootdir = 'D:\\Jochem\\Documents\\Uni\\Physics\\Level 4 Project\\New_col\\Ring_lens_axicon\\2_degrees'
# rootdir = 'D:\\Jochem\\Documents\\Uni\\Physics\\Level 4 Project\\New_col\\Ring_lens_axicon\\Ax_dist_range'
# rootdir = 'D:\\Jochem\\Documents\\Uni\\Physics\\Level 4 Project\\New_col\\Ring_lens_axicon\\Obs_obj'
# rootdir = 'D:\\Jochem\\Documents\\Uni\\Physics\\Level 4 Project\\New_col\\Ring_axicon_lens\\Obs_obj'
# rootdir = 'D:\\Jochem\\Documents\\Uni\\Physics\\Level 4 Project\\New_col\\Ring_axicon_lens\\Obs_obj\\closer_obj'
rootdir = 'D:\\Jochem\\Documents\\Uni\\Physics\\Level 4 Project\\New_col\\Flip_focus_ring_axicon\\New_set\\Iris_range'

pixel_array = []
exp_array = []
exp_float_factor = 100
exposure_err = 0.005 #In us
name_array = []

single_dir = True

saturation_value = 2**16-2**4
pixel_err = 2**6

plots = True
save_info = True
pixel_size = np.average([6.14/1280,4.9/1024]) #In mm
xticks = np.array([0,1.2,2.4,3.6,4.8,6])/pixel_size
yticks = np.array([0,0.9,1.8,2.7,3.6,4.5])/pixel_size
fontsize = 13
averaging_int_size = 20
avg_interval = 20
difference_limit = 0.0005
# difference_limit = 0.001

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
            exp_last_index = path.rfind("us")
            ereduced_path = path[:exp_last_index]
            print(ereduced_path)
            if path.find('no') == -1: #Images with no in them will not be used
                if path.find('bg') == -1:
                    exp_array += [float(ereduced_path[ereduced_path.rfind("_")+1:exp_last_index])/exp_float_factor]
                    name_array += [path[path.rfind("\\")+1:]]
                    
                    pixel_array += [np.array(im).T]
                    # pixel_im = plt.imshow(np.array(im),interpolation=None,cmap='plasma',aspect='auto',vmin=0,vmax=5000)#,vmax=saturation_value)
                    # plt.gca().set_aspect('equal')
                    # plt.xlim([0,1279])
                    # plt.ylim([0,1023])
                    # plt.colorbar(mappable=pixel_im,label='Normalised intensity')
                    # plt.title(file)
                    # plt.show()

exp_array = np.asarray(exp_array)
pixel_array = np.asarray(pixel_array,dtype=float)
name_array = np.asarray(name_array,dtype=str)
exp_len = len(exp_array)

sorted_indices = np.argsort(exp_array)
exp_array = exp_array[sorted_indices]
pixel_array = pixel_array[sorted_indices]
name_array = name_array[sorted_indices]

#Result arrays
bg_array = np.copy(pixel_array)
bg_err_array = np.full_like(bg_array,pixel_err)
pix_std_array = np.empty(exp_len)

indices = np.empty(exp_len,dtype=int)
# indices[:2] = np.array([30,1270],dtype=int)

im_shape = np.shape(pixel_array[0])
indices = np.empty((2,im_shape[1],exp_len),dtype=int)
R_pix_SE_array = np.empty((im_shape[1],exp_len))
L_pix_SE_array = np.empty((im_shape[1],exp_len))


x_values = np.arange(im_shape[0])
# x_range = np.asarray([np.average(x_values[:indices[0]]),np.average(x_values[indices[1]:])])


for i in range(exp_len):
    L_pix_resid = []
    R_pix_resid = []
    no_avg_lines = 0
    max_val = np.amax(pixel_array[i])
    for k in range(im_shape[1]):
        #Finding the point where the profile begins:
        pixel_line = pixel_array[i][:,k]
        avg_pix = interval_avg(pixel_line, averaging_int_size)
        avg_diff = (avg_pix[1:] - avg_pix[:-1])/max_val

        trough_array = np.asarray((avg_diff >= difference_limit).nonzero())
        if np.size(trough_array) <= 1:
            continue
        
        avg_trough_index_L = (trough_array[0,0]-1)*averaging_int_size
        
        
        if avg_trough_index_L <= 0:
            indices[0,k,i] = avg_interval
        else:
            # print(pixel_line[avg_trough_index_L:avg_trough_index_L+averaging_int_size])
            # print(avg_trough_index_L)
            # print()
            indices[0,k,i] = avg_trough_index_L+np.argmin(pixel_line[avg_trough_index_L:avg_trough_index_L+averaging_int_size])
        
        # plt.figure()
        # plt.plot(avg_diff)
        # plt.scatter(x_values[int(indices[0,k,i]/averaging_int_size)],avg_diff_L[int(indices[0,k,i]/averaging_int_size)])
        # plt.show()
        # plt.figure()
        # plt.plot(pixel_line)
        # plt.scatter(x_values[indices[0,k,i]],pixel_line[indices[0,k,i]])
        
        trough_array = np.asarray((avg_diff <= -difference_limit).nonzero())
        if np.size(trough_array) <= 1:
            continue
        
        avg_trough_index_R = (trough_array[0,-1]+1)*averaging_int_size
        #Get the index of the point furthest out where this difference exceeds the reference value
        # print(pixel_line[:avg_trough_index])
        if avg_trough_index_R == len(pixel_line)//averaging_int_size:
            indices[1,k,i] = avg_interval
        else:
            indices[1,k,i] = avg_trough_index_R+np.argmin(pixel_line[avg_trough_index_R:avg_trough_index_R+averaging_int_size])
        
        # plt.figure()
        # plt.plot(avg_diff)
        # plt.scatter(x_values[int(indices[1,k,i]/averaging_int_size)],avg_diff_L[int(indices[1,k,i]/averaging_int_size)])
        # plt.show()
        # plt.figure()
        # plt.plot(pixel_line)
        # plt.scatter(x_values[indices[1,k,i]],pixel_line[indices[1,k,i]])
        
        # print(indices[0,k,i])
        # print(type(indices[0,k,i]))
        
        R_pixels = pixel_array[i][indices[1,k,i]:indices[1,k,i]+avg_interval,k]
        # print(R_pixels)
        R_pix = np.average(R_pixels,axis=0)
        R_pix_resid += list(R_pixels-R_pix)
        R_pix_SE_array[k,i] = np.std(R_pixels)/np.sqrt(len(R_pixels))
        
        L_pixels = pixel_array[i][indices[0,k,i]-avg_interval:indices[0,k,i],k]
        L_pix = np.average(L_pixels,axis=0)
        L_pix_resid += list(L_pixels-L_pix)
        L_pix_SE_array[k,i] = np.std(L_pixels)/np.sqrt(len(L_pixels))
        
        pix = np.asarray([L_pix,R_pix])
        x_range = np.asarray([np.average(x_values[indices[0,k,i]-avg_interval:indices[0,k,i]]),
                              np.average(x_values[indices[1,k,i]:indices[1,k,i]+avg_interval])])
        
        f = interpolate.interp1d(x_range,pix,axis=0)
        bg_array[i][indices[0,k,i]:indices[1,k,i]-1,k] = f(x_values[indices[0,k,i]:indices[1,k,i]-1])
        x_weights_L = (x_values[indices[0,k,i]:indices[1,k,i]-1]-x_values[indices[0,k,i]])/(x_values[indices[1,k,i]]-x_values[indices[0,k,i]])
        x_weights_R = abs((x_values[indices[0,k,i]:indices[1,k,i]-1]-x_values[indices[1,k,i]])/(x_values[indices[1,k,i]]-x_values[indices[0,k,i]]))
        
        # print(x_weights_L)
        # print(x_weights_R)
        bg_err_array[i][indices[0,k,i]:indices[1,k,i]-1,k] = np.sqrt((L_pix_SE_array[k,i]*x_weights_L)**2 + (R_pix_SE_array[k,i]*x_weights_R)**2 + pixel_err**2)
        no_avg_lines += 1
        
    R_pix_resid = np.asarray(R_pix_resid)
    L_pix_resid = np.asarray(L_pix_resid)
    pix_std_array[i] = np.sqrt(np.sum(np.append(R_pix_resid**2,L_pix_resid**2))/(np.size(R_pix_resid) + np.size(L_pix_resid) - 2*no_avg_lines))
    bg_err_array[i] = np.sqrt(bg_err_array[i]**2 + pix_std_array[i]**2)
    print(pix_std_array[i])

    # print(pix_std)
    
    
    
    # R_pixels = pixel_array[i][indices[1]:,:]
    # R_pix = np.average(R_pixels,axis=0)
    # R_pix_resid = R_pixels-R_pix
    # R_pix_SE_array[i] = np.std(R_pixels)/np.sqrt(len(R_pix))
    
    # L_pixels = pixel_array[i][:indices[0],:]
    # L_pix = np.average(L_pixels,axis=0)
    # L_pix_resid = L_pixels-L_pix
    # L_pix_SE_array[i] = np.std(L_pixels)/np.sqrt(len(L_pix))

    # pix_std_array[i] = np.sqrt(np.sum(np.append(R_pix_resid**2,L_pix_resid**2))/(np.size(R_pix_resid) + np.size(L_pix_resid) - 2))
    # print(pix_std_array[i])
    

    # # print(pix_std)
    # pix = np.asarray([L_pix,R_pix])
    # f = interpolate.interp1d(x_range,pix,axis=0)
    # bg_array[i][indices[0]:indices[1]-1] = f(x_values[indices[0]:indices[1]-1])

    if save_info == True:
        bg_df = pd.DataFrame(bg_array[i])
        bg_df.to_csv(rootdir + '\\' + name_array[i][:-4] + '_bg.csv')
        bg_err_df = pd.DataFrame(bg_err_array[i])
        bg_err_df.to_csv(rootdir + '\\' + name_array[i][:-4] + '_bg_err.csv')
        
    if plots == True:
        plt.figure()
        plt.plot(np.append(R_pix_resid,L_pix_resid))
        plt.show()
    
        pixel_im = plt.imshow(bg_array[i].T,interpolation=None,cmap='plasma',aspect='auto',vmin=0,vmax=5000)#,vmax=saturation_value)
        plt.gca().set_aspect('equal')
        plt.xlim([0,1279])
        plt.ylim([0,1023])
        plt.colorbar(mappable=pixel_im,label='Normalised intensity')
        plt.title(name_array[i])
        plt.show()
        pixel_im = plt.imshow(pixel_array[i].T,interpolation=None,cmap='plasma',aspect='auto',vmin=0,vmax=5000)#,vmax=saturation_value)
        plt.gca().set_aspect('equal')
        plt.xlim([0,1279])
        plt.ylim([0,1023])
        plt.colorbar(mappable=pixel_im,label='Normalised intensity')
        y_values = np.array([np.linspace(0,1023,1024),np.linspace(0,1023,1024)])
        plt.scatter([indices[0,:,i],indices[1,:,i]],y_values,marker='.')
        plt.title(name_array[i])
        plt.show()
        plt.figure()
        image_ylim = 5000
        plt.errorbar(x_values,pixel_array[i][:,400])
        plt.errorbar(x_values,bg_array[i][:,400],yerr=bg_err_array[i][:,400])
        # plt.xlim([image_x-300,image_x+300])
        plt.ylim([0,image_ylim])
        plt.title('Original image with exp: '+str(exp_array[i]))
        plt.show()
    #The method below works when only one column of pixels is used:
    # difference_array = np.dstack([R_pix-L_pix]*im_shape[0])[0]
    # bg_array[i] = (difference_array/im_shape[0] * x_values).T + L_pix
    
# if save_info == True:
#     result_path = rootdir + '\\background_data.csv'

#     indices[2:] = 0
#     d = {"Bg data std.":pix_std_array,'R-avg. SE':R_pix_SE_array,'L-avg. SE':L_pix_SE_array,'Indices':indices}
#     #'Norm. const.':avg_b_array[top_index],
#     # 'Norm. const. error':avg_b_err_array[top_index],
#     # 'Total resid. std':total_std_array[top_index],
#     # 'Total resid. std error':total_std_err_array[top_index]
    
#     dataframe = pd.DataFrame(d)
#     dataframe.to_csv(result_path)
