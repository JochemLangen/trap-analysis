# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 11:14:53 2023

@author: joche
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

path = "D:\\Jochem\\Documents\\Uni\Physics\\Level 4 Project\\New_col\\Flipped_ring_axicon\\Dist_range\\Hollow_beam_149995us_8000mm.pgm"
# path = "D:\\Jochem\\Documents\\Uni\Physics\\Level 4 Project\\New_col\\Ring_axicon_lens\\New_set\\Hollow_beam_500000us_14000mm_bg.pgm"
# Hollow_beam_1317704us_0000mm
im = Image.open(path)

pixels = np.array(im, dtype=float)
im_shape = np.shape(pixels)

new_exposure = 9999.99 #microseconds
original_exposure = 1499.95 #microseconds
saturation_value = 2**16-2**4

plt.figure()
pixels_im = plt.imshow(pixels)#,vmin=2500,vmax=5500)
plt.colorbar(mappable=pixels_im,label='Pixel intensity')
plt.title('Original image: exposure = {}'.format(original_exposure))
plt.show()

# plt.figure()
# plt.plot(pixels[512,:])
# plt.show()

pixels *= new_exposure/original_exposure
print("The image is scaled by {}".format(new_exposure/original_exposure))
pixels[pixels > saturation_value] = saturation_value 
new_im = Image.fromarray(pixels.astype(np.uint8),mode='L') #L mode is the bitmap unint8 mode of images

pixels = pixels/saturation_value
plt.figure()
pixels_im = plt.imshow(pixels,vmin=0,vmax=0.3)
plt.colorbar(mappable=pixels_im,label='Pixel intensity')
plt.title('Scaled image: "exposure" = {}'.format(new_exposure))
plt.show()

# new_path = path[:-4]+"_exp_{}".format(new_exposure)+".bmp"
# print(new_path)
# new_im.save(new_path)
im_ref = Image.open("New_col\\Flipped_ring_axicon\\Dist_range\\Hollow_beam_999999us_8000mm.pgm")
plt.figure()
pixels_ref = np.array(im_ref, dtype=float)/saturation_value
pixels_ref_im = plt.imshow(pixels_ref,vmin=0,vmax=0.3)
plt.colorbar(mappable=pixels_ref_im,label='Pixel intensity')
plt.title('Ref image: "exposure" = {}'.format(new_exposure))
plt.show()
print(np.amax(pixels_ref))
print(saturation_value)

#Show difference:
resid = pixels_ref - pixels
plt.figure()
resid_im = plt.imshow(resid,vmax=0,vmin=-0.0000002)
plt.colorbar(mappable=resid_im,label='Pixel intensity')
plt.title('Residuals image: "exposure" = {}'.format(new_exposure))
plt.show()
print(np.amax(resid))