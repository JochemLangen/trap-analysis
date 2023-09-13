*** Guide to usage of the ring analysis software ***

-- Overview --

This folder contains the files that can be used to analyse a trap profile that has been imaged.
The files can be run consecutively, to see and potentially export intermediate results, or their contents can be copied into a single file to be run together.

The folder contains the following files:
- centre_photos (v1-v3)
- centre_photos_convergence (v1-v2)
- analyse_photos (v1-v5)
- comb_measurements
- dbl_comb_measurements
- funcs_photos
- generate_bg
- master_plot


-- "centre_photos" programme --
In a very broad sense, this programme does the following:
1. Extract & sort all images and parameters from the specified folder
3. Subtract the background using the appropriate and scaled background file
4. Find the position of the centre of the trap as well as its radius.
5. Combine the different images of the same profile (photos with varying exposure times to obtain a more detailed view of different intensity regimes).
The most recent code can be found in centre_photos_v3.py which uses functions from funcs_photos.py

How to use this programme:
1. Change rootdir to the appropriate folder that contains all the images of the profile that should be compared together
i.e. images of the profile with different exposure times, their background light measurements for the profile at each external parameter (e.g. distance along the optical axis)
The image files should have the following format:
"xx_xxus_xxmm.pgm"
Where the first "xx" can be given any name that does not contain "us" or "mm". The second "xx" contains the total exposure value float number, without the decimal point.
The third "xx" contains the total distance value float number, without the decimal point. 
NOTE: if no distances are used, the last part can be left off the file name.

The background files should be of the form:
"xx_xxus_xxmm_bg.pgm"

If an additional set of images is taken with an adjusted camera position to average out the imaging inaccuracy, these should be placed in a folder called "Adj_cam" inside the original folder. 
These are analysed separately in this file, but some of the analysis may differ.

2. Inside the file, the exp_float_factor needs to be set as the number by which the total exposure number integer should be divided to obtain the correct float value.
3. exposure_err should be updated to be the correct value.
4. saturation_value and pixel_err should be adjusted to reflect the number of bits of the pixels and the reported error from the camera.
5. There is the option further down to set single_dir to false. If this is done here then the programme will also explore subfolders.
6. corner_x and corner_y are the offsets from the centre of the image that are used for the initial "guess" of the centre. As long as this guess is within the circle, the algorithm should find the right position.
7. ring_type determines the polar orientation of the coordinate system and allows the possibility to flip the ring inside out. This should be kept at "inner" as the "outer" functionality has become deprecated.
8. The parameters: plots, subplots, save_fig & save_info determine what output information to generate.
9. distances sets whether a distance parameter is used (which would also be reflected in the file naming).
10. dist_float_factor is the number by which the total distance number integer should be divided to obtain the correct float value.
11. background_gen is set to False when background images are used and true when the "generate_bg" file has been used to generate background estimates.
12. pixel_size should hold the pixel size of the camera used. The currently filled in numbers form an average of the specified pixel size for each dimension.
13. The xticks and yticks should hold the values of the tick labels you want to use multiplied with the pixel size.
14. The following parameters are used within the centre finding algorithm and may be adjusted if the algorithm does not manage to consistently find the centre. They have been determined to provide good results for all of the different set-ups used previously:
- no_theta_points: The number of angles in each fitting range. This determines the number of points used to determine the centre of the circle (and circle radius).
- averaging_int_size: The number of points over each radial line of pixels is averaged to get a smooth curve
- darkness_limit: The minimum value of any pixel to be considered as part of the peak
- drop_limit: To be the drop-off from the first peak, the intensity difference between points must be below this value
- peak_size: When the peak has been found from the averaged points, this parameter determines the +/- area around this point within which the individual peak pixel might fall (determined by the max of this range). This is taken to be the same as the averaging interval.
- R_jump: The jump in corner coordinates if the circle centre guess does not fall in the centre
15. safety_frac sets the safety fraction on the used radius in combining the images. The saturated pixel closest to the centre sets the limit to what should be combined. This adds an additional safety radius reduction to avoid artifacts caused by bleeding.
16. In the unlikely scenario the fitting is not working well even after adjustment of the above parameters, further down in the file R_guess can be adjusted as the guess of the circle radius.
17. If save_info is set to true, an additional folder called "Processed" will be made inside the data folder which contains the results. If save_fig is set to true, the plots will be generated within the main data folder.

The programme generates the following output:
-The ring radius and its error.
-The x position of the circle centre and its error.
-The y position of the circle centre and its error.
-For ring, the final image with the background subtracted and with the relevant sections averaged between exposures.


-- "analyse_photos" programme --
In a broad sense the programme does the following:
1. Import and prepare all the data (sorting, generating coordinate systems, determining the right angles).
2. It splits the ring profile into a set number of sections and bins all datapoints onto a grid defined by the radius.
3. Identify the features of the profile.
4. Fit a power law to the inner ring profile.
5. Generate all the output parameters (see below) by comparing the different sections and parameter post analysis.
6. Provide various options of plot generation to show the profile parameters.

This programme takes the images from the centre_photos and analyses each of them to generate the following parameters and their errors:
- The parameters it imported from "centre_photos": ring radius and centre x and y position from the image centre.
- The standard deviation of the ring radius along the ring.
- The average exponent of the power law that is fitted to the inside of the ring.
- The standard deviation in the power law exponent along the ring.
- The average normalised constant of the power law fit (normalised to the profile mean intensity).
- The standard deviation in the normalised power law constant.
- The average normalised peak intensity of the ring (normalised to the profile mean intensity).
- The standard deviation in the normalised peak intensity.
- The average relative power law constant, i.e. darkness (relative to the peak intensity).
- The average residual standard deviation around the fit.
- The standard deviation of the residual std. around the fit.
- The standard deviation of all residuals together.

How to use this programme:
1. Change rootdir to the same folder as was used in "center_photos". 
Note: use the original folder, not the "processed" folder.
3. Top index determines the image that you want to set as the main image for plotting. If the report_plot parameter (see below) is set to true, this is the only image that is analysed.
2. dist_float_factor is the number by which the total distance number integer should be divided to obtain the correct float value.
3. The dist_err (the error of the distance measurement) should be updated.
4. Exposure_err and exp_ms_err, the exposure time errors are not used in this file anymore but may be put here for personal future reference.
4. pixel_pos_err, is the pixel error. As the pixel size is the unit size of distance, this error is 0.5.
5. max_pixel_value should be adjusted to reflect the number of bits of the pixels and describes the saturation value.
6. The following parameters are used within the peak finding algorithm and may be adjusted if the algorithm does not manage to consistently find the centre. They have been determined to provide good results for all of the different set-ups used previously:
- averaging_int_size: The number of points over each radial line of pixels is averaged to get a smooth curve which is used here to find the trap profile shape and the fitting area.
- peak_size: When the the right fitting positions have been found from the averaged points, this parameter determines the +/- area around this point within which the individual cut-off pixel might fall. This is taken to be the same as the averaging interval.
- darkness_limit: The minimum value of any pixel to be considered as part of the peak in the peak finding algorithm, separate from the two parameters above.
- drop_limit: To be the drop-off from the first peak, the intensity difference between points must be below this value in the peak finding algorithm, separate from the first 2 parameters.
- peak_finding_avg_int_size: The averaging number of points used to find the peak in the peak finding algorithm, separate from the first 2 parameters. This corresponds to averaging_int_size in the "centre_photos" programme

7. The parameters: plots, subplots, prints, slice_plot, report_plot, bad_fit_prints, image_result & save_info determine what output information to generate.
8. distances sets whether a distance parameter is used (which would also be reflected in the file naming).
9. Guesses for the power law fit parameters: m_guess, a_guess, b_guess.
10. Plotting parameters: fontsize, tickwidth, ticklength.
11. no_theta_areas, the number of areas to divide the ring up in. All the pixels in one of these areas will be binned onto a grid and the results from these section are compared to find the ring symmetry through the standard deviation
12. delta_r is the interval over which the pixels are averaged (in pixels) when creating the sections
13. There is the option further down to set single_dir to false. If this is done here then the programme will also explore subfolders.
14. If save_info is set to true, the results will be in a file with the same name as the one created from "centre_photos" though with "_analysed" added to the end.


-- "centre_photos_convergence" programme --
This file can be used to test the convergence of the centre finding algorithm with varying number of points to find the parameters from.
This file is based on v2 from "centre_photos" but may be used to test the convergence of the algorithm for the relevant trap.


-- "generate_bg" programme --
This file can be used to create a background estimate in case a background image has not or cannot be reasonably taken for the relevant trap image.
It makes use of an interpolation between areas on the outside of the image that are not part of the profile, determined with a relevant algorithm.

How to use this programme:
1. Change rootdir to the appropriate folder that contains all the images of the profile that should be compared together
2. Inside the file, the exp_float_factor needs to be set as the number by which the total exposure number integer should be divided to obtain the correct float value.
3. exposure_err should be updated to be the correct value.
4. There is the option further down to set single_dir to false. If this is done here then the programme will also explore subfolders.
5. saturation_value and pixel_err should be adjusted to reflect the number of bits of the pixels and the reported error from the camera.
6. The parameters: plots & save_info determine what output information to generate.
7. pixel_size should hold the pixel size of the camera used. The currently filled in numbers form an average of the specified pixel size for each dimension.
8. The xticks and yticks should hold the values of the tick labels you want to use multiplied with the pixel size.
9. The fontsize can be set.
10. averaging_int_size is used to average the image profile to get a smoother result and average out noise, used to find the ring profile.
11. avg_interval the area used to look for the specific relevant pixel. Set equal to averaging_int_size
12. Difference_limit is the parameter used to compare the data to as the hard-coded value to determine with where the profile begins, this may be adjusted if the results are not reasonable.
13. If save_info is set to true, the file generates background csv files for each image and a background error csv.


-- "comb_measurements" programme --
This file is used to average the results from the data set that was taken with a slightly adjusted camera position. 
It is used to average out the inaccuracy of the camera positioning. This data should be put in a folder within the main image folder called "Adj_cam". 
This folder needs to be processed using the programmes above as well.

The programme provides further analysis of the trap parameters, generating relative standard deviations of the various parameters.
Additionally, the programme uses the distance values corresponding to each image to determine the linear parameter variability around a given trap layout.

How to use this programme:
1. Change rootdir_unpr_1 to the appropriate folder that contains the original set of images (not the processed subfolder) and that contains the "Adj_cam" folder.
2. The top_index gives the index of the main image in the trap that is used to extract the parameters from and to determine the linear parameter variation around.
3. In case any values at the end of the array should not be used, "last_index" should be set to what will be used or 0 if everything should be.
4. Set the fontsize
5. Set the error in the image distances
6. If save_results is set to true, the new parameters are saved in a folder given by the "results_folder" path towards the end of the file. 
All the results in this folder can later be used to compare different data sets / traps.
7. distances determines whether the linear variation should be determined.
8. The pixel_size and pixel_size_err (error) should be adjusted to represent the camera used. The default currently present uses a weighted average based on the slightly different pixel size numbers present for the camera (due to rounding).


-- "dbl_comb_measurements" programme --
This programme is similar to the "comb_measurements" programme though does not provide the linear parameters.
It can be used to generate a plot with two image sets added together, e.g. two data sets that fall on a distance scale after one-another.


-- "master_plot" programme --
This programme is used to generate a plot with an overview of all the trap parameters of various different data sets and traps. 
It also determines the efficiency of the traps.

How to use the programme:
1. Change rootdir to the folder that was used in "comb_measurements" to store the results.
2. Change the plotting parameters: fontsize, tickwidth, ticklength, mtickwidth and mticklength. plot determines whether the graph is provided or not.
3. Further down in the file, markers and labels needs to be adjusted to represent the correct number of traps that are being compared with the right labels.
4. The section just below these parameters contains the efficiency calculation. This needs to be adjusted depending on the measured efficiencies. The final value used is the trap peak intensity divided by the gaussian peak intensity coming out of the fibre.
5. The ratios calculations further down allow the numerical comparison of different parameters, with the traps selected using the indices.
6. At the bottom, the layout of the plot can be changed including which axes are logarithmic.
7. The final result will be saved in the rootdir folder.


-- "funcs_photos" --
This file contains additional functions used by the programmes that were not defined in those scripts.


-- Camera integration --
Depending on the camera that is used, there may be an option to integrate it with python. 
This was the case for the camera used in the research report: CM3-U3-13Y3C-CS 1/2" Chameleon®3 Color Camera.

Camera integration can provide quicker feedback on the quality of some of the profile parameters, such as the radial symmetry.

The v2 version of centre photos, could be coupled to the camera for this use.
This version lacks the folder extraction and the subtraction of the background, which requires multiple images and would not be possible live.
The no_theta_points variable can be significantly reduced to obtain a slightly worse but faster estimate of the results.
For an idea of the number that should be used, the plots can be investigated or centre_photos_convergence_v2 can be used.

v5 of the analyse_photos file provides the most accurate analysis. However, it is based on the use of a folder. This part could be deleted when integrated with the camera.
As the photos that are used would not have their backgrounds subtracted, some of the reference values in both analyse_photos may need to be verified dependent on the background.
To do so, the plots can be generated to see whether the right elements of the ring profile are identified in the fitting processes.



For any questions regarding these programmes, please contact Jochem Langen via: jochem@langen.one

