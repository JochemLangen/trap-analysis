*** Overview of the light propagation simulation software ***

The files in this folder provide the additional functions necessary for the simulations and several example and test files.
Alongside these files, the python Lightpipes library needs to be installed: https://opticspy.github.io/lightpipes/ .

File overview:
-LightpipesFuncs: this file contains the additional optical element functions and a function to plot the shape of this element.
This regards a correct version of the axicon, rather than the built-in Lightpipes axicon, which at the time of writing did not appear to provide correct results for the use of simulating optical ring traps.
-OpticsFuncs: this file provides various functions that can be used to plot the simulated results.
-Optical_SetUp, test_file and test_file: these files provide examples of tests with the simulation software relevant for the trapping profile
-example and example2: these files provide examples with the original Lightpipes axicon function
-HedgehogVsAnalytic: this file contains a validation of the Hedgehog propagation method (implemented in Lightpipes through the function Forvard).
This file is provided by Prof Stuart Adams. For further details, see Optics f2f: From Fourier to Fresnel by Adams, C. S. & Hughes, I. G. .

For further details on lightpipes and the simulations, please see the website above or contact the author Jochem Langen via: jochem@langen.one
