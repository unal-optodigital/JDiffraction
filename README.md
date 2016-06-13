# JDiffraction
JDiffraction is a numerical wave propagation library for Java. Includes angular spectrum, Fresnel-Fourier and [Fresnel-Bluestein](http://dx.doi.org/10.1364/AO.49.006430) methods. Aditionally it includes an utilities class, designed to work with the complex arrays required by the library.

The library supports calculation on CPU and GPU, the latter is done using [JCuda](http://www.jcuda.org/). To use the GPGPU versions of the library, [JCuda](http://www.jcuda.org/) 0.7.5 jars must be in the java path, CUDA 7.5 and a CUDA capable GPU must be installed on the PC.

If you are using JDiffraction and find a bug, please contact us.

## API
JDiffraction API can be found [here](http://unal-optodigital.github.io/JDiffraction/javadoc/index.html).

## Downloads
`.jar` files and `.zip` for APi can be found in the [downloads](http://unal-optodigital.github.io/JDiffraction/#downloads) section of the project [page](http://unal-optodigital.github.io/JDiffraction/).

## Credits
JDiffraction uses [JTransforms](https://sites.google.com/site/piotrwendykier/software/jtransforms) FFT routines and [JCuda](http://www.jcuda.org/) Java bindings for NVIDIA CUDA.

## Contact

- Pablo Piedrahita-Quintero ([jppiedrahitaq@unal.edu.co](mailto:jppiedrahitaq@unal.edu.co))
- Carlos Trujillo ([catrujila@unal.edu.co](mailto:catrujila@unal.edu.co))
- Jorge Garcia-Sucerquia ([jigarcia@unal.edu.co](mailto:jigarcia@unal.edu.co))


