# MATLAB motion PSF experiment

`my_motion_psf.m` is the recovered standalone MATLAB implementation of a motion-blur point-spread function. `test_motion_psf.m` compares with MATLAB `fspecial('motion', ...)`, which requires Image Processing Toolbox. The historical report records a 63-case comparison with zero maximum absolute difference, **but MATLAB has not been rerun during this GitHub packaging**. Results should be verified independently in the user's MATLAB environment.

```matlab
cd lab03_motion_psf/matlab
test_motion_psf
```

This motion PSF experiment is separate from inverse/Wiener deconvolution; the latter is not claimed as completed here.
