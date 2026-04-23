# Noise reduction

For particularly noisy image data, you can add an optional noise reduction step in the pipeline.
This can be inserted anywhere it is useful, for example between image loading and background correction,
or just before segmentation.

## Example: Gaussian noise reduction before background correction

```toml
[steps.load]
pipeline_class = "pyopia.instrument.silcam.SilCamLoad"

[steps.noisereduction]
pipeline_class = "pyopia.noise.ReduceNoise"
method = "gaussian"
image_source = "imraw"
gaussian_sigma = 1.0

[steps.correctbackground]
pipeline_class = "pyopia.background.CorrectBackgroundAccurate"
bgshift_function = "accurate"
average_window = 5
image_source = "im_denoised"
```

## Example: CLAHE before segmentation

```toml
[steps.imageprep]
pipeline_class = "pyopia.instrument.silcam.ImagePrep"
image_level = "im_corrected"

[steps.noisereduction]
pipeline_class = "pyopia.noise.ReduceNoise"
method = "clahe"
image_source = "im_minimum"
clahe_clip_limit = 0.01
clahe_nbins = 256

[steps.segmentation]
pipeline_class = "pyopia.process.Segment"
threshold = 0.85
segment_source = "im_denoised"
```
