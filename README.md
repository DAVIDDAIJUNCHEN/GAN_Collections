# GAN Model Collection

## Overview

GAN (generative adversial networks) and some variants implementation.

### List of GAN and its variants:
* InfoGan: infomative GAN model.
* AeGan: auto-encoder GAN model.
* WGan-GP: Wasserstein GAN model with gradient peanlty. 
* LsGan: least-square GAN model.
* Gan-cls: LSGAN model with features and labels together in input of discriminator.

## Requirements

- [Python 3.5+](https://anaconda.org/anaconda/python)
- [TensorFlow 1.3.0+](https://www.tensorflow.org/install/)
 
## Usages

To train/infer/generate GAN models, you may need to run codes by following the steps in each model folder:

the below steps are based on InfoGAN model, others have similar steps,
### Train GAN models

```
> python Train_Mnist.py 

```

<img align='center' img src="./GAN/InfoGan/images/InfoGan_Train.PNG">

### Infer (classify) a new sample in GAN models

```
> python Inference_Mnist.py
```

<img align='center' img src="./GAN/InfoGan/images/InfoGan_Inference.PNG">

### Generate samples

```
> python Generate_Mnist.py
```
 
<img align='center' img src="./GAN/InfoGan/images/InfoGan_Generate.PNG">

### Plot images

```
> python Plot.py
```

<img align='center' img src="./GAN/InfoGan/images/InfoGan_Plot.PNG">
