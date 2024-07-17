### Exercise 0: Python Boot Camp

Please open the google colab notebook by clicking on this [link](https://colab.research.google.com/github/dlmbl/boot/blob/main/exercise.ipynb).

You can open the solutions in colab by clicking [here](https://colab.research.google.com/github/dlmbl/boot/blob/main/solution.ipynb).


### Overview

```
In this notebook, we will go through some basic image processing in Python, come across standard tasks required while setting up deep learning pipelines, and familiarize ourselves with popular packages such as `glob`, `tifffile`, `tqdm`, `albumenations` and more.

We will learn about:
- Loading images (This is important as images are the primary input to most deep learning models)
- Normalizing images (This is important as it helps in faster convergence of models becuse it helps in reducing the scale of the input data and hence the scale of the gradients)
- Cropping images (This is important as it helps in creating smaller images from the original images which is useful for training models in a memory efficient way)
- Downsampling images (This is important as it helps in reducing the size of the images which is useful for training models in a memory efficient way)
- Flipping images (This is important as it helps in creating new images from the original data which is useful for training models in a memory efficient way)
- Batching images (As we train in a SGD manner, batching is important as it helps in training the model in a memory efficient way and smoothens the optimization process)
- Convolutions (This is important as it is the primary operation in Convolutional Neural Networks)
- Data Augmentation (This is important as it helps in artificially increasing the size of the training data which is useful for training models in a memory efficient way)

### Dataset
We will be using sample images from the *MoNuSeg* dataset provided by [Kumar et al, 2018](https://ieeexplore.ieee.org/document/8880654). The data was publicly made available [here](https://monuseg.grand-challenge.org/) by the authors of the publication.
This dataset shows Hematoxylin and Eosin (H&E) Stained Images showing nuclei in different shapes.
```