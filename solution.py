# %% [markdown]
# # Python Boot Camp
#
#
#
# Welcome! 😃👋
#
# In this notebook, we will go through a comprehensive Python boot camp focused on foundational computer vision concepts. You will learn about:

# * **Preparing and understanding image data**: How to programmatically download a dataset, inspect image properties like shape and data type, and preprocess them using `tifffile` and `numpy`.
# * **Data augmentation**: Implementing common image transformations such as flipping and rotation using both basic Python libraries like `numpy` and deep-learning frameworks like PyTorch's `torchvision`.
# * **Convolutions**: Understanding and implementing 2D convolutions, the foundational operation of Convolutional Neural Networks (CNNs), and seeing how different filters can extract specific features from an image.
# * **Efficient data loading**: Creating batches of data and building an efficient data loading pipeline using PyTorch's `Dataset` and `DataLoader` classes.
# * **Advanced image analysis**: Using libraries like `scikit-image` and `matplotlib` to perform analyses, such as visualizing cell size distributions and overlaying segmentation masks on original images.
#
# We will be using sample images from the *MoNuSeg* dataset provided by [Kumar et al, 2018](https://ieeexplore.ieee.org/document/8880654). The data was publicly made available [here](https://monuseg.grand-challenge.org/) by the authors of the publication. This dataset shows Hematoxylin and Eosin (H&E) Stained Images showing nuclei in different shapes.

# %% [markdown]
# ## Chapter 0: Downloading the data


# %% [markdown]
# Let us first download the images from an external url, which is a zip file containing the dataset.
# We will then extract the zip file to a folder named `monuseg-2018`.

# %%
from pathlib import Path
import urllib.request, zipfile


def extract_data(zip_url, project_name):
    zip_path = Path(project_name + ".zip")
    if zip_path.exists():
        print("Zip file was downloaded and extracted before!")
    else:
        urllib.request.urlretrieve(zip_url, zip_path)
        print("Downloaded data as {}".format(zip_path))
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall("./")
    print("Unzipped data to {}".format(Path(project_name)))
    zip_path.unlink()


extract_data(
    zip_url="https://owncloud.mpi-cbg.de/index.php/s/xwYonC9LucjLsY6/download",
    project_name="monuseg-2018",
)


# %% [markdown]
"""
<div class="alert alert-info">

### Task 0.1
Use the Explorer (left panel) and manually check the directory structure of the downloaded data. Where are the images and masks stored?  Can you programmatically count the number of images and masks?

*Hint*: you can run any bash command in a jupyter notebook by prefixing it with `!`. You might find the command `wc` (which stands for "word count") and the pipe operator `|` useful here.
"""


# %% tags=["task"]
##########################
######## To Do ###########
##########################

# %% tags=["solution"]
##########################
####### Solution #########
##########################

# !ls monuseg-2018/download/images | wc -l
# !ls monuseg-2018/download/masks | wc -l


# %% [markdown]
# ## Chapter 1: Understanding the data

# %% [markdown]
# ### Image Basic Concepts

# %% [markdown]
# In this chapter, we will explore the dataset and build some basic understanding of the data.
#
# 2D Images are often represented as numpy arrays of shape (`height`, `width`, `num_channels`).
# Let's first load the images and masks and visualize them.
#
# ![RGB image as a np array](https://github.com/dlmbl/boot/assets/34229641/ce1ad3f3-dc34-46d1-b301-198768fbc369)
#
# <div style="text-align: right"> Credit: <a href="https://e2eml.school/convert_rgb_to_grayscale.html">Brandon Rohrer’s Blog</a></div>

# %% [markdown]
"""
<div class="alert alert-info">

### Task 1.1
Load the images and masks using `tifffile.imread`. Define a `visualize` function using `matplotlib.pyplot.imshow` to display the image and mask side by side.
"""


# %% tags=["task"]
##########################
######## To Do ###########
##########################
from tifffile import imread
import matplotlib.pyplot as plt

img_path = "monuseg-2018/download/images/TCGA-2Z-A9J9-01A-01-TS1.tif"
mask_path = "monuseg-2018/download/masks/TCGA-2Z-A9J9-01A-01-TS1.tif"

img = ...  # TODO
mask = ...  # TODO


def visualize(im1, im2):
    plt.figure(figsize=(10, 10))
    plt.subplot(121)
    plt.imshow(...)  # TODO
    plt.subplot(122)
    plt.imshow(...)  # TODO
    plt.tight_layout()


visualize(img, mask)

# %% tags=["solution"]
##########################
######## Solution ###########
##########################
from tifffile import imread
import matplotlib.pyplot as plt

img_path = "monuseg-2018/download/images/TCGA-2Z-A9J9-01A-01-TS1.tif"
mask_path = "monuseg-2018/download/masks/TCGA-2Z-A9J9-01A-01-TS1.tif"

img = imread(img_path)
mask = imread(mask_path)


def visualize(im1, im2):
    plt.figure(figsize=(10, 10))
    plt.subplot(121)
    plt.imshow(im1)
    plt.subplot(122)
    plt.imshow(im2)
    plt.tight_layout()


visualize(img, mask)


# %% [markdown]
"""
<div class="alert alert-info">

### Task 1.2
How many channels does the image have? How about the mask?

*Hint*: <a href="https://assets.datacamp.com/blog_assets/Numpy_Python_Cheat_Sheet.pdf">np cheatsheet</a>
"""

# %% tags=["task"]
##########################
######## To Do ###########
##########################

# %% tags=["solution"]
##########################
######## Solution ###########
##########################

print(img.shape)
print(mask.shape)

# %% [markdown]
# Next, let's figure out the data type of the images and masks.
# Images can be represented by a variety of data types. It is important to understand the data type of images and what values they can take.
# For example, here are some common data types used in images:
# - `bool`: binary, 0 or 1
# - `uint8`: unsigned integers, 0 to 255 range
# - `int8`: signed integers, -128 to 127 range
# - `float32`: floating point numbers, 32-bit precision
#
# Here is a comprehensive [guide](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.number) on numpy data types.

# %% [markdown]
"""
<div class="alert alert-info">

### Task 1.3
What is the data type of <code>img</code> and the <code>mask</code> ? What are the minimum and maximum intensity values of <code>img</code> and <code>mask</code>?

*Hint*: <a href="https://assets.datacamp.com/blog_assets/Numpy_Python_Cheat_Sheet.pdf">np cheatsheet</a></div>
"""

# %% tags=["task"]
##########################
######## To Do ###########
##########################

# %% tags=["solution"]
##########################
####### Solution #########
##########################

print("data type: ", img.dtype, mask.dtype)
print("Image min and max: ", img.min(), img.max())
print("Mask min and max: ", mask.min(), mask.max())


# %% [markdown]
"""<div class="alert alert-info">

### Task 1.4
Let's dive deeper into the mask. What does the minimum and maximum value of the mask represent?
How many unique values does the mask have? Can you visualize the region where the values is 0 and where the value is 100?
"""

# %% tags=["task"]
##########################
######## To Do ###########
##########################
import numpy as np

unique_values = ...  # TODO
print(f"There are {len(unique_values)} Unique values in mask.")

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(...)  # TODO
plt.subplot(122)
plt.imshow(...)  # TODO
plt.tight_layout()

# %% tags=["solution"]
##########################
####### Solution #########
##########################
import numpy as np

unique_values = np.unique(mask)
print(f"There are {len(unique_values)} unique values in mask.")

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(mask == 0)
plt.subplot(122)
plt.imshow(mask == 100)
plt.tight_layout()


# %% [markdown]
# ### Transpose images

# %% [markdown]
# In deep learning, images are represented as (`num_channels`, `height`, `width`).
# But the image which we are working with has the `channel` as the last axis.
# Therefore, we need to reshape (by swapping) the image to the correct shape.

# To make it clearer and so that we don't swap height and width by mistake, we will first 
# crop a rectangular patch so we have different height and width and then we will 
# swap the axes to have the channel as the first axis.

# %% [markdown]
"""<div class="alert alert-info">

### Task 1.5
Crop a rectangle (e.g., 500x800) from the image and then transpose the image to have the channel as the first axis. Use `np.transpose` to achieve this.
"""
# %% tags=["task"]
##########################
######## To Do ###########
##########################
# Create a rectangular crop of the image of size 500x800
cropped_img = # TODO
print("Original image shape: ", cropped_img.shape)

# Transpose the image to have the channel as the first axis
reshaped_img = ...  # TODO
print("Reshaped image shape: ", reshaped_img.shape)

# %% tags=["solution"]
##########################
####### Solution #########
##########################
# Create a rectangular crop of the image of size 500x800
cropped_img = img[:500, :800, :]
print("Original image shape: ", cropped_img.shape)

# Transpose the image to have the channel as the first axis
reshaped_img = np.transpose(cropped_img, (2, 0, 1))
print("Reshaped image shape: ", reshaped_img.shape)

# %% [markdown]
"""
<div class="alert alert-success">

## Checkpoint 1

Great Job! 🎊 Please flag the sticky note when you reach this checkpoint.

In the first chapter, we learned about:

<li> image data type and shape </li>
<li> reshaping images </li>
<li> visualizing images </li>

</div>
"""


# %% [markdown]
# ## Chapter 2: Data Augmentation

# %% [markdown]
# Data augmentation is a technique used to artificially increase the size of a dataset by applying various transformations to the existing data. This is particularly useful in deep learning, where large datasets are often required to
# train models effectively. In this chapter, we will explore how to perform data augmentation using numpy and pytorch.
#
# Let's start with implementing some basic transformations, including flipping, rotation, cropping, and scaling.

# %% [markdown]
# ### Implementing transformations with numpy

# %% [markdown]
"""<div class="alert alert-info">

### Task 2.1
Flip the image horizontally and vertically using numpy functions.
"""


# %% tags=["task"]
##########################
######## To Do ###########
##########################
def flip_image(im):
    flipped_horizontally = ...  # TODO
    flipped_vertically = ...  # TODO
    return flipped_horizontally, flipped_vertically


flipped_horizontally, flipped_vertically = flip_image(img)
visualize(img, flipped_horizontally)
visualize(img, flipped_vertically)


# %% tags=["solution"]
###########################
####### Solution #########
###########################
def flip_image(im):
    flipped_horizontally = np.flip(im, axis=1)  # Flip horizontally
    flipped_vertically = np.flip(im, axis=0)  # Flip vertically
    return flipped_horizontally, flipped_vertically


flipped_horizontally, flipped_vertically = flip_image(img)
visualize(img, flipped_horizontally)
visualize(img, flipped_vertically)


# %% [markdown]
"""<div class="alert alert-info">

### Task 2.2
Implement a function to flip the image horizontally and vertically by directly manipulating the arrays without any additional functions.

Hint: you will find numpy [slicing and striding](https://numpy.org/doc/stable/user/basics.indexing.html#slicing-and-striding) useful here.
"""


# %% tags=["task"]
##########################
######## To Do ###########
##########################
def flip_image(im):
    flipped_horizontally = ...  # TODO
    flipped_vertically = ...  # TODO
    return flipped_horizontally, flipped_vertically


flipped_horizontally, flipped_vertically = flip_image(img)
visualize(img, flipped_horizontally)
visualize(img, flipped_vertically)


# %% tags=["solution"]
###########################
####### Solution #########
###########################
def flip_image(im):
    flipped_horizontally = im[::-1, :, :]  # Flip horizontally
    flipped_vertically = im[:, ::-1, :]  # Flip vertically
    return flipped_horizontally, flipped_vertically


flipped_horizontally, flipped_vertically = flip_image(img)
visualize(img, flipped_horizontally)
visualize(img, flipped_vertically)


# %% [markdown]
"""<div class="alert alert-info">

### Task 2.3
Rotate the image by 90 degrees clockwise and counter-clockwise using numpy.
"""


# %% tags=["task"]
##########################
######## To Do ###########
##########################
def rotate_image(im):
    rotated_clockwise = ...  # TODO
    rotated_counter_clockwise = ...  # TODO
    return rotated_clockwise, rotated_counter_clockwise


rotated_clockwise, rotated_counter_clockwise = rotate_image(img)
visualize(img, rotated_clockwise)
visualize(img, rotated_counter_clockwise)


# %% tags=["solution"]
##########################
####### Solution #########
##########################
def rotate_image(image):
    rotated_clockwise = np.rot90(image, k=-1)  # Rotate clockwise
    rotated_counter_clockwise = np.rot90(image, k=1)  # Rotate counter-clockwise
    return rotated_clockwise, rotated_counter_clockwise


rotated_clockwise, rotated_counter_clockwise = rotate_image(img)
visualize(img, rotated_clockwise)
visualize(img, rotated_counter_clockwise)

# %% [markdown]
"""<div class="alert alert-info">

### Task 2.4
Implement a function to crop out the top left quardrant of the image and rescale it to the original size using numpy.

Hint: `skimage.transform.resize` can be used to rescale the image.
"""


# %% tags=["task"]
##########################
######## To Do ###########
##########################
from skimage.transform import resize


def crop_and_rescale(im):
    height, width, _ = im.shape
    top_left = ...  # TODO

    # Rescale to original size
    top_left_rescaled = ...  # TODO
    return top_left_rescaled


top_left_rescaled = crop_and_rescale(img)
visualize(img, top_left_rescaled)


# %% tags=["solution"]
##########################
####### Solution #########
##########################
from skimage.transform import resize


def crop_and_rescale(im):
    height, width, _ = im.shape
    top_left = im[: height // 2, : width // 2, :]

    # Rescale to original size
    top_left_rescaled = resize(top_left, im.shape)
    return top_left_rescaled


top_left_rescaled = crop_and_rescale(img)
visualize(img, top_left_rescaled)


# %% [markdown]
# ### Implementing transformations with pytorch

# %% [markdown]
# Now that we have implemented basic transformations using numpy, let's explore how to do the same using PyTorch.

# %% [markdown]
# PyTorch provides a powerful library called `torchvision` that includes many built-in transformations for
# data augmentation. These transformations can be easily applied to images and are optimized for performance.
# The `transforms` module provides a wide range of transformations that can be applied to images.
# We can compose multiple transformations together using `transforms.Compose` and randomly apply them to the images on-the-fly during training.
# Here is an example of how to use `torchvision.transforms` to perform some of transformations as above.

# %% [markdown]
"""<div class="alert alert-info">

### Task 2.5
Let's compose a series of transformations using `transforms.Compose()` that includes:
- Converting the numpy array to a PIL image using `transforms.ToPILImage()` (required for many torchvision transforms)
- Randomly flip the image horizontally and vertically with a probability of 0.5
- Randomly rotate the image by 90 degrees
- Randomly crop the image to a size of 500x500
- Resize the image to a size of 1000x1000

"""


# %% tags=["task"]
##########################
######## To Do ###########
##########################
import torchvision.transforms as transforms

# Define a series of transformations
transform = transforms.Compose(
    ...  # TODO
)

transformed_img = transform(img)
visualize(img, transformed_img)

# %% tags=["solution"]
##########################
####### Solution #########
##########################
import torchvision.transforms as transforms

# Define a series of transformations
transform = transforms.Compose(
    [
        transforms.ToPILImage(),  # Convert numpy array to PIL Image
        transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip the image horizontally
        transforms.RandomVerticalFlip(p=0.5),  # Randomly flip the image vertically
        transforms.RandomRotation(degrees=90),  # Randomly rotate the image by 90
        transforms.RandomCrop(size=(500, 500)),  # Randomly crop the image to 500x500
        transforms.Resize(size=(1000, 1000)),  # Resize the image to 1000x1000
    ]
)

transformed_img = transform(img)
visualize(img, transformed_img)

# %% [markdown]
# ### Normalization

# %% [markdown]
# After applying the transformations, the images are often normalized before being fed into the model.
# Normalization is a technique used to scale the pixel values of an image to a specific range, typically [0, 1] or [-1, 1].
# This helps in stabilizing the training process and improving the convergence of the model.
#
# One way of normalizing an image is to divide the intensity on each pixel by the maximum allowed intensity for the available data type.

# %% [markdown]
"""<div class="alert alert-info">

### Task 2.6
Normalize the image by dividing each pixel value by the maximum allowed intensity for the data type. Does the data type of the image change? What are the minimum and maximum values of the normalized image?
"""
# %% tags=["task"]
##########################
######## To Do ###########
##########################
normalized_img = ...  # TODO
print("Normalized image: ", normalized_img.dtype)
print("Normalized image min: ", normalized_img.min(), "max: ", normalized_img.max())
# %% tags=["solution"]
##########################
####### Solution #########
##########################
normalized_img = img / 255.0
print("Normalized image: ", normalized_img.dtype)
print("Normalized image min: ", normalized_img.min(), "max: ", normalized_img.max())

# %% [markdown]
"""
<div class="alert alert-success">

## Checkpoint 2

Wow! 🤟 Flag the sticky note when you reach this checkpoint!

In the second chapter, we learnt about:

<li> data augmentation methods and implementation in numpy </li>
<li> pytorch transformations </li>
<li> how to normalize images </li>

"""

# %% [markdown]
# ## Chapter 3: Convolutions

# ### Implementing convolutions

# %% [markdown]
# Convolutions are the elementary operations used in Convolutional Neural Networks (CNNs). <br> The images are convolved with filters as below: <br>
#
# ![](https://raw.githubusercontent.com/vdumoulin/conv_arithmetic/master/gif/no_padding_no_strides.gif)
#
#
# Please read this section https://en.wikipedia.org/wiki/Kernel_(image_processing)#Convolution on convolutions to learn how to implement a your own convolution function!

# %% [markdown]
"""
<div class="alert alert-info">

### Task 3.1
Implement a function that performs a convolution of an image with a filter. 
<br> Assume that your image is square and that your filter is square and has an odd width and height.
<br> Also assume that stride is 1.
"""

# %% tags=["task"]
##########################
######## To Do ###########
##########################


def conv2d(img, kernel):
    # Ensure the kernel is square and has an odd size
    assert kernel.shape[0] == kernel.shape[1]
    assert kernel.shape[0] % 2 != 0

    h, w = img.shape[0], img.shape[1]  # Starting size of image
    d_k = kernel.shape[0]  # Size of kernel

    h_new = h - d_k + 1  # Calculate the new height of the array
    w_new = w - d_k + 1  # Calculate the new width of the array
    output = np.zeros((h_new, w_new))

    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            # Extract the curent patch or window from the image
            patch = ... # TODO

            # Element-wise multiplication between the patch and the kernel, then sum the result to get the convolved value at (i, j)
            output[i, j] = ...  # TODO
    return output


# %% tags=["solution"]
##########################
####### Solution #########
##########################


def conv2d(img, kernel):
    # Ensure the kernel is square and has an odd size
    assert kernel.shape[0] == kernel.shape[1]
    assert kernel.shape[0] % 2 != 0

    h, w = img.shape[0], img.shape[1]  # Starting size of image
    d_k = kernel.shape[0]  # Size of kernel

    h_new = h - d_k + 1  # Calculate the new height of the array
    w_new = w - d_k + 1  # Calculate the new width of the array
    output = np.zeros((h_new, w_new))

    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            # Extract the curent patch or window from the image
            patch = img[i : i + d_k, j : j + d_k]

            # Element-wise multiplication between the patch and the kernel, then sum the result to get the convolved value at (i, j)
            output[i, j] = np.sum(patch * kernel)
    return output


# %%
# Run this cell to check your function

# Identity Kernel
identity = np.array([[0, 0, 0], 
                     [0, 1, 0], 
                     [0, 0, 0]])
# Let's take a 256x256 center crop of the image for better visualization of the effect of the convolution
test_crop = img[128:384, 128:384, 0]  # Take the first channel for simplicity
new_im = conv2d(test_crop, identity)

# An identity kernel should produce an output that is a copy of the intput image (minus the edges during convolution)
print(f"Input shape: {test_crop.shape} -> Output shape: {new_im.shape}")


# Lets visualize the original image and the convolved image and the filter
plt.figure(figsize=(10, 10))
plt.subplot(131)
plt.imshow(test_crop)
plt.title("Original Image")
plt.subplot(132)
plt.imshow(identity)
plt.title("Kernel")
plt.subplot(133)
plt.imshow(new_im)
plt.title("Convolved Image")
plt.tight_layout()


# %% [markdown]
"""
<div class="alert alert-info">

### Task 3.2

We noticed that the output image is smaller than the input image! <br>

Given an input image of size $H \times W$, a filter of size $K_h \times K_w$ , and strides $S_h$ and $S_w$, 
can you come up with an analytical relationship regarding how much smaller the output image is compared to the input image?

Feel free to play with this [visualizer](https://ezyang.github.io/convolution-visualizer/index.html) to get an intuition (ignore "Padding" and "Dilation" for now)!

**Hint**: Look at your output from Task 3.1. Your image transformed from a shape of 256 to 254. Using teh Kernel Size ($K$) and Stride ($S$), can you figure out the mathematical relationship that produced this specific change?
"""
# %% tags=["task"]
##########################
######## To Do ###########
##########################

# %% [markdown] tags=["solution"]

# ```
# ##########################
# ####### Solution #########
# ##########################
# ```

# - Given an input image of size $H \times W$, a filter of size $K_h \times K_w$ , and strides $S_h$ and $S_w$
# the output size (height $H_{out}$ and width $W_{out}$) can be calculated using the following formulas (Note that $\lfloor.\rfloor$ is the floor operator):

# $$
# \begin{equation*}
#     H_{out} = \left\lfloor \frac{H - K_h}{S_h} \right\rfloor + 1
# \end{equation*}
# $$
# $$
# \begin{equation*}
#     W_{out} = \left\lfloor \frac{W - K_w}{S_w} \right\rfloor + 1
# \end{equation*}
# $$

# %% [markdown]
# ### Different types of kernels
#
# Let's explore how different kernels affect the output image.
# The following is known as the [Sobel filter](https://en.wikipedia.org/wiki/Sobel_operator):

# $$
# \begin{bmatrix}
#     1 & 2 & 1 \\
#     0 & 0 & 0 \\
#     -1 & -2 & -1
# \end{bmatrix}
# $$
#

# %% [markdown]
"""
<div class="alert alert-info">

### Task 3.3

Apply the Sobel filter and describe the output image. What features does it highlight? <br>
"""

# %% tags=["task"]
##########################
######## To Do ###########
##########################

filter = ...  # TODO
output_img = conv2d(img[128:384, 128:384, 0], filter)
visualize(img[128:384, 128:384, 0], output_img)

# %% tags=["solution"]
##########################
####### Solution #########
##########################

filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
output_img = conv2d(img[128:384, 128:384, 0], filter)
visualize(img[128:384, 128:384, 0], output_img)


# %% [markdown]
"""
<div class="alert alert-info">

### Task 3.4 (Bonus)

Try different [kernels](https://www.geeksforgeeks.org/deep-learning/types-of-convolution-kernels/#basic-convolution-kernels) and visualize the results. <br>
What features do they highlight? <br>
"""

# %% tags=["task"]
##########################
######## To Do ###########
##########################


filter = ...  # TODO
output_img = conv2d(img[128:384, 128:384, 0], filter)
visualize(img[128:384, 128:384, 0], output_img)


# %% tags=["solution"]
##########################
####### Solution #########
##########################

# Up to you to try different filters!
filter = np.array(
    [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
)  # For example, Laplacian filter
output_img = conv2d(img[128:384, 128:384, 0], filter)
visualize(img[128:384, 128:384, 0], output_img)


# %% [markdown]
# Convolutions are a powerful tool for extracting features from images. They are the basic building blocks of Convolutional Neural Networks (CNNs), which are widely used in computer vision tasks.
# In these exercises, we defined the kernels ourself, but in practice, the kernels are learned during the training process of the CNN.


# %% [markdown]
"""
<div class="alert alert-success">

Good job! 🤟 Flag the sticky note when you reach this checkpoint!

## Checkpoint 3

In the third chapter, we learnt about:

<li> Convolutions and its implementation </li>
<li> Different types of kernels </li>

"""


# %% [markdown]
# ## Chapter 4: Batching
# ### Loading data and Sampling a batch

# %% [markdown]
# In this chapter, we will learn how to create batches of images and masks.
# Batching is a technique used to group multiple samples together to speed up the training process and make better use of the GPU memory.

# %% [markdown]
# We will use the `glob` module to load all the images and masks from the `monuseg-2018/download/images` and `monuseg-2018/download/masks` directories.
# `glob` is a module that allows us to search for files and directories matching a specified pattern.

# %% [markdown]
"""<div class="alert alert-info">

### Task 4.1
Load all the images and masks using `glob` and `tifffile.imread`.
"""
# %% tags=["task"]
##########################
######## To Do ###########
##########################
from glob import glob
import tifffile as tiff

image_paths = ...  # TODO
mask_paths = ...  # TODO
images = [tiff.imread(path) for path in image_paths]
masks = [tiff.imread(path) for path in mask_paths]
print(f"Loaded {len(images)} images and {len(masks)} masks.")
# %% tags=["solution"]
##########################
####### Solution #########
##########################
from glob import glob
import tifffile as tiff

image_paths = glob("monuseg-2018/download/images/*.tif")
mask_paths = glob("monuseg-2018/download/masks/*.tif")
images = [tiff.imread(path) for path in image_paths]
masks = [tiff.imread(path) for path in mask_paths]
print(f"Loaded {len(images)} images and {len(masks)} masks.")
# %% [markdown]
# Now that we have loaded the images and masks, we can create a mini-batch of images and masks.
# A mini-batch is a small subset of the dataset that is used to train the model in one iteration.
# %% [markdown]
"""<div class="alert alert-info">

### Task 4.2
We want to create a batch, a small group of images randomly chosen from the whole dataset. We need to pick the same images and masks to create the batch.
You have to choose a set of random numbers that represent the position of the images in our list and then use those numbers to reach the images/masks and create the batch.
"""
# %% tags=["task"]
##########################
######## To Do ###########
##########################
import random

batch_size = 5
indices = ...  # TODO
batch_images = ...  # TODO
batch_masks = ...  # TODO
print(
    f"Created a mini-batch of {len(batch_images)} images and {len(batch_masks)} masks."
)
# %% tags=["solution"]
##########################
####### Solution #########
##########################
import random

batch_size = 5
indices = random.sample(range(len(images)), batch_size)
batch_images = [images[i] for i in indices]
batch_masks = [masks[i] for i in indices]
print(
    f"Created a mini-batch of {len(batch_images)} images and {len(batch_masks)} masks."
)

# %% [markdown]
# ### Pytorch dataset and dataloader

# %% [markdown]
# We can also create a custom dataset using PyTorch's `Dataset` class.
# A custom dataset can be created by subclassing the `torch.utils.data.Dataset` class and implementing the `__init__`, `__len__` and `__getitem__` methods.
# The `__init__` method is an initialization procedure. The `__len__` method returns the number of samples in the dataset, and the `__getitem__` method returns a sample from the dataset at a given index.

# %% [markdown]
"""<div class="alert alert-info">

### Task 4.3
Create a custom dataset using PyTorch's `Dataset` class that loads the images and masks.
"""
# %% tags=["task"]
##########################
######## To Do ###########
##########################
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self): ...  # TODO

    def __getitem__(self, idx): ...  # TODO


my_dataset = MyDataset(images, masks)
print(len(my_dataset))
print(my_dataset[0])

# %% tags=["solution"]
##########################
####### Solution #########
##########################
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        return image, mask


my_dataset = MyDataset(images, masks)
print(len(my_dataset))
print(my_dataset[0])

# %% [markdown]
# Now that we have created a custom dataset, we can use it to create batches of images and masks.
# We can use the `DataLoader` class from PyTorch to create batches of images and masks.
# The `DataLoader` class takes a dataset as input and provides an iterable over the dataset, allowing us to easily load the images and masks in batches during training.

# %% [markdown]
"""<div class="alert alert-info">

### Task 4.4
Create a `DataLoader` to load the dataset in batches.
"""
# %% tags=["task"]
##########################
######## To Do ###########
##########################
from torch.utils.data import DataLoader

batch_size = 5
data_loader = ...  # TODO
for batch_images, batch_masks in data_loader:
    print(f"Loaded a batch of {len(batch_images)} images and {len(batch_masks)} masks.")
    break  # Just to check the first batch
# %% tags=["solution"]
##########################
####### Solution #########
##########################
from torch.utils.data import DataLoader

batch_size = 5
data_loader = DataLoader(my_dataset, batch_size=batch_size)
for batch_images, batch_masks in data_loader:
    print(f"Loaded a batch of {len(batch_images)} images and {len(batch_masks)} masks.")
    break  # Just to check the first batch
# %% [markdown]
"""
<div class="alert alert-success">

Nice! 🤟 Flag the sticky note when you reach this checkpoint!

## Checkpoint 4

In the fourth chapter, we learnt about:

<li> Using `glob` to find all data that matched certain pattern </li>
<li> Batching data </li>
<li> Using pytorch dataset and dataloader </li>

"""


# %% [markdown]
# ## Chapter 5: Advanced Analysis (Bonus)

# %% [markdown]
# This chapter focuses on a more advanced analysis of the image data and masks.
# We will begin by analyzing cell sizes to visualize their distribution and then create an overlay of the masks on the original images.
# These analyses are crucial for gaining a better understanding of the dataset and closely examining the quality of our segmentation results.

# %% [markdown]
"""<div class="alert alert-info">

### Task 5.1 (Bonus)
Let's find the sizes of the cells in the image and visualize the distribution.

Hint: `skimage.measure.regionprops` can be useful here.
"""

# %% tags=["task"]
##########################
######## To Do ###########
##########################
from skimage import measure

def analyze_area(mask):
    regions = ... # TODO
    areas = ...  # TODO
    plt.hist(areas, bins=50)
    plt.xlabel("Size")
    plt.ylabel("Frequency")
    plt.title("Histogram of Cell Sizes")
    plt.show()


analyze_area(mask)

# %% tags=["solution"]
##########################
####### Solution #########
##########################
from skimage import measure

def analyze_area(mask):
    regions = measure.regionprops(mask)
    areas = [region.area for region in regions]
    plt.hist(areas, bins=50)
    plt.xlabel("Size")
    plt.ylabel("Frequency")
    plt.title("Histogram of Cell Sizes")
    plt.show()


analyze_area(mask)


# %% [markdown]
"""<div class="alert alert-info">

### Task 5.2 (Bonus)

Let's overlay the masks' boundaries on the images to visualize the results.

Hint: `skimage.segmentation.find_boundaries` can be useful here.
"""
# %% tags=["task"]
##########################
######## To Do ###########
##########################
from skimage.segmentation import mark_boundaries


def overlay_masks_on_images(im, mask):
    plt.figure(figsize=(10, 10))
    combined_im = ...  # TODO
    plt.imshow(combined_im)


overlay_masks_on_images(img[:250, :250], mask[:250, :250])


# %% tags=["solution"]
##########################
####### Solution #########
##########################
from skimage.segmentation import mark_boundaries


def overlay_masks_on_images(im, mask):
    plt.figure(figsize=(10, 10))
    combined_im = mark_boundaries(im, mask)
    plt.imshow(combined_im)


overlay_masks_on_images(img[:250, :250], mask[:250, :250])


# %% [markdown]
"""
<div class="alert alert-success">

Hurrah! 😃 Post in the chat when you reach this checkpoint! 

## Checkpoint 5 (Bonus)

In this chapter, we learned about:

<li> analyzing the size of cells in the images </li>
<li> visualizing the masks on top of the images </li>
"""
