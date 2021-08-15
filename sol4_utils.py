# This file contains the solutions for exercise 4.
# COURSE: Image Processing 2020.
# AUTHORS: Jason Elter and Image Processing staff.
# FILE NAME: sol4_utils.py

import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage.filters import convolve as filter_convolve
from scipy.signal import convolve as signal_convolve
from skimage.color import rgb2gray
from imageio import imread

RGB_DIM = 3
GRAY_OUT = 1
TO_FRACTION = 255
FIRST_FILTER = np.array([1.0, 2.0, 1.0], dtype=np.float64)


# ------------------------- HELPER FUNCTIONS -------------------------
# Helper function that returns a 1d Gaussian filter of the given size.
def __get_filter(filter_size):
    if filter_size == 1:
        return np.array([1], dtype=np.float64).reshape((1, 1))

    result = FIRST_FILTER
    # Only odd numbers so we can calculate each 2 steps together.
    for _ in range((filter_size // 2) - 1):
        result = signal_convolve(result, FIRST_FILTER)

    # Normalize
    return (result / np.sum(result)).reshape((1, filter_size))


# Reduce image by half on each axis.
def __reduce(im, filter_2d):
    return filter_convolve(im, filter_2d)[::2, ::2]


# Generator for making gaussian pyramid.
def __gaussian_generator(im, max_levels, filter_2d):
    yield im

    i = 1
    current_im = im
    height = im.shape[0] / 2
    length = im.shape[1] / 2
    while i < max_levels and length > 16 and height > 16:
        current_im = __reduce(current_im, filter_2d)
        yield current_im
        height = current_im.shape[0] / 2
        length = current_im.shape[1] / 2
        i += 1


# -------------------------- UTIL FUNCTIONS --------------------------
def gaussian_kernel(kernel_size):
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = convolve2d(kernel, conv_kernel, 'full')
    return kernel / kernel.sum()


def blur_spatial(img, kernel_size):
    kernel = gaussian_kernel(kernel_size)
    blur_img = np.zeros_like(img)
    if len(img.shape) == 2:
        blur_img = convolve2d(img, kernel, 'same', 'symm')
    else:
        for i in range(3):
            blur_img[..., i] = convolve2d(img[..., i], kernel, 'same', 'symm')
    return blur_img


def read_image(filename, representation):
    """This function reads and returns an image represented by a matrix of
    type np.float64 normalized to range [0, 1].
    :param filename: The filename of the image to read.
    :param representation: If equals 1 -> grayscale output. Else if equals 2 -> RGB output.
    :return: Matrix representing the image.
    """
    image = imread(filename)

    # Convert image to grayscale, if required.
    if representation == GRAY_OUT and image.ndim == RGB_DIM:
        image = rgb2gray(image)
    else:
        # Normalize image to [0, 1].
        image = image.astype(np.float64)
        image /= TO_FRACTION
    return image


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    Constructs a Gaussian pyramid and returns it and the row vector representing the filter used.

    :param im: A grayscale image with double values in [0,1].
    :param max_levels: The maximal number of levels in the resulting pyramid.
    :param filter_size: The size of the Gaussian filter (odd scalar) to construct from.
    :return: pyr, filter_vec (pyr is the resulting pyramid, filter_vec is the row vector filter used).
    """
    filter_vec = __get_filter(filter_size)
    filter_2d = convolve2d(filter_vec, filter_vec.T)
    return list(__gaussian_generator(im, max_levels, filter_2d)), filter_vec
