# This file contains the solutions for exercise 1.
# COURSE: Image Processing 2020.
# AUTHOR: Jason Elter.
# FILE NAME: sol1.py

import numpy as np
from skimage.color import rgb2gray
from imageio import imread
import matplotlib.pyplot as plt

RGB_DIM = 3
GRAY_OUT = 1
MAX_VALUE = TO_FRACTION = 255
RGB_TO_YIQ_MATRIX = np.array([[0.299, 0.587, 0.114],
                              [0.596, -0.275, -0.321],
                              [0.212, -0.523, 0.311]])
YIQ_TO_RGB_MATRIX = np.linalg.inv(RGB_TO_YIQ_MATRIX)
Y_CHANNEL = IMAGE_LOCATION = MIN_VALUE = HISTOGRAM_LOCATION = 0


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


def imdisplay(filename, representation):
    """This function displays an image in a given representation.

    :param filename: The filename of the image to read.
    :param representation: If equals 1 -> grayscale output. Else if equals 2 -> RGB output.
    """
    image = read_image(filename, representation)

    if representation == GRAY_OUT:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)

    plt.show()


def rgb2yiq(imRGB):
    """Transforms a given RGB image into the YIQ color space.

       :param imRGB: Matrix representing the image in RGB (np.float64).

       :return: Matrix representing the image in YIQ.
   """
    return __image_color_conversion(imRGB, RGB_TO_YIQ_MATRIX)


def yiq2rgb(imYIQ):
    """Transforms a given YIQ image into the RGB color space.

    :param imYIQ: Matrix representing the image in YIQ (with float64 values).

    :return: Matrix representing the image in RGB.
    """
    return __image_color_conversion(imYIQ, YIQ_TO_RGB_MATRIX)


# Helper function for converting an image from one color space to another.
def __image_color_conversion(image, conversion_matrix):
    converted_image = np.empty(image.shape)
    for i in range(RGB_DIM):
        converted_image[:, :, i] = image.dot(conversion_matrix[i])
    return converted_image


def histogram_equalize(im_orig):
    """Preforms histogram equalization on the given image(grayscale or RGB).

    :param im_orig: Image with values in [0, 1] (grayscale or RGB with float64 values).

    :return: A list [im_eq, hist_orig, hist_eq] where:
            im_eq - the equalized image.
            hist_orig - a 256 bin histogram of the original image.
            hist_eq - a 256 bin histogram of the equalized image.
    """
    if im_orig.ndim == RGB_DIM:
        im_yiq = rgb2yiq(im_orig)  # Convert to YIQ space.

        # Histogram equalize only Y channel.
        result = __gray_histogram_equalize(im_yiq[:, :, Y_CHANNEL])
        im_yiq[:, :, Y_CHANNEL] = result[IMAGE_LOCATION]

        # Convert back to RGB.
        im_rgb = yiq2rgb(im_yiq)
        im_rgb_min = np.min(im_rgb)
        im_rgb = (im_rgb - im_rgb_min) / (np.max(im_rgb) - im_rgb_min)  # Normalize.
        result[IMAGE_LOCATION] = im_rgb
        return result
    # Otherwise, just histogram equalize.
    return __gray_histogram_equalize(im_orig)


# Helper function that preforms histogram equalization on the given grayscale image.
def __gray_histogram_equalize(im_gray):
    # Create histogram and cumulative histogram.
    hist_orig = np.histogram(im_gray, bins=256)[HISTOGRAM_LOCATION]
    cumulative_hist = np.cumsum(hist_orig, dtype=np.float64)

    # Normalize.
    num_pixels = cumulative_hist[MAX_VALUE]
    cumulative_hist = cumulative_hist * (MAX_VALUE / num_pixels)

    # Linear stretch, round and turn to int
    look_up_table = np.round(cumulative_hist)
    if look_up_table[MIN_VALUE] != MIN_VALUE or look_up_table[MAX_VALUE] != MAX_VALUE:
        min_non_zero = cumulative_hist[np.nonzero(look_up_table)[MIN_VALUE][MIN_VALUE]]
        num_pixels = cumulative_hist[MAX_VALUE]
        look_up_table = np.round((cumulative_hist - min_non_zero) / (num_pixels - min_non_zero) * MAX_VALUE)
    look_up_table = look_up_table.astype(np.int32)

    # Apply table to image and return result.
    eq_image = look_up_table[np.round(im_gray * MAX_VALUE).astype(np.int32)] / MAX_VALUE
    return [eq_image, hist_orig, np.histogram(eq_image, bins=256)[HISTOGRAM_LOCATION]]


def quantize(im_orig, n_quant, n_iter):
    """Preforms optimal quantization of a given image (grayscale or RGB).

    :param im_orig: The image to be quantized (grayscale or RGB with float64 values in range [0, 1]).
    :param n_quant: The number of intensities to quantize the image to.
    :param n_iter: The maximum number of iterations of the optimization procedure.

    :return: The list [im_quant, error] where:
            im_quant - The new quantized image.
            error - An array with shape (n_iter,) (or less) of the total intensities error
                    of each iteration of the quantization procedure.
    """
    if im_orig.ndim == RGB_DIM:
        # Quantize only Y channel.
        im_yiq = rgb2yiq(im_orig)
        result = __gray_quantize(im_yiq[:, :, Y_CHANNEL], n_quant, n_iter)
        im_yiq[:, :, Y_CHANNEL] = result[IMAGE_LOCATION]

        # Convert back to RGB space.
        result[IMAGE_LOCATION] = yiq2rgb(im_yiq)
        return result
    # Otherwise, just quantize.
    return __gray_quantize(im_orig, n_quant, n_iter)


# Helper function that preforms optimal quantization of the given grayscale image.
def __gray_quantize(im_gray, n_quant, n_iter):
    # Create histogram and cumulative histogram.
    hist = (np.histogram(im_gray, bins=256, range=(0, 1))[HISTOGRAM_LOCATION]).astype(np.float64)
    cum_hist = np.cumsum(hist)
    num_pixels = cum_hist[-1]

    # Initialize z, q and error arrays.
    q_hist = np.empty(256)
    z = np.empty(n_quant + 1).astype(np.int32).astype(np.int32)
    q = np.empty(n_quant)
    error = np.zeros(n_iter)

    # Initial boundaries.
    z[0] = 0
    z[n_quant] = 256
    # Create segments by roughly equal sized pixel amount.
    for i in range(n_quant - 1):
        z[i + 1] = (np.where(cum_hist <= ((i + 1) * num_pixels / n_quant))[0][-1])
    __calc_q(n_quant, hist, z, q)

    # Computes better q and z.
    for iteration in range(n_iter):
        temp_z = __calc_z(n_quant, q)
        if np.array_equal(temp_z, z):
            error = error[:iteration]
            break
        z = temp_z
        __calc_q(n_quant, hist, z, q)
        __calc_quantized_histogram(n_quant, z, q, q_hist)
        error[iteration] = __calc_quantization_error(hist, q_hist)

    q_hist = np.round(q_hist * MAX_VALUE).astype(np.int32)
    im_quant = q_hist[np.round(im_gray * MAX_VALUE).astype(np.int32)] / MAX_VALUE

    return [im_quant, error]


# Calculates q according to z.
def __calc_q(n_quant, hist, z, q):
    for i in range(n_quant):
        hist_segment = hist[z[i]: z[i + 1]]
        q[i] = np.dot(hist_segment, np.arange(z[i], z[i + 1]) / MAX_VALUE) / np.sum(hist_segment)


# Returns newly calculated z's according to q.
def __calc_z(n_quant, q):
    z = np.empty(n_quant + 1).astype(np.int32)
    z[0] = 0
    z[n_quant] = 256
    for i in range(n_quant - 1):
        z[i + 1] = np.round(MAX_VALUE * (q[i] + q[i + 1]) / 2)
    return z


# Applies new quantization to histogram.
def __calc_quantized_histogram(n_quant, z, q, q_hist):
    for i in range(n_quant):
        q_hist[z[i]: z[i + 1]] = q[i]


# Returns the quantization error.
def __calc_quantization_error(hist, q_hist):
    return np.sum(np.dot(hist, (np.round(q_hist - np.arange(0, 256)) ** 2)))

# Toy example
# x = np.hstack([np.repeat(np.arange(0, 50, 2), 10)[None, :], np.array([255] * 6)[None, :]])
# grad = np.tile(x, (256, 1)).astype(np.float64) / 255
