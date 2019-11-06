"""
Functions to transform the ultrasound from raw reflection data to real world proportions.

Date: Dec 2017
Author: Aciel Eshky

"""

import math

import numpy as np
from scipy import ndimage


def cart2pol_vectorised(x, y):
    """
    A vectorised version of the cartesian to polar conversion.

    :param x:
    :param y:
    :return:
    """
    r = np.sqrt(np.add(np.power(x, 2), np.power(y, 2)))
    th = np.arctan2(y, x)
    return r, th


def get_cart2pol_coordinates_vectorised(output_coordinates, origin=(0, 0), num_scanlines=63, angle=0.038,
                                        zero_offset=50, pixels_per_mm=1):
    """
    A function to get the polar to cartesian coordinates.

    :param output_coordinates:
    :param origin:
    :param num_scanlines:
    :param angle:
    :param zero_offset:
    :param pixels_per_mm:
    :return:
    """

    # shift by the origin
    (r, th) = cart2pol_vectorised(np.subtract(output_coordinates[0], origin[0]),
                                  np.subtract(output_coordinates[1], origin[1]))

    r = np.multiply(r, pixels_per_mm)
    cl = np.floor(np.divide(num_scanlines, 2))

    return np.subtract(cl, np.divide(np.subtract(th, np.divide(np.pi, 2)), angle)), np.subtract(r, zero_offset)


def transform_ultrasound(ult, spline_interpolation_order=2, background_colour=255, num_scanlines=63, size_scanline=412,
                         angle=0.038, zero_offset=50, pixels_per_mm=1):
    """
    A function to transform ultrasound from raw to world. Can be applied to an utterance (seuqnece of ultrasound
    frames) or a single ultrasound frame.

    :param ult: ultrasound data. 1d, 2d, and 3d shapes all accepted.
    :param spline_interpolation_order:
    :param background_colour:
    :param num_scanlines:
    :param size_scanline:
    :param angle:
    :param zero_offset:
    :param pixels_per_mm: number to divide resolution by

    :return: 3 dimensional ultrasound. if one frame was pased, the first dimension is 1.
    """

    if pixels_per_mm == 0:
        pixels_per_mm = 1
        print("Zero value provided for resolution_multiplier. Value set to 1.")

    if angle == 0:
        angle = 0.038
        print("Zero value provided for angle. Value set to 0.038.")

    # ideal output size for ultrasuite data is (884, 488)
    width = math.sqrt(math.pow(num_scanlines, 2) + math.pow(size_scanline, 2)) * 2 + zero_offset
    height = size_scanline + zero_offset * 1.5

    # reducing resolution using pixel per mm
    output_shape = (int(width // pixels_per_mm),
                    int(height // pixels_per_mm))

    origin = (int(output_shape[0] // 2), 0)

    xx, yy = np.meshgrid(np.arange(output_shape[0]), np.arange(output_shape[1]))
    coordinates_in_input = get_cart2pol_coordinates_vectorised((xx, yy), origin=origin, num_scanlines=num_scanlines,
                                                               angle=angle, zero_offset=zero_offset,
                                                               pixels_per_mm=pixels_per_mm)
    transformed_ult = []  # output

    if len(ult.shape) == 1:  # raw ultrasound has not yet been reshaped -> reshape it.

        ult = ult.reshape(-1, num_scanlines, size_scanline)

    if len(ult.shape) == 2:

        assert (ult.shape[0] == num_scanlines and ult.shape[1] == size_scanline)

        transformed_ult = np.zeros((1, output_shape[0], output_shape[1]))

        transformed_ult[0] = ndimage.map_coordinates(ult, coordinates_in_input, order=spline_interpolation_order,
                                                     cval=background_colour).transpose()

    elif len(ult.shape) == 3:

        assert (ult.shape[1] == num_scanlines and ult.shape[2] == size_scanline)

        transformed_ult = np.zeros((ult.shape[0], output_shape[0], output_shape[1]))

        for i, frame in enumerate(ult):
            transformed_ult[i] = ndimage.map_coordinates(frame, coordinates_in_input, order=spline_interpolation_order,
                                                         cval=background_colour).transpose()

    return transformed_ult
