"""
Functions to transform the ultrasound from raw reflection data to real world proportions.

Date: Dec 2017
Author: Aciel Eshky

"""

import math
import sys

import numpy as np
from scipy import ndimage

from ustools.read_core_files import parse_parameter_file, read_ultrasound_file
from ustools.reshape_ultrasound import reshape_ultrasound_array
from ustools.visualise_ultrasound import display_2d_ultrasound_frame


def pol2cart(r, th):
    x = r * math.cos(th)
    y = r * math.sin(th)
    return x, y


def cart2pol(x, y):
    r = math.sqrt(x**2 + y**2)
    th = math.atan2(y, x)
    return r, th


def ultrasound_cart2pol(output_coordinates,
                        origin=(0, 0),
                        num_of_vectors=63,
                        angle=0.038,
                        zero_offset=50,
                        pixels_per_mm=1):
    """
    A function to transform ultrasound from cartesian to polar coordiates.

    :param output_coordinates:
    :param origin:
    :param num_of_vectors:
    :param angle:
    :param zero_offset:
    :param pixels_per_mm: controls the resolution.
    :return:
    """
    (r, th) = cart2pol(output_coordinates[0] - origin[0],
                       output_coordinates[1] - origin[1])

    r *= pixels_per_mm
    cl = num_of_vectors // 2

    return cl - ((th - np.pi / 2) / angle), r - zero_offset


def transform_raw_ult_to_world(raw_ult_frame,
                               spline_interpolation_order=2,
                               background_colour=255,
                               num_of_vectors=63,
                               size_of_vectors=412,
                               angle=0.038,
                               zero_offset=50,
                               pixels_per_mm=1
                               ):
    """
    Transform the raw ultrasound to real world proportions.

    :param raw_ult_frame:
    :param spline_interpolation_order:
    :param background_colour:
    :param num_of_vectors:
    :param size_of_vectors:
    :param angle:
    :param zero_offset:
    :param pixels_per_mm:
    :return:
    """
    if pixels_per_mm <= 0:
        pixels_per_mm = 1
        print("division by zero not allowed: pixels_per_mm set to 1.")

    height = math.sqrt(math.pow(size_of_vectors, 2) + math.pow(num_of_vectors, 2)) + zero_offset
    width = height * 2

    output_shape = (860 // pixels_per_mm, 480 // pixels_per_mm)
    output_shape = (int(width // pixels_per_mm),  # 63 -> 860 round(num_of_vectors * 13.65)
                    int(height // pixels_per_mm))  # 412 -> 480 round(num_of_vectors * 7.65)
    origin = (int(output_shape[0] // 2), 0)

    world_ult_frame = ndimage.geometric_transform(
        raw_ult_frame,
        mapping=ultrasound_cart2pol,
        output_shape=output_shape,
        order=spline_interpolation_order,
        cval=background_colour,
        extra_keywords={
            'origin': origin,
            'num_of_vectors': num_of_vectors,
            'angle': angle,
            'zero_offset': zero_offset,
            'pixels_per_mm': pixels_per_mm})

    return world_ult_frame


def transform_raw_ult_to_world_multi_frames(ult_3d,
                                            spline_interpolation_order=2,
                                            background_colour=255,
                                            num_of_vectors=63,
                                            size_of_vectors=412,
                                            angle=0.038,
                                            zero_offset=50,
                                            pixels_per_mm=1
                                            ):
    """

    :param ult_3d:
    :param spline_interpolation_order:
    :param background_colour:
    :param num_of_vectors:
    :param size_of_vectors:
    :param angle:
    :param zero_offset:
    :param pixels_per_mm:
    :return:
    """

    trans_ult = []

    # loop to transform each frame separately
    for ult in ult_3d:
        trans_ult.append(transform_raw_ult_to_world(ult,
                                                    spline_interpolation_order=spline_interpolation_order,
                                                    background_colour=background_colour,
                                                    num_of_vectors=num_of_vectors,
                                                    size_of_vectors=size_of_vectors,
                                                    angle=angle,
                                                    zero_offset=zero_offset,
                                                    pixels_per_mm=pixels_per_mm))

    return np.array(trans_ult)

