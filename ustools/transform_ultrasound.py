"""
Functions to transform the ultrasound from raw reflection data to real world proportions.

Date: Dec 2017
Author: Aciel Eshky

"""

import math
import numpy as np
from scipy import ndimage


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
    :param angle:
    :param zero_offset:
    :param pixels_per_mm:
    :return:
    """
    if pixels_per_mm <= 0:
        pixels_per_mm = 1
        print("division by zero not allowed: pixels_per_mm set to 1.")
    output_shape = (int(860 // pixels_per_mm), int(480 // pixels_per_mm))
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
                                            angle=0.038,
                                            zero_offset=50,
                                            pixels_per_mm=1
                                            ):
    """

    :param ult_3d:
    :param spline_interpolation_order:
    :param background_colour:
    :param num_of_vectors:
    :param angle:
    :param zero_offset:
    :param pixels_per_mm:
    :return:
    """

    # transform the first frame to get the dimensions
    x = transform_raw_ult_to_world(ult_3d[0],
                                   spline_interpolation_order=spline_interpolation_order,
                                   background_colour=background_colour,
                                   num_of_vectors=num_of_vectors,
                                   angle=angle,
                                   zero_offset=zero_offset,
                                   pixels_per_mm=pixels_per_mm)

    # create an empty* numpy array (*contains nans)
    trans_ult = np.full([ult_3d.shape[0], x.shape[0], x.shape[1]], np.nan)

    # loop to transform each frame separately
    for i in range(0, ult_3d.shape[0]):
        trans_ult[i] = transform_raw_ult_to_world(ult_3d[i],
                                                  spline_interpolation_order=spline_interpolation_order,
                                                  background_colour=background_colour,
                                                  num_of_vectors=num_of_vectors,
                                                  angle=angle,
                                                  zero_offset=zero_offset,
                                                  pixels_per_mm=pixels_per_mm)
    return trans_ult
