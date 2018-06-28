"""
Functions to visualise ultrasound data

Date: Dec 2017
Author: Aciel Eshky

"""

import matplotlib.pyplot as plt


def display_2d_ultrasound_frame(ult_frame, dpi=300, figsize=(10, 10), aspect="equal", interpolation=None,
                                title="A single ultrasound frame"):
    """
    A function to plot a single ultrasound frame (either raw or transformed).

    :param ult_frame: the ultrasound frame
    :param dpi: can be set to None to get imshow's default behaviour
    :param figsize: can be set to None to get imshow's default behaviour
    :param aspect:
    :param interpolation:
    :param title:
    :return:
    """

    plt.figure(figsize=figsize, dpi=dpi)
    plt.title(title)
    plt.imshow(ult_frame.T, aspect=aspect, origin='lower', cmap='gray', interpolation=interpolation)
    plt.show()
