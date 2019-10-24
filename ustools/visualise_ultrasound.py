"""
Functions to visualise ultrasound data

Date: Dec 2017
Author: Aciel Eshky

"""

import matplotlib.pyplot as plt


def display_2d_ultrasound_frame(ult_frame, dpi=72, figsize=(5, 5), aspect="equal", interpolation=None,
                                title="", output_file=None):
    """
    A function to plot a single ultrasound frame (either raw or transformed).

    :param ult_frame: the ultrasound frame
    :param dpi: can be set to None to get imshow's default behaviour
    :param figsize: can be set to None to get imshow's default behaviour
    :param aspect:
    :param interpolation:
    :param title:
    :param output_file:
    :return:
    """

    plt.figure(figsize=figsize, dpi=dpi)
    plt.title(title)
    plt.imshow(ult_frame.T, aspect=aspect, origin='lower', cmap='gray', interpolation=interpolation)

    if output_file:
        plt.savefig(output_file)

    plt.show()
