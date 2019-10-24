"""

Date: Mar 2018
Author: Aciel Eshky

"""
import numpy as np


def get_segment(signal, sampling_rate=22050, start_time=0, end_time=None):
    """
    A function to get part of a signal, where start and end are specified in time.
    :param signal:
    :param sampling_rate:
    :param start_time: in seconds
    :param end_time: in seconds
    :return: The signal after the start and/or end have been trimmed.
    """
    start_frame = sampling_rate * start_time
    if not end_time:
        return signal[int(round(start_frame)):]
    end_frame = sampling_rate * end_time
    return signal[int(round(start_frame)):int(round(end_frame))]


def window_signal(signal, sampling_rate=22050, start_time=0, end_time=None, time_window=0.2):
    """
    A function to get windows of a signal where start, end, and window are specified as time.
    :param signal:
    :param sampling_rate:
    :param start_time: in seconds
    :param end_time: in seconds
    :param time_window: in seconds
    :return: The windowed signal as a list. Windows can vary in size.
    """
    a = []
    if not end_time:
        end_time = len(signal) / sampling_rate

    for value in np.arange(start_time, end_time - time_window, time_window):
        a.append(get_segment(signal=signal, sampling_rate=sampling_rate, start_time=value, end_time=value+time_window))

    return a


def get_zero_regions(signal, num_repetitions=2):
    """
    A function to get the zwero regions of a signal. This is useful for selecting the regions that were zero-ed during
    anonymisation.

    :param signal: 
    :param num_repetitions:
    :return: 
    """
    indices = []
    i = 0

    while i < len(signal) - 1:

        start_index = i

        while i < len(signal) - 1 and signal[i] == signal[i + 1] and signal[i] == 0:
            i = i + 1

        end_index = i

        if (start_index != end_index) & (end_index - start_index >= num_repetitions - 1):
            indices.append((start_index, end_index))

        i = i + 1

    return indices
