"""
Each utterance is represented as a tuple of four files:

    The prompt file: .txt
    The audio file: .wav
    The ultrasound file: .ult
    The parameter file: .param

This file contains functions to read each.


Date: Dec 2017
Author: Aciel Eshky

"""

import io
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.io import wavfile


def parse_prompt_file(prompt_file):
    """
    Parse the UltraSuite prompt file. First line contains the prompt and the second contains the date and time.
    :param prompt_file:
    :return:
    """
    with io.open(prompt_file, mode="r", encoding='utf-8', errors='ignore') as prompt_f:
        return [prompt_f.readline().rstrip(),
                prompt_f.readline().rstrip()]


def get_datetime_object(datetime_string):
    """
    Interpret the UltraSuite prompt date and time string as a python datetime object
    :param datetime_string:
    :return:
    """
    return datetime.strptime(datetime_string, '%d/%m/%Y %H:%M:%S')


def parse_parameter_file(param_file):

    """
    A function to parse parameter file as a pandas dataframe.
    The parameter file contains the following parameters:

    NumVectors                 63.00000     the number of scanlines
    PixPerVector              412.00000     the number of datapoints per scanline
    ZeroOffset                 50.00000     the scanline offset from the origin (zero) when plotting
    BitsPerPixel                8.00000     the number of bits per pixel
    Angle                       0.03800     the angle of the scanline
    Kind                        0.00000
    PixelsPerMm                10.00000     the number of pixels per millimeter
    FramesPerSec              120.87700     the frame rate
    TimeInSecsOfFirstFrame      0.49265     the synchronisation offset in seconds relative to the audio

    :param param_file: the name of the parameter file. This ends with extension .param and is a text file.
    :return: a input frame containing the value of each parameter in a separate column
    """

    return pd.read_table(param_file, sep='=', index_col=0, names=["value"]).transpose()


def read_ultrasound_file(ult_file):

    """
    A function which read an ultrasound file as a numpy array
    :param ult_file: the name of the ultrasound file
    :return: numpy array
    """

    return np.fromfile(open(ult_file, "rb"), dtype=np.uint8)


def read_wav_file(wave_file):
    """
    A function to read the wave file.

    :param wave_file: .wav file
    :return: returns two values, the first is the frame rate (e.g., 22,050 Hz) and the second is a 1 dimensional
     numpy array or amplitude values
    """
    return wavfile.read(wave_file)


