"""

Date: Jun 2018
Author: Aciel Eshky

"""
import python_speech_features as psf
import matplotlib.pyplot as plt
import numpy as np


def get_logfbank_feat(wav, samplerate=22050, winlen=0.02, winstep=0.01):
    """

    :param wav:
    :param samplerate:
    :param winlen:
    :return:
    """
    return psf.logfbank(signal=wav, samplerate=samplerate, winlen=winlen, winstep=winstep)


def get_mfcc_feat(wav, samplerate=22050, winlen=0.02, winstep=0.01, drop_first_mfcc=False):
    """

    :param wav: the waveform
    :param samplerate: the same rate of the waveform, e.g., 22050 HZ
    :param winlen: The size of the window. This should be equal to the ultrasound window in seconds.
    The skip will be calculated as size of window / 2
    :param drop_first_mfcc: discard the first mfcc
    :return:
    """
    mfcc_feat = psf.mfcc(signal=wav, samplerate=samplerate, winlen=winlen, winstep=winstep)

    if drop_first_mfcc:
        return mfcc_feat[:, 1:]  # get only the 2nd-13th DCT coefficients (indices 1-13 inclusive)
    else:
        return mfcc_feat


def visualise_logfbank_feat(logfbank_feat):
    """

    :param logfbank_feat:
    :return:
    """
    plt.figure(figsize=(15, 5))
    plt.ylabel("Frequency (KHz)", fontsize=25)

    plt.xlabel("Time (s)", fontsize=25)

    plt.tick_params(labelsize=15)
    plt.imshow(logfbank_feat.T, cmap="jet", origin='lower', aspect='auto')
    plt.show()


def visualise_mfcc_feat(mfcc_feat, start_index=2):
    """

    :param mfcc_feat:
    :return:
    """
    plt.figure(figsize=(15, 5))
    plt.ylabel("MFCC Coefficient", fontsize=25)
    plt.yticks(np.arange(0, mfcc_feat.shape[1] + 1, step=1), np.arange(start_index, mfcc_feat.shape[1] + 2, step=1))

    plt.xlabel("Time (s)", fontsize=25)

    plt.tick_params(labelsize=15)
    plt.imshow(mfcc_feat.T, cmap="jet", origin='lower', aspect='auto')
    plt.show()
