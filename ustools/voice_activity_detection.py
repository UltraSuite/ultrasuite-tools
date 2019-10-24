"""

Date: Sep 2018
Author: Aciel Eshky

"""

import os
import subprocess
import struct
import webrtcvad
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt


def detect_voice_activity(wav, sample_rate,
                          vad_wav_sample_rate=16000, aggressiveness=2, window_duration=0.03, bytes_per_sample=2):
    """
    Run voice activity detection on a given wav signal and return time segements of size "window_duration" with
    a boolean indicating whether or not the segment is speech.

    Code adapted from a Kaggle tutorial: https://www.kaggle.com/holzner/voice-activity-detection-example/notebook

    :param wav: numpy array containing a wav signal

    :param sample_rate: sample rate of the signal

    :param vad_wav_sample_rate: must be 8000, 16000, 32000 or 48000 Hz

    :param aggressiveness: an integer between 0 and 3. 0 is the least aggressive when filtering out non-speech,
            3 is the most aggressive.

    :param window_duration: in seconds. A frame must be either 0.01, 0.02, or 0.03 s in duration.

    :param bytes_per_sample:

    :return:
    """

    # VAD operates on a frame rate of 16000
    # so first I down-sample the wav form using ffmpeg while writing to disl
    # ffmpeg is the most reliable way I found of doing this.

    # 1) Down-sample wav:

    temp_wav = "temp_wav.wav"
    temp_downsampled_wav = "temp_downsampled_wav.wav"

    wavfile.write(filename=temp_wav, rate=sample_rate, data=wav)
    subprocess.call(["ffmpeg", "-loglevel", "panic", "-i", temp_wav, "-ar", str(vad_wav_sample_rate), temp_downsampled_wav])
    new_sample_rate, samples = wavfile.read(filename=temp_downsampled_wav)

    assert new_sample_rate == vad_wav_sample_rate

    # remove temp wav files
    os.remove(temp_wav)
    os.remove(temp_downsampled_wav)

    # 2) set up VAD:

    vad = webrtcvad.Vad(aggressiveness)  # set aggressiveness from 0 to 3 (low to high filtering of non-speech).

    raw_samples = struct.pack("%dh" % len(samples), *samples)
    samples_per_window = int(window_duration * new_sample_rate + 0.5)

    # 3) run VAD:

    time_segments = []

    for start in np.arange(0, len(samples) - samples_per_window, samples_per_window):
        stop = min(start + samples_per_window, len(samples))

        is_speech = vad.is_speech(raw_samples[start * bytes_per_sample: stop * bytes_per_sample],
                                  sample_rate=new_sample_rate)

        time_segments.append(dict(
            start=start / new_sample_rate,
            stop=stop / new_sample_rate,
            is_speech=is_speech))

    return time_segments


def visualise_voice_activity_detection(wav, sample_rate, time_segments):
    """
    Visualise the output of the VAD function "detection_voice_activity"
    
    :param wav: 
    :param sample_rate: 
    :param time_segments: 
    :return: 
    """

    plt.figure(figsize=(10, 7))
    plt.plot(wav)

    y_max = max(wav)

    # plot segment identifed as speech
    for segment in time_segments:
        if segment['is_speech']:
            start = segment['start'] * sample_rate
            stop = segment['stop'] * sample_rate
            plt.plot([start, stop - 1], [y_max * 1.1, y_max * 1.1], color='orange')

    plt.xlabel('sample')
    plt.grid()


def separate_silence_and_speech(signal, sample_rate, time_segments):
    """
    This can be applied to wav files and ultrasound files to separate silence and speech in a signal
    based on the time segments produced by the VAD function.

    :param signal:
    :param sample_rate:
    :param time_segments: output of the VAD function "detection_voice_activity"
    :return: two numpy arrays: "silence" and "speech"
    """

    silence = signal
    speech = signal

    for segment in reversed(time_segments):

        start = int(segment['start'] * sample_rate)
        stop = int(segment['stop'] * sample_rate)

        if segment['is_speech']:
            silence = np.delete(silence, np.s_[start:stop], axis=0)
        else:
            speech = np.delete(speech, np.s_[start:stop], axis=0)

    return silence, speech
