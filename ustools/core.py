"""
Class UltraSuiteCore to read / process UltraSuite data.

Date: Jul 2018
Author: Aciel Eshky

"""
import os
import io
from datetime import datetime
from scipy.io import wavfile
import numpy as np
from ustools.reshape_ultrasound import get_segment


class UltraSuiteCore:

    def __init__(self, directory=None, file_basename=None, apply_sync_param=False, ult_frame_skip=None):
        """
        Initialise new object
        :param directory: the directory containing the files
        :param file_basename: base file name without extension
        """

        self.basename = ""
        self.speaker_id = ""
        self.prompt = ""
        self.datetime = datetime(1900, 1, 1)

        self.wav = np.zeros(0)
        self.ult = np.zeros(0)
        self.params = {}

        if directory and file_basename:
            self.basename = file_basename
            self.read_prompt(os.path.join(directory, file_basename + ".txt"))
            self.read_wav(os.path.join(directory, file_basename + ".wav"))
            self.read_param(os.path.join(directory, file_basename + ".param"))
            self.read_ult(os.path.join(directory, file_basename + ".ult"))

            if apply_sync_param:
                self.apply_sync()

            if ult_frame_skip:
                self.skip_ult_frames()

    def read_prompt(self, file):
        """
        Read prompt file containing prompt, datetime, and speaker ID
        :param file:
        :return:
        """
        with io.open(file, mode="r", encoding='utf-8', errors='ignore') as prompt_f:
            self.prompt = prompt_f.readline().rstrip()
            self.datetime = datetime.strptime(prompt_f.readline().rstrip(), '%d/%m/%Y %H:%M:%S')
            self.speaker_id = prompt_f.readline().rstrip()

    def read_wav(self, file):
        """
        Read wave file into numpy array and store sampling rate in parameter dictionay.
        :param file:
        :return:
        """
        self.params['wav_fps'], self.wav = wavfile.read(file)

    def read_param(self, file):
        """
        Read parameter file.
        Adapted from https://stackoverflow.com/a/9161531/5190279
        :param file:
        :return:
        """
        with open(file) as myfile:
            for line in myfile:
                name, var = line.partition("=")[::2]
                self.params[name.strip()] = float(var)

        self.params.pop('Kind')  # not needed
        self.params['num_scanlines'] = int(self.params.pop('NumVectors'))
        self.params['size_scanline'] = int(self.params.pop('PixPerVector'))
        self.params['zero_offset'] = self.params.pop('ZeroOffset')
        self.params['angle'] = self.params.pop('Angle')
        self.params['bits_per_pixel'] = self.params.pop('BitsPerPixel')
        self.params['pixel_per_mm'] = self.params.pop('PixelsPerMm')
        self.params['ult_fps'] = self.params.pop('FramesPerSec')
        self.params['sync'] = self.params.pop('TimeInSecsOfFirstFrame')
        self.params['sync_applied'] = False

    def read_ult(self, file):
        """
        Read ultrasound file into a numpy array and reshape it
        :param file:
        :return:
        """
        with open(file, "rb") as f:
            self.ult = np.fromfile(f, dtype=np.uint8)
            self.ult = self.ult.reshape(-1, self.params['num_scanlines'], self.params['size_scanline'])

        return

    def apply_sync(self):
        """
        Synchronise the two signals and trim the end of the longer of the two.
        If the sync parameter is positive, then the wav is leading and the ult is lagging, therefore crop the wav.
        If the sync parameter is negative, then the ult is leading and the wav is lagging, therefore crop the ult.
        If the sync parameter is equal to zero, then the signals are already synchronised.
        :return:
        """

        if self.params['sync'] > 0:
            # if it's a positive value then the wav is leading and the ult is lagging. Therefore, crop the wav.
            self.wav = get_segment(signal=self.wav, sampling_rate=self.params['wav_fps'],
                                   start_time=self.params['sync'])

        elif self.params['sync'] < 0:
            # if it's a negative value then the ult is leading and the wav is lagging. Therefore, crop the ult.
            self.ult = get_segment(signal=self.ult, sampling_rate=self.params['ult_fps'],
                                   start_time=abs(self.params['sync']))

        self.trim_signal_end()  # trim the end of the longer of the two signals

        self.params['sync_applied'] = True

    def trim_signal_end(self):
        """
        Trim the end of the longer of the two signals.
        :return:
        """
        wav_dur = self.wav.shape[0] / self.params['wav_fps']
        ult_dur = self.ult.shape[0] / self.params['ult_fps']

        if wav_dur > ult_dur:
            self.wav = get_segment(signal=self.wav, sampling_rate=self.params['wav_fps'], start_time=0,
                                   end_time=ult_dur)
        elif wav_dur < ult_dur:
            self.ult = get_segment(signal=self.ult, sampling_rate=self.params['ult_fps'], start_time=0,
                                   end_time=wav_dur)

    def skip_ult_frames(self, skip=5):
        """
        Skip some ultrasound frames to reduce the frame rate.
        :param skip:
        :return:
        """
        self.ult = self.ult[0::skip]
        self.params['ult_fps'] /= 5
