"""

Date: Jul 2018
Author: Aciel Eshky

"""

import os
import io
from datetime import datetime

import numpy as np
import samplerate
from scipy.io import wavfile
from skimage.measure import block_reduce
from skimage.transform import resize

from ustools.segment_signal import get_segment, get_zero_regions
from ustools.transform_ultrasound import transform_ultrasound
from ustools.voice_activity_detection import detect_voice_activity, separate_silence_and_speech


class UltraSuiteCore:

    def __init__(self, directory=None, file_basename=None):
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
        self.ult_t = np.zeros(0)
        self.params = {}

        if directory and file_basename:
            self.basename = file_basename
            self.read_prompt(os.path.join(directory, file_basename + ".txt"))
            self.read_wav(os.path.join(directory, file_basename + ".wav"))
            self.read_param(os.path.join(directory, file_basename + ".param"))
            self.read_ult(os.path.join(directory, file_basename + ".ult"))

    def process(self,
                skip_ult_frames=False, stride=None,
                change_frame_rate=False, new_frame_rate=None,
                apply_sync=False, remove_zero_regions=False, apply_vad=False, transform_ult=False,
                resize_ult_frames_by_ratio=False, ratio=None,
                resize_ult_frames_by_size=False, new_frame_size=None
                ):
        """

        :param skip_ult_frames: first alternative for reducing the ult frame rate by skipping frames
        :param stride: take every nth frame, e.g., n=5
        :param change_frame_rate: first alternative for reducing the ult frame rate by re-sampling and interpolating
        :param new_frame_rate: e.g., 24 fps
        :param apply_sync:
        :param remove_zero_regions:
        :param apply_vad:
        :param transform_ult:
        :param resize_ult_frames_by_ratio: first alternative for changing the ult frame sizes by specifying a ratio
        :param ratio: e.g., (1, 3)
        :param resize_ult_frames_by_size: second alternative for changing the ult frame sizes by specifying a size
        :param new_frame_size: e.g., (63, 138)
        :return:
        """

        # two alternatives for changing the frame rate
        if skip_ult_frames:
            if not stride:  # if the stride has not been specified then it defaults to 5
                stride = 5
            self.skip_ult_frames(stride=stride)

        elif change_frame_rate:
            if not new_frame_rate:
                new_frame_rate = 24
            self.change_ult_frame_rate(new_frame_rate=new_frame_rate)

        # ultrasound transformation should apply to original ultrasound size
        if transform_ult:
            self.transform_ult()

        # two alternatives for changing the size of the ultrasound frames
        if resize_ult_frames_by_ratio:
            if not ratio:
                ratio = (1, 3)
            self.resize_ult_frames_by_ratio(ratio=ratio)

        elif resize_ult_frames_by_size:
            if not new_frame_size:
                new_frame_size=(63, 138)
            self.resize_ult_frames(output_size=new_frame_size)

        # applying sync
        if apply_sync:
            self.apply_sync()

            if remove_zero_regions:  # should be applied only to synchronised signals
                self.remove_zero_regions()

            if apply_vad:  # should be applied only to synchronised signals
                self.apply_vad()

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

    def write_prompt(self, directory):
        """
        Read prompt file containing prompt, datetime, and speaker ID
        :param directory:
        :return:
        """
        with io.open(os.path.join(directory, self.basename + ".txt"),
                     mode="w", encoding='utf-8', errors='ignore') as prompt_f:
            prompt_f.write(self.prompt + '\n')
            prompt_f.write(self.datetime.strftime('%d/%m/%Y %H:%M:%S') + '\n')
            prompt_f.write(self.speaker_id)

    def read_wav(self, file):
        """
        Read wave file into numpy array and store sampling rate in parameter dictionay.
        :param file:
        :return:
        """
        self.params['wav_fps'], self.wav = wavfile.read(file)

    def write_wav(self, directory):
        """
        Read wave file into numpy array and store sampling rate in parameter dictionay.
        :param directory:
        :return:
        """
        wavfile.write(os.path.join(directory, self.basename + ".wav"), self.params['wav_fps'], self.wav)

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

        self.params['kind'] = int(self.params.pop('Kind'))  # not needed
        self.params['num_scanlines'] = int(self.params.pop('NumVectors'))
        self.params['size_scanline'] = int(self.params.pop('PixPerVector'))
        self.params['zero_offset'] = self.params.pop('ZeroOffset')
        self.params['angle'] = self.params.pop('Angle')
        self.params['bits_per_pixel'] = self.params.pop('BitsPerPixel')
        self.params['pixel_per_mm'] = self.params.pop('PixelsPerMm')
        self.params['ult_fps'] = self.params.pop('FramesPerSec')
        self.params['sync'] = self.params.pop('TimeInSecsOfFirstFrame')
        self.params['sync_applied'] = False
        self.params['zero_removed'] = False
        self.params['vad_applied'] = False
        self.params['ult_transformed'] = False
        self.params['ult_frame_rate_changed'] = False
        self.params["ult_frame_resized"] = False

    def write_param(self, directory):
        """
        Read parameter file.
        Adapted from https://stackoverflow.com/a/9161531/5190279
        :param directory:
        :return:
        """
        with io.open(os.path.join(directory, self.basename + ".param"), mode="w", encoding='utf-8', errors='ignore') as param_file:
            param_file.write('Kind=' + str(self.params['kind']) + '\n')
            param_file.write('NumVectors=' + str(self.params['num_scanlines']) + '\n')
            param_file.write('PixPerVector=' + str(self.params['size_scanline']) + '\n')
            param_file.write('ZeroOffset=' + str(self.params['zero_offset']) + '\n')
            param_file.write('Angle=' + str(self.params['angle']) + '\n')
            param_file.write('BitsPerPixel=' + str(self.params['bits_per_pixel']) + '\n')
            param_file.write('PixelsPerMm=' + str(self.params['pixel_per_mm']) + '\n')
            param_file.write('FramesPerSec=' + str(self.params['ult_fps']) + '\n')
            param_file.write('TimeInSecsOfFirstFrame=' + str(self.params['sync']))

    def read_ult(self, file):
        """
        Read ultrasound file into a numpy array and reshape it
        :param file:
        :return:
        """
        with open(file, "rb") as f:
            self.ult = np.fromfile(f, dtype=np.uint8)
            self.ult = self.ult.reshape(-1, self.params['num_scanlines'], self.params['size_scanline'])

    def write_ult(self, directory):
        """
        Read ultrasound file into a numpy array and reshape it
        :param directory:
        :return:
        """
        with open(os.path.join(directory, self.basename + ".ult"), "wb") as f:
            self.ult.astype(np.uint8).tofile(f)

    def skip_ult_frames(self, stride=5):
        """
        Skip some ultrasound frames to reduce the frame rate.
        :param stride: the step size. e.g. if stride == 5 then the following indices are retained: 0, 5, 10, 15, etc.
        :return:
        """
        if not self.params['ult_frame_rate_changed']:
            self.ult = self.ult[0::stride]
            self.params['ult_fps'] /= 5
            self.params['ult_frame_rate_changed'] = True

    def change_ult_frame_rate(self, new_frame_rate):
        if not self.params['ult_frame_rate_changed']:
            ratio = new_frame_rate / self.params['ult_fps']
            temp = np.apply_along_axis(lambda x: samplerate.resample(x, ratio, 'linear'), 0, self.ult)
            self.ult = temp.round().astype(int)
            self.params['ult_fps'] = new_frame_rate
            self.params['ult_frame_rate_changed'] = True

    def transform_ult(self):
        """
        Transform the ultrasound.
        :return:
        """
        if self.params['ult_frame_resized'] and not self.params['ult_transformed']:
            print("ultrasound has been down-sampled. No transform applied.")

        elif not self.params['ult_frame_resized'] and not self.params['ult_transformed']:
            self.ult_t = transform_ultrasound(self.ult, num_scanlines=self.params['num_scanlines'],
                                              size_scanline=self.params['size_scanline'], angle=self.params['angle'],
                                              zero_offset=self.params['zero_offset'], pixels_per_mm=3)
            self.params['ult_transformed'] = True

    def resize_ult_frames_by_ratio(self, ratio=(1, 3), func=np.mean):
        """
        down-sample the ultrasound frames
        :param ratio:
        :param func:
        :return:
        """

        if not self.params['ult_frame_resized']:

            resized = []
            for i, image in enumerate(self.ult):
                temp = block_reduce(image, block_size=ratio, func=func)
                temp = temp.round().astype(int)
                resized.append(temp)

            self.ult = np.array(resized)
            self.params['num_scanlines'] = self.ult.shape[1]
            self.params['size_scanline'] = self.ult.shape[2]
            self.params['ult_frame_resized'] = True

    def resize_ult_frames(self, output_size=(63, 138)):
        """
        down-sample the ultrasound frames
        :param output_size:
        :return:
        """

        if not self.params['ult_frame_resized']:

            resized = []
            for i, image in enumerate(self.ult):
                temp = resize(image, output_shape=output_size, order=0, mode='reflect', clip=False,
                              preserve_range=True, anti_aliasing=False)
                temp = temp.round().astype(int)
                resized.append(temp)

            self.ult = np.array(resized)
            self.params['num_scanlines'] = output_size[0]
            self.params['size_scanline'] = output_size[1]
            self.params["ult_frame_resized"] = True

            assert (self.ult.shape[1] == output_size[0])
            assert (self.ult.shape[2] == output_size[1])

    def apply_sync(self):
        """
        Synchronise the two signals and trim the end of the longer of the two.
        If the sync parameter is positive, then the wav is leading and the ult is lagging, therefore crop the wav.
        If the sync parameter is negative, then the ult is leading and the wav is lagging, therefore crop the ult.
        If the sync parameter is equal to zero, then the signals are already synchronised.
        :return:
        """
        if not self.params['sync_applied']:

            if self.params['sync'] > 0:
                # if it's a positive value then the wav is leading and the ult is lagging. Therefore, crop the wav.
                self.wav = get_segment(signal=self.wav, sampling_rate=self.params['wav_fps'],
                                       start_time=self.params['sync'])

            elif self.params['sync'] < 0:
                # if it's a negative value then the ult is leading and the wav is lagging. Therefore, crop the ult.
                self.ult = get_segment(signal=self.ult, sampling_rate=self.params['ult_fps'],
                                       start_time=abs(self.params['sync']))

            self.trim_signal_end()  # trim the end of the longer of the two signals

            self.remove_zero_regions() #

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

    def remove_zero_regions(self):
        """
        Some regions of audio were zero-ed to remove personal or identifying information.
        This function removes the corresponding ultrasound.
        :return:
        """
        if not self.params['zero_removed']:
            indices_of_zero_regions_wav = get_zero_regions(self.wav, num_repetitions=100)

            for start, end in reversed(indices_of_zero_regions_wav):
                self.wav = np.delete(self.wav, np.s_[start:end + 1])

            indices_of_zero_regions_ult = np.multiply(indices_of_zero_regions_wav,
                                                      self.params['ult_fps'] / self.params['wav_fps'])
            indices_of_zero_regions_ult = np.rint(indices_of_zero_regions_ult)
            indices_of_zero_regions_ult = indices_of_zero_regions_ult.astype(int)

            for start, end in reversed(indices_of_zero_regions_ult):
                self.ult = np.delete(self.ult, np.s_[start:end + 1], axis=0)

            self.params['zero_removed'] = True

    def apply_vad(self):
        """
        Apply voice activity detection.
        :return:
        """
        if not self.params['vad_applied']:

            # get time segments
            time_segments = detect_voice_activity(wav=self.wav,
                                                  sample_rate=self.params['wav_fps'],
                                                  vad_wav_sample_rate=16000,
                                                  aggressiveness=2,  # was 2
                                                  window_duration=0.03,
                                                  bytes_per_sample=2)

            # apply to wav
            silence, speech = separate_silence_and_speech(self.wav, self.params['wav_fps'], time_segments)
            self.wav = speech

            # apply to ult
            silence, speech = separate_silence_and_speech(self.ult, self.params['ult_fps'], time_segments)
            self.ult = speech

            # if ult has been transformed, apply to ult_t
            if self.params['ult_transformed']:
                silence, speech = separate_silence_and_speech(self.ult_t, self.params['ult_fps'], time_segments)
                self.ult_t = speech

            # set the vad_applied parameter to true
            self.params['vad_applied'] = True
