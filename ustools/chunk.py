"""

Date: Jul 2018
Author: Aciel Eshky

"""

import numpy as np
from ustools.speech_features import get_mfcc_feat, get_logfbank_feat

IDEAL_ULT_FPS = 121.5 / 5


class Chunk:
    def __init__(self, core, ult_chunk_size=5, mfcc_feat=False, drop_first_mfcc=False, fbank_feat=False, transform_ult=False):

        if (core.ult.size != 0 and core.wav.size != 0 and core.params != {} and core.params['ult_fps'] != ""
                and core.params['wav_fps'] != "" and core.params['ult_transformed'] != ""):

            self.core = core

            self.ult_chunk_size = ult_chunk_size
            self.time_window = ult_chunk_size / core.params['ult_fps']
            self.speech_feature_time_window = self.time_window / (self.ult_chunk_size * 2)
            self.speech_feature_time_step = self.time_window / (self.ult_chunk_size * 4)
            self.drop_first_mfcc = drop_first_mfcc

            self.ult_chunks = np.zeros(0)
            self.wav_chunks = np.zeros(0)
            self.mfcc_chunks = np.zeros(0)
            self.fbank_chunks = np.zeros(0)
            self.ult_t_chunks = np.zeros(0)
            self.chunk_ids = np.zeros(0)

            self.get_wav_chunks()
            self.get_ult_chunks()
            self.generate_chunk_ids()

            if mfcc_feat:
                self.get_mfcc_chunks()

            if fbank_feat:
                self.get_fbank_chunks()

            if transform_ult:
                self.get_transformed_ult_chunks()

            self.force_shortest_size()

    @staticmethod
    def chunk_array(a, step_size, window_length=None):
        """

        :param a:
        :param step_size:
        :param window_length:
        :return:
        """
        if not window_length:
            window_length = step_size

        chunks = []  # list of chunks
        for i in range(0, len(a), step_size):
            item = a[i:i + window_length]
            if len(item) < window_length:
                break
            else:
                chunks.append(item)

        # now all of them should be the same length unless there was an error
        if not all(len(item) == window_length for item in chunks):
            raise ValueError("Unequal array lengths")

        return np.array(chunks)

    def get_wav_chunks(self):
        """
        We chunk the audio based on the ult chunk size. The wav step size should be the same for all utterances.
         However the ult frame rate varies. Therefore, the start of the frame is calculated using the
         utterance-specific frame rate while the step size is calculated using the ideal frame rate
         (IDEAL_ULT_FPS = 121.5 / n, where n is the step size when we down-sampled the ult).

        :return:
        """
        # The step size is utterance-specific
        step_size = self.ult_chunk_size * int(round(self.core.params["wav_fps"] / self.core.params["ult_fps"]))

        # The window length should be the same for all utterances
        window_length = self.ult_chunk_size * int(round(self.core.params["wav_fps"] / IDEAL_ULT_FPS))

        self.wav_chunks = self.chunk_array(self.core.wav, step_size=step_size, window_length=window_length)
        self.wav_chunks = np.expand_dims(self.wav_chunks, axis=1)

    def get_ult_chunks(self):
        """

        :return:
        """
        self.ult_chunks = self.chunk_array(self.core.ult, step_size=self.ult_chunk_size)

    def generate_chunk_ids(self):
        """

        :return:
        """
        length = max([len(self.wav_chunks), len(self.ult_chunks)])
        self.chunk_ids = np.array(["ch_" + str(i) for i in range(0, length)])

    def get_mfcc_chunks(self):
        """

        :return:
        """
        mfcc_feat = get_mfcc_feat(wav=self.core.wav, samplerate=self.core.params['wav_fps'],
                                  winlen=self.speech_feature_time_window, winstep=self.speech_feature_time_step,
                                  drop_first_mfcc=self.drop_first_mfcc)
        self.mfcc_chunks = self.chunk_array(mfcc_feat, step_size=self.ult_chunk_size * 4)
        self.mfcc_chunks = np.expand_dims(self.mfcc_chunks, axis=1)

    def get_fbank_chunks(self):
        """

        :return:
        """
        fbank_feat = get_logfbank_feat(wav=self.core.wav, samplerate=self.core.params['wav_fps'],
                                       winlen=self.speech_feature_time_window, winstep=self.speech_feature_time_step)
        self.fbank_chunks = self.chunk_array(fbank_feat, step_size=self.ult_chunk_size * 4)
        self.fbank_chunks = np.expand_dims(self.fbank_chunks, axis=1)

    def get_transformed_ult_chunks(self):
        """

        :return:
        """
        if not self.core.params['ult_transformed']:
            self.core.transform_ult()

        self.ult_t_chunks = self.chunk_array(self.core.ult_t, step_size=self.ult_chunk_size)

    def force_shortest_size(self):
        """
        When chunking, some lists will be longer than others,
        :return:
        """
        lengths = {len(self.ult_chunks),
                   len(self.ult_t_chunks),
                   len(self.wav_chunks),
                   len(self.mfcc_chunks),
                   len(self.fbank_chunks),
                   len(self.chunk_ids)}

        if 0 in lengths:
            lengths.remove(0)

        if len(lengths) != 0:
            a = min(lengths)
            self.ult_chunks = self.ult_chunks[:a]
            self.ult_t_chunks = self.ult_t_chunks[:a]
            self.wav_chunks = self.wav_chunks[:a]
            self.mfcc_chunks = self.mfcc_chunks[:a]
            self.fbank_chunks = self.fbank_chunks[:a]
            self.chunk_ids = self.chunk_ids[:a]

        else:
            print("Warning: empty chunks.")

    @staticmethod
    def save_sync_data(filename, raw_ult=None, trans_ult=None, raw_wav=None, logfbank_feat=None, mfcc_feat=None):
        """

        :param filename:
        :param raw_ult:
        :param trans_ult:
        :param raw_wav:
        :param logfbank_feat:
        :param mfcc_feat:
        :return:
        """
        np.savez(filename,
                 raw_ult=raw_ult,
                 trans_ult=trans_ult,
                 raw_wav=raw_wav,
                 logfbank_feat=logfbank_feat,
                 mfcc_feat=mfcc_feat)

    @staticmethod
    def load_sync_data(filename):
        """

        :param filename:
        :return:
        """
        return np.load(filename + '.npz')
