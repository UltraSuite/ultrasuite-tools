"""

Date: Jul 2018
Author: Aciel Eshky

"""
from unittest import TestCase, main
from ustools.core import UltraSuiteCore
from ustools.chunk import Chunk


class TestChunk(TestCase):

    def test_all_1(self):
        core = UltraSuiteCore(directory="/Users/acieleshky/Dropbox/Work/Ultrax2020/input/", file_basename="sample")
        core.process(apply_sync=True, skip_ult_frames=True, stride=5)

        chunk = Chunk(core=core, ult_chunk_size=1, fbank_feat=True, mfcc_feat=True, transform_ult=True)
        self.assertEqual(chunk.wav_chunks.shape, (178, 1, 907))
        self.assertEqual(chunk.ult_chunks.shape, (178, 1, 63, 412))
        self.assertEqual(chunk.ult_t_chunks.shape, (178, 1, 286, 160))
        self.assertEqual(chunk.mfcc_chunks.shape, (178, 1, 4, 12))
        self.assertEqual(chunk.fbank_chunks.shape, (178, 1, 4, 26))

    def test_all_5(self):
        core = UltraSuiteCore(directory="/Users/acieleshky/Dropbox/Work/Ultrax2020/input/", file_basename="sample")
        core.process(apply_sync=True, skip_ult_frames=True, stride=5)

        chunk = Chunk(core=core, ult_chunk_size=5, fbank_feat=True, mfcc_feat=True, transform_ult=True)
        self.assertEqual(chunk.wav_chunks.shape, (35, 1, 4535))
        self.assertEqual(chunk.ult_chunks.shape, (35, 5, 63, 412))
        self.assertEqual(chunk.ult_t_chunks.shape, (35, 5, 286, 160))
        self.assertEqual(chunk.mfcc_chunks.shape, (35, 1, 20, 12))
        self.assertEqual(chunk.fbank_chunks.shape, (35, 1, 20, 26))

    def test_no_ult_trans(self):
        core = UltraSuiteCore(directory="/Users/acieleshky/Dropbox/Work/Ultrax2020/input/", file_basename="sample")
        core.process(apply_sync=True, skip_ult_frames=True, stride=5)

        chunk = Chunk(core=core, ult_chunk_size=5, fbank_feat=True, mfcc_feat=True, transform_ult=False)
        self.assertEqual(chunk.wav_chunks.shape, (35, 1, 4535))
        self.assertEqual(chunk.ult_chunks.shape, (35, 5, 63, 412))
        self.assertEqual(chunk.ult_t_chunks.shape, (0,))
        self.assertEqual(chunk.mfcc_chunks.shape, (35, 1, 20, 12))
        self.assertEqual(chunk.fbank_chunks.shape, (35, 1, 20, 26))


if __name__ == '__main__':
    main()
