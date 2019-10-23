"""
Unit tests for class UltraSuiteCore.

Date: Jul 2018
Author: Aciel Eshky

"""
from datetime import datetime
from unittest import TestCase, main
from ustools.core import UltraSuiteCore


class TestUltraSuiteCore(TestCase):

    def test_empty(self):
        core = UltraSuiteCore()
        self.assertEqual(core.basename, "")
        self.assertEqual(core.speaker_id, "")
        self.assertEqual(core.prompt, "")
        self.assertEqual(core.datetime, datetime(1900, 1, 1))
        self.assertEqual(core.wav.size, 0)
        self.assertEqual(core.ult.size, 0)
        self.assertEqual(core.params, {})

    def test_default(self):
        core = UltraSuiteCore(directory="../input/", file_basename="sample")
        self.assertEqual(core.basename, "sample")
        self.assertEqual(core.speaker_id, "UPX_01F_BL2")
        self.assertEqual(core.prompt, "packing Hague top guy")
        self.assertEqual(core.datetime, datetime(2015, 6, 26, 15, 9, 25))
        self.assertEqual(core.wav.size, 173056)
        self.assertEqual(core.ult.shape, (928, 63, 412))
        self.assertEqual(core.params['angle'],  0.038)
        self.assertEqual(core.params['bits_per_pixel'], 8.0)
        self.assertEqual(core.params['num_scanlines'], 63)
        self.assertEqual(core.params['pixel_per_mm'], 10.00)
        self.assertEqual(core.params['size_scanline'], 412)
        self.assertEqual(core.params['sync'], 0.5073)
        self.assertEqual(core.params['sync_applied'], False)
        self.assertEqual(core.params['ult_fps'], 121.618)
        self.assertEqual(core.params['wav_fps'], 22050)
        self.assertEqual(core.params['zero_offset'],  51.0)

    def test_sync(self):
        core = UltraSuiteCore(directory="../input/", file_basename="sample", apply_sync_param=True)
        self.assertEqual(core.basename, "sample")
        self.assertEqual(core.speaker_id, "UPX_01F_BL2")
        self.assertEqual(core.prompt, "packing Hague top guy")
        self.assertEqual(core.datetime, datetime(2015, 6, 26, 15, 9, 25))
        self.assertEqual(core.wav.size, 161870)
        self.assertEqual(core.ult.shape, (893, 63, 412))
        self.assertEqual(core.params['angle'], 0.038)
        self.assertEqual(core.params['bits_per_pixel'], 8.0)
        self.assertEqual(core.params['num_scanlines'], 63)
        self.assertEqual(core.params['pixel_per_mm'], 10.00)
        self.assertEqual(core.params['size_scanline'], 412)
        self.assertEqual(core.params['sync'], 0.5073)
        self.assertEqual(core.params['sync_applied'], True)
        self.assertEqual(core.params['ult_fps'], 121.618)
        self.assertEqual(core.params['wav_fps'], 22050)
        self.assertEqual(core.params['zero_offset'], 51.0)

    def test_sync_and_skip(self):
        core = UltraSuiteCore(directory="../input/", file_basename="sample", apply_sync_param=True, ult_frame_skip=5)
        self.assertEqual(core.basename, "sample")
        self.assertEqual(core.speaker_id, "UPX_01F_BL2")
        self.assertEqual(core.prompt, "packing Hague top guy")
        self.assertEqual(core.datetime, datetime(2015, 6, 26, 15, 9, 25))
        self.assertEqual(core.wav.size, 161870)
        self.assertEqual(core.ult.shape, (179, 63, 412))
        self.assertEqual(core.params['angle'], 0.038)
        self.assertEqual(core.params['bits_per_pixel'], 8.0)
        self.assertEqual(core.params['num_scanlines'], 63)
        self.assertEqual(core.params['pixel_per_mm'], 10.00)
        self.assertEqual(core.params['size_scanline'], 412)
        self.assertEqual(core.params['sync'], 0.5073)
        self.assertEqual(core.params['sync_applied'], True)
        self.assertEqual(core.params['ult_fps'], 24.3236)
        self.assertEqual(core.params['wav_fps'], 22050)
        self.assertEqual(core.params['zero_offset'], 51.0)


if __name__ == '__main__':
    main()
