import unittest
import os
from audio import start_system_audio, stop_system_audio


class TestAudioModule(unittest.TestCase):

    def test_audio_functions_exist_and_callable(self):
        self.assertTrue(callable(start_system_audio))
        self.assertTrue(callable(stop_system_audio))

    def test_stop_system_audio_when_idle(self):
        result = stop_system_audio()
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()