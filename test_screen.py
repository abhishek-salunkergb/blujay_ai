import unittest
import os
from screen import capture_screen


class TestScreenModule(unittest.TestCase):

    def test_capture_screen_callable(self):
        self.assertTrue(callable(capture_screen))


if __name__ == "__main__":
    unittest.main()