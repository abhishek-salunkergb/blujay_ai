import os
import sys
import json
import unittest
from unittest.mock import MagicMock, patch

# Ensure repo root is on path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from speech import DeepgramLiveWebSocketClient


class TestDeepgramFluxTurnEvents(unittest.TestCase):
    """
    Unit test suite verifying Deepgram Flux integration:
    1. Sample-level 80ms PCM byte accumulation (exact 2560-byte chunks).
    2. TurnInfo events: EagerEndOfTurn, TurnResumed, EndOfTurn.
    3. Defensive alias: SpeechResumed.
    4. Resilient auto-fallback to Nova-3.
    """

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


    def test_01_sample_level_80ms_accumulation(self):
        """Verify that incoming 30ms frames (960 bytes) accumulate and slice into exact 2560-byte (80ms) chunks."""
        client = DeepgramLiveWebSocketClient(api_key="test_key", model="flux-general-en")
        client.is_running = True
        
        # 1 frame = 960 bytes (30ms). 960 < 2560 -> no packet sent yet
        frame_30ms = b"\x00" * 960
        client.send_frame(frame_30ms)
        self.assertTrue(client.audio_queue.empty())
        self.assertEqual(len(client._flux_pcm_buffer), 960)

        # 2nd frame = 960 bytes (total 1920 bytes). Still < 2560 -> no packet sent yet
        client.send_frame(frame_30ms)
        self.assertTrue(client.audio_queue.empty())
        self.assertEqual(len(client._flux_pcm_buffer), 1920)

        # 3rd frame = 960 bytes (total 2880 bytes). 2880 >= 2560!
        # Exactly ONE 2560-byte packet must be enqueued, and remainder (320 bytes) kept in buffer!
        client.send_frame(frame_30ms)
        self.assertFalse(client.audio_queue.empty())
        packet = client.audio_queue.get_nowait()
        self.assertEqual(len(packet), 2560)
        self.assertEqual(len(client._flux_pcm_buffer), 320)

    def test_02_flux_eager_eot_event(self):
        """Verify that EagerEndOfTurn triggers on_eager_eot callback."""
        mock_eager = MagicMock()
        mock_final = MagicMock()
        client = DeepgramLiveWebSocketClient(
            api_key="test_key",
            model="flux-general-en",
            on_eager_eot=mock_eager,
            on_final=mock_final
        )
        client.is_running = True

        mock_ws = MagicMock()
        # Deepgram Flux v2 TurnInfo message
        flux_msg = json.dumps({
            "type": "TurnInfo",
            "event": "EagerEndOfTurn",
            "transcript": "Can you explain how indexing works",
            "turn_index": 1
        })
        mock_ws.recv.side_effect = [flux_msg, StopIteration]

        client._receiver_loop(mock_ws)

        mock_eager.assert_called_once_with("Can you explain how indexing works", "mic")
        mock_final.assert_not_called()

    def test_03_flux_turn_resumed_event(self):
        """Verify that TurnResumed triggers on_turn_resumed cancellation callback."""
        mock_resumed = MagicMock()
        client = DeepgramLiveWebSocketClient(
            api_key="test_key",
            model="flux-general-en",
            on_turn_resumed=mock_resumed
        )
        client.is_running = True

        mock_ws = MagicMock()
        flux_msg = json.dumps({
            "type": "TurnInfo",
            "event": "TurnResumed",
            "turn_index": 1
        })
        mock_ws.recv.side_effect = [flux_msg, StopIteration]

        client._receiver_loop(mock_ws)

        mock_resumed.assert_called_once_with("mic")

    def test_04_flux_speech_resumed_alias(self):
        """Verify that SpeechResumed alias also triggers on_turn_resumed cancellation callback."""
        mock_resumed = MagicMock()
        client = DeepgramLiveWebSocketClient(
            api_key="test_key",
            model="flux-general-en",
            on_turn_resumed=mock_resumed
        )
        client.is_running = True

        mock_ws = MagicMock()
        flux_msg = json.dumps({
            "type": "TurnInfo",
            "event": "SpeechResumed",
            "turn_index": 1
        })
        mock_ws.recv.side_effect = [flux_msg, StopIteration]

        client._receiver_loop(mock_ws)

        mock_resumed.assert_called_once_with("mic")

    def test_05_flux_end_of_turn_event(self):
        """Verify that EndOfTurn triggers on_final callback."""
        mock_final = MagicMock()
        client = DeepgramLiveWebSocketClient(
            api_key="test_key",
            model="flux-general-en",
            on_final=mock_final
        )
        client.is_running = True

        mock_ws = MagicMock()
        flux_msg = json.dumps({
            "type": "TurnInfo",
            "event": "EndOfTurn",
            "transcript": "Can you explain how indexing works specifically in PostgreSQL?",
            "turn_index": 1
        })
        mock_ws.recv.side_effect = [flux_msg, StopIteration]

        client._receiver_loop(mock_ws)

        mock_final.assert_called_once_with("Can you explain how indexing works specifically in PostgreSQL?", "mic")

    def test_06_auto_fallback_to_nova3_url(self):
        """Verify that if model is nova-3, _build_url generates v1/listen endpoint."""
        client = DeepgramLiveWebSocketClient(api_key="test_key", model="nova-3")
        url = client._build_url()
        self.assertIn("v1/listen", url)
        self.assertIn("model=nova-3", url)

        # Flux generates v2/listen endpoint with eager_eot_threshold and eot_threshold
        flux_client = DeepgramLiveWebSocketClient(api_key="test_key", model="flux-general-en")
        flux_url = flux_client._build_url()
        self.assertIn("v2/listen", flux_url)
        self.assertIn("model=flux-general-en", flux_url)
        self.assertIn("eager_eot_threshold=0.5", flux_url)
        self.assertIn("eot_threshold=0.8", flux_url)


if __name__ == "__main__":
    unittest.main()
