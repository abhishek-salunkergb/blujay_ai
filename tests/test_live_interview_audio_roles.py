"""
Automated Test Suite for Live Interview Mode Audio Roles & Startup Architecture
================================================================================
Verifies:
1. Dual Mode: System Loopback = Interviewer (Answers triggered for questions), Candidate Mic = Candidate (Context stored, NO answers triggered).
2. Loopback Only Mode: System = Interviewer.
3. Mic Only Mode: Single Mic = Multi-signal heuristic attribution.
4. Interviewer Statement Protection: System statements/fillers -> Transcript only, NO answer.
5. Interviewer Question Detection: System questions -> Interviewer question, answer generated.
6. Candidate Speech Protection: Candidate Mic speech -> Candidate role, context updated, NO answer.
7. Immediate Startup: start() starts instantly without device probing or artificial delay.
8. Loopback Failure Resilience: Graceful error status without crash, session remains alive.
9. Microphone Failure Resilience: Graceful error status without crash, session remains alive.
10. Multi-Turn Context Continuity: System Q1 -> Mic Answer -> System Q2 (Answer Intelligence fuses context).
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


import sys
import os
import unittest
import time
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import interview_engine
from interview_engine import (
    InterviewSession,
    ContinuousTranscriber,
    detect_speaker_turn,
    classify_and_understand_utterance,
    STTProvider,
    GroqWhisperProvider
)


class MockSTTProvider(STTProvider):
    def __init__(self, response_text: str = ""):
        self.response_text = response_text
        
    def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000, prompt: str = "") -> str:
        return self.response_text


class TestLiveInterviewAudioRoles(unittest.TestCase):

    def setUp(self):
        self.session = InterviewSession()
        self.session.clear()

    # ============================================================
    # TEST 1: DEFAULT DUAL MODE SETTINGS
    # ============================================================
    def test_01_default_session_audio_source(self):
        """Verify default session audio_source is 'dual' and candidate_context_enabled is True."""
        session = InterviewSession()
        self.assertEqual(session.audio_source, "dual")
        self.assertTrue(session.candidate_context_enabled)

    # ============================================================
    # TEST 2: DUAL MODE - SYSTEM LOOPBACK IS INTERVIEWER
    # ============================================================
    def test_02_dual_mode_system_loopback_question(self):
        """In Dual mode, system loopback source maps to Interviewer and triggers answer for questions."""
        analysis = classify_and_understand_utterance(
            utterance="Why did you choose React?",
            session=self.session,
            source="system"
        )
        self.assertEqual(analysis["speaker"], "Interviewer")
        self.assertTrue(analysis["should_generate_answer"])
        self.assertIn("React", analysis["refined_question"])

    def test_03_dual_mode_system_loopback_statement(self):
        """In Dual mode, system loopback statements are logged as Interviewer but do NOT trigger answers."""
        analysis = classify_and_understand_utterance(
            utterance="Okay, give me a second to check your code.",
            session=self.session,
            source="system"
        )
        self.assertEqual(analysis["speaker"], "Interviewer")
        self.assertFalse(analysis["should_generate_answer"])
        self.assertIn(analysis["utterance_type"], ("STATEMENT", "ACKNOWLEDGEMENT"))

    # ============================================================
    # TEST 3: DUAL MODE - CANDIDATE MIC IS CANDIDATE (CONTEXT ONLY)
    # ============================================================
    def test_04_dual_mode_candidate_mic_speech(self):
        """In Dual mode, candidate_mic source maps to Candidate, stores context, and NEVER triggers answers."""
        analysis = classify_and_understand_utterance(
            utterance="I used React and Django for the e-commerce project.",
            session=self.session,
            source="candidate_mic"
        )
        self.assertEqual(analysis["speaker"], "Candidate")
        self.assertFalse(analysis["should_generate_answer"])
        self.assertEqual(analysis["utterance_type"], "CANDIDATE_ANSWER")
        
        # Verify candidate speech was recorded in session context
        self.assertTrue(any("React and Django" in r["text"] for r in self.session.candidate_responses_log))

    def test_05_dual_mode_candidate_mic_interrogative_never_answers(self):
        """Even if candidate utterance sounds like a question, candidate_mic channel NEVER triggers AI answer."""
        analysis = classify_and_understand_utterance(
            utterance="Can you hear me properly? What should I explain first?",
            session=self.session,
            source="candidate_mic"
        )
        self.assertEqual(analysis["speaker"], "Candidate")
        self.assertFalse(analysis["should_generate_answer"])

    # ============================================================
    # TEST 4: LOOPBACK ONLY MODE
    # ============================================================
    def test_06_loopback_only_mode(self):
        """In loopback-only mode, system source maps to Interviewer."""
        turn = detect_speaker_turn("What database did you use?", session=self.session, source="system")
        self.assertEqual(turn["speaker"], "INTERVIEWER")
        self.assertAlmostEqual(turn["confidence"], 0.99, delta=0.01)

    # ============================================================
    # TEST 5: MIC ONLY MODE (HEURISTIC ATTRIBUTION PRESERVED)
    # ============================================================
    def test_07_mic_only_heuristic_mode(self):
        """In Mic-only mode (source='mic'), semantic heuristic attribution is preserved."""
        # 1. Interviewer question starter
        turn_q = detect_speaker_turn("Why did you choose MongoDB?", session=self.session, source="mic")
        self.assertEqual(turn_q["speaker"], "INTERVIEWER")
        
        # 2. Candidate 1st-person response
        turn_c = detect_speaker_turn("I chose MongoDB because of flexible JSON documents.", session=self.session, source="mic")
        self.assertEqual(turn_c["speaker"], "CANDIDATE")

    # ============================================================
    # TEST 6: IMMEDIATE STARTUP (NO DEVICE PROBING DELAY)
    # ============================================================
    def test_08_immediate_startup_no_delay(self):
        """Verify ContinuousTranscriber.start() starts immediately without device probing or artificial wait."""
        client_mock = MagicMock()
        transcriber = ContinuousTranscriber(
            client=client_mock,
            on_transcript_chunk=lambda text, is_q, source="system": None,
            provider=MockSTTProvider("test")
        )
        
        t0 = time.perf_counter()
        # Mock soundcard / sounddevice stream to prevent hardware dependency in test
        with patch.object(transcriber, "_run_system_audio_loop"), patch.object(transcriber, "_run_microphone_loop"):
            transcriber.start(source="dual")
            t_elapsed = time.perf_counter() - t0
            self.assertTrue(transcriber.is_running)
            self.assertLess(t_elapsed, 0.05, "start() took too long; unexpected blocking wait or device probing")
            transcriber.stop()
            self.assertFalse(transcriber.is_running)

    # ============================================================
    # TEST 7: FAILURE RESILIENCE
    # ============================================================
    def test_09_loopback_failure_graceful_handling(self):
        """If system loopback fails to initialize, transcriber logs status, does not crash, and remains alive."""
        status_messages = []
        client_mock = MagicMock()
        transcriber = ContinuousTranscriber(
            client=client_mock,
            on_transcript_chunk=lambda text, is_q, source="system": None,
            on_status_change=lambda msg: status_messages.append(msg),
            provider=MockSTTProvider("test")
        )
        
        # Simulate soundcard exception during loopback run
        def _stop_on_loopback_error(msg):
            status_messages.append(msg)
            transcriber.is_running = False

        transcriber.on_status_change = _stop_on_loopback_error
        with patch("interview_engine.SOUNDCARD_AVAILABLE", True), \
             patch("interview_engine.sc.default_speaker", side_effect=Exception("Audio loopback endpoint unavailable")):
            # Running loop should catch exception without crashing
            transcriber.is_running = True
            transcriber._run_system_audio_loop()
            self.assertTrue(any("Loopback audio error" in msg for msg in status_messages))

    def test_10_microphone_failure_graceful_handling(self):
        """If microphone fails to initialize, transcriber logs status and does not crash."""
        status_messages = []
        client_mock = MagicMock()
        transcriber = ContinuousTranscriber(
            client=client_mock,
            on_transcript_chunk=lambda text, is_q, source="system": None,
            on_status_change=lambda msg: status_messages.append(msg),
            provider=MockSTTProvider("test")
        )

        def _stop_on_mic_error(msg):
            status_messages.append(msg)
            transcriber.is_running = False

        transcriber.on_status_change = _stop_on_mic_error
        with patch("interview_engine.sd.RawInputStream", side_effect=Exception("Microphone access denied")):
            transcriber.is_running = True
            transcriber._run_microphone_loop(channel_source="candidate_mic")
            self.assertTrue(any("Microphone error" in msg for msg in status_messages))

    # ============================================================
    # TEST 8: MULTI-TURN CONTEXT CONTINUITY IN DUAL MODE
    # ============================================================
    def test_11_multi_turn_context_continuity(self):
        """
        Verify multi-turn flow in Dual Mode:
        Turn 1 (System): 'Why did you choose MongoDB?' -> Interviewer Q -> Answer generated.
        Turn 2 (Mic): 'Because the schema was flexible.' -> Candidate context stored, NO answer.
        Turn 3 (System): 'What about scalability?' -> Interviewer follow-up with Answer Intelligence seeing Turn 1 and Turn 2.
        """
        # Turn 1: System Loopback (Interviewer Question)
        a1 = classify_and_understand_utterance("Why did you choose MongoDB?", session=self.session, source="system")
        self.assertEqual(a1["speaker"], "Interviewer")
        self.assertTrue(a1["should_generate_answer"])
        self.session.add_transcript("Why did you choose MongoDB?", role="Interviewer", is_question=True)
        self.session.active_topic = "MongoDB"
        self.session.latest_detected_question = "Why did you choose MongoDB?"

        # Turn 2: Microphone (Candidate Answer)
        a2 = classify_and_understand_utterance("Because the schema was flexible.", session=self.session, source="candidate_mic")
        self.assertEqual(a2["speaker"], "Candidate")
        self.assertFalse(a2["should_generate_answer"])
        self.session.add_transcript("Because the schema was flexible.", role="Candidate", is_question=False)

        # Turn 3: System Loopback (Interviewer Follow-Up)
        a3 = classify_and_understand_utterance("What about scalability?", session=self.session, source="system")
        self.assertEqual(a3["speaker"], "Interviewer")
        self.assertTrue(a3["should_generate_answer"])
        self.assertEqual(a3["utterance_type"], "FOLLOW_UP")
        # Refined question should fuse the candidate context from microphone turn
        self.assertIn("schema was flexible", a3["refined_question"].lower())

    # ============================================================
    # TEST 9: MANUAL INPUT BYPASSES AUDIO SPEAKER DETECTION
    # ============================================================
    def test_12_manual_input_bypasses_speaker_detection(self):
        """Manual typed input (Ctrl+Enter) triggers unified question processor without labeling as Candidate or Interviewer."""
        a = classify_and_understand_utterance(
            utterance="How does JWT authentication work?",
            session=self.session,
            source="MANUAL_INPUT",
            is_manual=True
        )
        self.assertTrue(a["should_generate_answer"])
        self.assertEqual(a["input_source"], "MANUAL_INPUT")
        self.assertNotIn(a["speaker"], ["Candidate", "Interviewer"])

    # ============================================================
    # TEST 10: CANDIDATE SPOKEN QUESTION ON MIC REMAINS SILENT
    # ============================================================
    def test_13_candidate_mic_spoken_question_remains_silent(self):
        """If candidate speaks a question into the mic in Dual mode, it is classified as Candidate and NEVER generates an answer."""
        a = classify_and_understand_utterance(
            utterance="How does JWT authentication work?",
            session=self.session,
            source="candidate_mic"
        )
        self.assertEqual(a["speaker"], "Candidate")
        self.assertFalse(a["should_generate_answer"])

    # ============================================================
    # TEST 11: TEXT HEURISTICS DO NOT OVERRIDE SYSTEM LOOPBACK
    # ============================================================
    def test_14_candidate_text_heuristics_do_not_override_system(self):
        """If system loopback contains 1st-person phrases, it remains INTERVIEWER in Dual mode."""
        a = classify_and_understand_utterance(
            utterance="In our team, how did you implement authentication?",
            session=self.session,
            source="system"
        )
        self.assertEqual(a["speaker"], "Interviewer")
        self.assertTrue(a["should_generate_answer"])

    # ============================================================
    # TEST 12: INTERVIEWER INTERRUPTION
    # ============================================================
    def test_15_interviewer_interruption(self):
        """Candidate speaking on mic followed by system question takes immediate priority."""
        # 1. Candidate speaking
        a_mic = classify_and_understand_utterance("I used Django because...", session=self.session, source="candidate_mic")
        self.assertEqual(a_mic["speaker"], "Candidate")
        self.assertFalse(a_mic["should_generate_answer"])

        # 2. Interviewer interrupts on system loopback
        a_sys = classify_and_understand_utterance("Why didn't you use Flask?", session=self.session, source="system")
        self.assertEqual(a_sys["speaker"], "Interviewer")
        self.assertTrue(a_sys["should_generate_answer"])


if __name__ == "__main__":
    unittest.main()
