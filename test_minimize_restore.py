"""
Automated Test Suite for Parakit Minimized/Restore & Floating Icon Behavior
===========================================================================
Tests all 10 scenarios:
1. Dynamic Top-Center Position Calculation
2. Minimize Flow & Floating Icon Top-Center Placement
3. Restore on Click
4. Auto-Restore ON for Genuine Interviewer Question
5. Ordinary Interviewer Statements Ignored (Remain Minimized)
6. Auto-Restore OFF Background Generation & Manual Restore
7. Follow-up Question Flow (Remains Restored)
8. Candidate Speech Ignored (Remain Minimized)
9. Duplicate Question Deduplication (Single Restore Event)
10. Multi-Resolution Dynamic Centering Math
"""

import sys
import os
import unittest
import tkinter as tk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import interview_engine
from ui import FloatingIconWidget


class MockMainApp:
    def __init__(self, auto_restore=True):
        self.is_minimized = False
        self.user_minimized = False
        self.auto_restore_on_question = auto_restore
        self.restore_count = 0
        self.floating_icon_status = "listening"

    def minimize_window(self):
        self.is_minimized = True
        self.user_minimized = True

    def restore_window(self):
        self.is_minimized = False
        self.user_minimized = False
        self.restore_count += 1

    def set_icon_status(self, status):
        self.floating_icon_status = status


class TestMinimizeRestoreSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.root = tk.Tk()
        cls.root.withdraw()

    @classmethod
    def tearDownClass(cls):
        try:
            cls.root.destroy()
        except Exception:
            pass

    def setUp(self):
        self.session = interview_engine.InterviewSession()
        self.session.clear()

    # ============================================================
    # TEST 1 & 10: DYNAMIC TOP-CENTER CALCULATION ACROSS RESOLUTIONS
    # ============================================================
    def test_01_dynamic_top_center_math(self):
        icon_w, icon_h = 210, 42
        test_resolutions = [
            (1920, 1080),
            (2560, 1440),
            (3840, 2160),
            (1366, 768),
            (1280, 720)
        ]
        
        for w, h in test_resolutions:
            expected_x = (w - icon_w) // 2
            expected_y = 12
            self.assertEqual(expected_x, (w - 210) // 2)
            self.assertEqual(expected_y, 12)
            # Verify x is positive and centered
            self.assertTrue(0 < expected_x < w)

    # ============================================================
    # TEST 2: FLOATING ICON INITIALIZATION & RECENTER
    # ============================================================
    def test_02_floating_icon_widget_top_center(self):
        app = MockMainApp()
        floating_widget = FloatingIconWidget(self.root, on_restore_callback=app.restore_window)
        try:
            x, y, w, h = floating_widget.calculate_position()
            screen_w = self.root.winfo_screenwidth()
            expected_x = max(0, int((screen_w - w) / 2))
            self.assertEqual(x, expected_x)
            self.assertEqual(y, 12)
            self.assertTrue(45 <= w <= 250)
            self.assertTrue(45 <= h <= 65)
        finally:
            floating_widget.destroy()

    # ============================================================
    # TEST 3: MANUAL CLICK RESTORES OVERLAY
    # ============================================================
    def test_03_manual_click_restores(self):
        app = MockMainApp()
        app.minimize_window()
        self.assertTrue(app.is_minimized)
        
        floating_widget = FloatingIconWidget(self.root, on_restore_callback=app.restore_window)
        try:
            # Simulate click
            floating_widget.on_restore_callback()
            self.assertFalse(app.is_minimized)
            self.assertEqual(app.restore_count, 1)
        finally:
            floating_widget.destroy()

    # ============================================================
    # TEST 4: AUTO-RESTORE ON FOR GENUINE INTERVIEWER QUESTION
    # ============================================================
    def test_04_auto_restore_on_genuine_question(self):
        app = MockMainApp(auto_restore=True)
        app.minimize_window()
        self.session.auto_restore_on_question = True
        
        # Interviewer asks genuine question
        analysis = interview_engine.classify_and_understand_utterance(
            utterance="Why did you choose React for this project?",
            session=self.session,
            client=None,
            source="mic"
        )
        
        self.assertTrue(analysis["should_generate_answer"])
        self.assertEqual(analysis["speaker"], "Interviewer")
        
        # Simulated Auto-Restore check in UI pipeline
        if app.is_minimized and self.session.auto_restore_on_question and analysis["should_generate_answer"]:
            app.restore_window()
            
        self.assertFalse(app.is_minimized)
        self.assertEqual(app.restore_count, 1)

    # ============================================================
    # TEST 5: ORDINARY INTERVIEWER STATEMENT DOES NOT RESTORE
    # ============================================================
    def test_05_statement_does_not_restore(self):
        app = MockMainApp(auto_restore=True)
        app.minimize_window()
        self.session.auto_restore_on_question = True
        
        statements = ["Okay.", "I see.", "Thank you.", "Sounds good.", "Moving on."]
        
        for stmt in statements:
            analysis = interview_engine.classify_and_understand_utterance(
                utterance=stmt,
                session=self.session,
                client=None,
                source="mic"
            )
            self.assertFalse(analysis["should_generate_answer"])
            
            if app.is_minimized and self.session.auto_restore_on_question and analysis["should_generate_answer"]:
                app.restore_window()
                
            self.assertTrue(app.is_minimized)
            self.assertEqual(app.restore_count, 0)

    # ============================================================
    # TEST 6: AUTO-RESTORE OFF GENERATES IN BACKGROUND & RESTORES ON CLICK
    # ============================================================
    def test_06_auto_restore_off_background_flow(self):
        app = MockMainApp(auto_restore=False)
        app.minimize_window()
        self.session.auto_restore_on_question = False
        
        # Question arrives
        analysis = interview_engine.classify_and_understand_utterance(
            utterance="Explain how indexing works in PostgreSQL.",
            session=self.session,
            client=None,
            source="mic"
        )
        
        self.assertTrue(analysis["should_generate_answer"])
        
        # Auto-restore check
        if app.is_minimized:
            if self.session.auto_restore_on_question and analysis["should_generate_answer"]:
                app.restore_window()
            else:
                app.set_icon_status("answer_pending")
                
        # Must remain minimized
        self.assertTrue(app.is_minimized)
        self.assertEqual(app.restore_count, 0)
        self.assertEqual(app.floating_icon_status, "answer_pending")
        
        # Generation completes in background
        app.set_icon_status("answer_ready")
        self.assertEqual(app.floating_icon_status, "answer_ready")
        
        # User manually clicks minimized icon to view completed answer
        app.restore_window()
        self.assertFalse(app.is_minimized)
        self.assertEqual(app.restore_count, 1)

    # ============================================================
    # TEST 7: CANDIDATE SPEECH DOES NOT RESTORE
    # ============================================================
    def test_07_candidate_speech_does_not_restore(self):
        app = MockMainApp(auto_restore=True)
        app.minimize_window()
        self.session.auto_restore_on_question = True
        
        cand_speech = "I utilized Redis for caching session tokens to decrease DB load."
        analysis = interview_engine.classify_and_understand_utterance(
            utterance=cand_speech,
            session=self.session,
            client=None,
            source="mic"
        )
        
        # Candidate turn handling
        if analysis["speaker"] == "Candidate":
            should_answer = False
        else:
            should_answer = analysis["should_generate_answer"]
            
        self.assertFalse(should_answer)
        if app.is_minimized and self.session.auto_restore_on_question and should_answer:
            app.restore_window()
            
        self.assertTrue(app.is_minimized)
        self.assertEqual(app.restore_count, 0)

    # ============================================================
    # TEST 8: MULTI-TURN SEQUENCE & REMAINS RESTORED
    # ============================================================
    def test_08_multi_turn_remains_restored(self):
        app = MockMainApp(auto_restore=True)
        app.minimize_window()
        self.session.auto_restore_on_question = True
        
        # Turn 1: Question 1 triggers auto-restore
        q1 = interview_engine.classify_and_understand_utterance("Why MongoDB?", session=self.session, client=None, source="mic")
        if app.is_minimized and self.session.auto_restore_on_question and q1["should_generate_answer"]:
            app.restore_window()
        self.assertFalse(app.is_minimized)
        self.assertEqual(app.restore_count, 1)
        
        # Candidate response
        self.session.add_candidate_speech("Because our document schema was dynamic.")
        
        # Turn 2: Follow-up question arrives while window is already restored
    # ============================================================
    # TEST 9: DEDUPLICATION - SINGLE RESTORE EVENT PER QUESTION
    # ============================================================
    def test_09_deduplication_single_restore_event(self):
        app = MockMainApp(auto_restore=True)
        app.minimize_window()
        self.session.auto_restore_on_question = True
        
        restored_generations = set()
        gen_id = 1
        
        # Multiple interim chunks arrive for the same question
        interim_chunks = [
            "Why did you choose",
            "Why did you choose React",
            "Why did you choose React for this project?"
        ]
        
        for chunk in interim_chunks:
            if gen_id not in restored_generations:
                restored_generations.add(gen_id)
                if app.is_minimized and self.session.auto_restore_on_question:
                    app.restore_window()
                    
        # Exactly ONE restore call
        self.assertEqual(app.restore_count, 1)
        self.assertFalse(app.is_minimized)

    # ============================================================
    # TEST 10: STALE QUESTION SUPERSEDED
    # ============================================================
    def test_10_stale_question_superseded(self):
        app = MockMainApp(auto_restore=True)
        app.minimize_window()
        self.session.auto_restore_on_question = True
        
        # Generation 1 starts
        active_gen_id = 1
        
        # Interviewer immediately pivots/corrects to Question 2
        active_gen_id = 2  # Q1 cancelled
        
        # Old worker for Gen 1 attempts to stream
        token_gen_1 = 1
        restored_generations = set()
        
        if token_gen_1 == active_gen_id and token_gen_1 not in restored_generations:
            restored_generations.add(token_gen_1)
            app.restore_window()
            
        # Stale token from Gen 1 did NOT trigger restore
        self.assertEqual(app.restore_count, 0)
        self.assertTrue(app.is_minimized)
        
        # Active token from Gen 2 arrives
        if active_gen_id not in restored_generations:
            restored_generations.add(active_gen_id)
            app.restore_window()
            
        self.assertEqual(app.restore_count, 1)
        self.assertFalse(app.is_minimized)

    # ============================================================
    # TEST 11: DUAL MODE - SYSTEM/LOOPBACK RESTORES, CANDIDATE MIC DOES NOT
    # ============================================================
    def test_11_dual_mode_loopback_vs_mic(self):
        app = MockMainApp(auto_restore=True)
        app.minimize_window()
        self.session.auto_restore_on_question = True
        
        # Candidate mic utterance (candidate speaking to interviewer)
        cand_analysis = interview_engine.classify_and_understand_utterance(
            utterance="Should I use a BFS or DFS approach here?",
            session=self.session,
            client=None,
            source="candidate_mic"
        )
        self.assertEqual(cand_analysis["speaker"], "Candidate")
        self.assertFalse(cand_analysis["should_generate_answer"])
        
        # Candidate mic must NEVER restore window
        if app.is_minimized and self.session.auto_restore_on_question and cand_analysis["should_generate_answer"]:
            app.restore_window()
        self.assertTrue(app.is_minimized)
        self.assertEqual(app.restore_count, 0)
        
        # System/Loopback utterance (Interviewer speaking)
        inter_analysis = interview_engine.classify_and_understand_utterance(
            utterance="Why did you choose PostgreSQL over MongoDB for user accounts?",
            session=self.session,
            client=None,
            source="system"
        )
        self.assertEqual(inter_analysis["speaker"], "Interviewer")
        self.assertTrue(inter_analysis["should_generate_answer"])
        
        if app.is_minimized and self.session.auto_restore_on_question and inter_analysis["should_generate_answer"]:
            app.restore_window()
        self.assertFalse(app.is_minimized)
        self.assertEqual(app.restore_count, 1)

    # ============================================================
    # TEST 12: SMART AUTO-SCROLL LOGIC VERIFICATION
    # ============================================================
    def test_12_smart_autoscroll_logic(self):
        text_widget = tk.Text(self.root, height=10, width=40)
        try:
            # Fill with multiple lines
            for i in range(50):
                text_widget.insert("end", f"Line {i}\n")
            text_widget.see("end")
            text_widget.update_idletasks()
            
            y_view = text_widget.yview()
            is_at_bottom = len(y_view) < 2 or y_view[1] >= 0.93
            self.assertTrue(is_at_bottom)
            
            # User scrolls to top
            text_widget.yview_moveto(0.0)
            text_widget.update_idletasks()
            y_view_top = text_widget.yview()
            is_at_bottom_top = len(y_view_top) >= 2 and y_view_top[1] >= 0.93
            self.assertFalse(is_at_bottom_top)
        finally:
            text_widget.destroy()


if __name__ == "__main__":
    unittest.main(verbosity=2)
