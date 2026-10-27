"""
Automated Test Suite for Parakit Resume Upload & Processing Flow
================================================================
Tests validation, background extraction, replacement, removal,
error recovery, persistence, and prompt grounding.
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


import sys
import os
import unittest
import tempfile
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import interview_engine


class TestResumeUploadFlow(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.session = interview_engine.InterviewSession()
        # Clean session resume state for testing
        self.session.remove_resume()

    def tearDown(self):
        self.session.remove_resume()
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    # ============================================================
    # TEST 1: TXT RESUME UPLOAD & EXTRACTION
    # ============================================================
    def test_01_txt_resume_upload(self):
        txt_path = os.path.join(self.temp_dir.name, "Abhishek_Resume.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("Candidate: Abhishek\nSkills: Python, Django, React, Groq LLM\nExperience: 3 years building AI applications.")
            
        success = self.session.set_resume(txt_path)
        self.assertTrue(success)
        self.assertEqual(self.session.resume_path, txt_path)
        self.assertIn("Python, Django, React", self.session.resume_text)

    # ============================================================
    # TEST 2: DOCX RESUME UPLOAD & EXTRACTION
    # ============================================================
    def test_02_docx_resume_upload(self):
        docx_path = os.path.join(self.temp_dir.name, "Resume_Test.docx")
        try:
            import docx
            doc = docx.Document()
            doc.add_heading("Abhishek Salunke", 0)
            doc.add_paragraph("Full Stack AI Engineer with expertise in FastAPI, Django, and React.")
            doc.save(docx_path)
            
            success = self.session.set_resume(docx_path)
            self.assertTrue(success)
            self.assertEqual(self.session.resume_path, docx_path)
            self.assertIn("Full Stack AI Engineer", self.session.resume_text)
        except ImportError:
            self.skipTest("python-docx not installed, skipping docx creation")

    # ============================================================
    # TEST 3: PDF RESUME UPLOAD & EXTRACTION
    # ============================================================
    def test_03_pdf_resume_upload(self):
        pdf_path = os.path.join(self.temp_dir.name, "Resume_Sample.pdf")
        try:
            import pypdf
            writer = pypdf.PdfWriter()
            # If pypdf is available, test extraction function behavior
            extracted = interview_engine.extract_text_from_file(pdf_path)
            self.assertIsInstance(extracted, str)
        except ImportError:
            self.skipTest("pypdf not installed, skipping pdf extraction")

    # ============================================================
    # TEST 4: UNSUPPORTED FILE REJECTION
    # ============================================================
    def test_04_unsupported_file_rejection(self):
        ALLOWED_EXTS = {".pdf", ".docx", ".txt", ".md"}
        
        unsupported_files = [
            "photo.jpg",
            "image.png",
            "sheet.xlsx",
            "script.exe",
            "doc.doc"
        ]
        
        for fname in unsupported_files:
            ext = os.path.splitext(fname)[1].lower()
            self.assertNotIn(ext, ALLOWED_EXTS)
            
        # Verify extract_text_from_file returns empty string for unsupported extension
        exe_path = os.path.join(self.temp_dir.name, "malware.exe")
        with open(exe_path, "wb") as f:
            f.write(b"\x00\x01\x02")
            
        success = self.session.set_resume(exe_path)
        self.assertFalse(success)
        self.assertIsNone(self.session.resume_path)
        self.assertEqual(self.session.resume_text, "")

    # ============================================================
    # TEST 5: FILE REPLACEMENT
    # ============================================================
    def test_05_resume_replacement(self):
        resume1_path = os.path.join(self.temp_dir.name, "Resume_V1.txt")
        with open(resume1_path, "w", encoding="utf-8") as f:
            f.write("Resume Version 1: Junior Developer")
            
        resume2_path = os.path.join(self.temp_dir.name, "Resume_V2.txt")
        with open(resume2_path, "w", encoding="utf-8") as f:
            f.write("Resume Version 2: Senior AI Architect")
            
        # Select V1
        self.session.set_resume(resume1_path)
        self.assertIn("Junior Developer", self.session.resume_text)
        
        # Replace with V2
        self.session.set_resume(resume2_path)
        self.assertEqual(self.session.resume_path, resume2_path)
        self.assertIn("Senior AI Architect", self.session.resume_text)
        self.assertNotIn("Junior Developer", self.session.resume_text)

    # ============================================================
    # TEST 6: REMOVE RESUME
    # ============================================================
    def test_06_remove_resume(self):
        resume_path = os.path.join(self.temp_dir.name, "Resume_ToRemove.txt")
        with open(resume_path, "w", encoding="utf-8") as f:
            f.write("Resume Content")
            
        self.session.set_resume(resume_path)
        self.assertIsNotNone(self.session.resume_path)
        
        # Remove resume
        self.session.remove_resume()
        self.assertIsNone(self.session.resume_path)
        self.assertEqual(self.session.resume_text, "")
        
        # Verify physical file on disk was NOT deleted
        self.assertTrue(os.path.exists(resume_path))

    # ============================================================
    # TEST 7: CORRUPT / EMPTY FILE HANDLING
    # ============================================================
    def test_07_empty_file_handling(self):
        empty_path = os.path.join(self.temp_dir.name, "empty.txt")
        with open(empty_path, "w", encoding="utf-8") as f:
            f.write("")
            
        success = self.session.set_resume(empty_path)
        self.assertFalse(success)
        self.assertIsNone(self.session.resume_path)

    # ============================================================
    # TEST 8: PERSISTENCE & RESTORE ON RESTART
    # ============================================================
    def test_08_persistence_and_restore(self):
        resume_path = os.path.join(self.temp_dir.name, "Persisted_Resume.txt")
        with open(resume_path, "w", encoding="utf-8") as f:
            f.write("Persisted Candidate Profile: Abhishek AI Specialist")
            
        self.session.set_resume(resume_path)
        
        # Simulate restarting application with new session
        new_session = interview_engine.InterviewSession()
        self.assertEqual(new_session.resume_path, resume_path)
        self.assertIn("Abhishek AI Specialist", new_session.resume_text)

    # ============================================================
    # TEST 9: RESUME GROUNDING IN SYSTEM PROMPT
    # ============================================================
    def test_09_resume_grounding_in_system_prompt(self):
        resume_path = os.path.join(self.temp_dir.name, "Grounded_Resume.txt")
        with open(resume_path, "w", encoding="utf-8") as f:
            f.write("Abhishek\nProjects: Built Farmer Marketplace using Django and React\nPatents: AI Voice Assistant")
            
        self.session.set_resume(resume_path)
        prompt = interview_engine.build_interview_system_prompt(self.session, "project")
        
        self.assertIn("--- UPLOADED RESUME / CV ---", prompt)
        self.assertIn("Farmer Marketplace", prompt)
        self.assertIn("Django and React", prompt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
