"""
BluJay AI - Engine Utilities & Document Parsers
===============================================
Encapsulates config persistence, client caching, and document text extractors.
"""

import os
import json
from typing import Dict, Any, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

_CACHED_OPENAI_CLIENT = None
_CACHED_OPENAI_KEY = None


def get_openai_client():
    """Returns or initializes the persistent keep-alive client for OpenAI."""
    global _CACHED_OPENAI_CLIENT, _CACHED_OPENAI_KEY
    openai_key = os.getenv("OPENAI_API_KEY", "").strip().strip('"').strip("'")
    if openai_key:
        if _CACHED_OPENAI_CLIENT is None or _CACHED_OPENAI_KEY != openai_key:
            import openai
            _CACHED_OPENAI_CLIENT = openai.OpenAI(api_key=openai_key, max_retries=1, timeout=12.0)
            _CACHED_OPENAI_KEY = openai_key
        return _CACHED_OPENAI_CLIENT
    return None


def extract_text_from_file(file_path: str) -> str:
    """Extract plain text from PDF, DOCX, or TXT files."""
    if not file_path or not os.path.exists(file_path):
        return ""
    
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext in [".txt", ".md"]:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error reading text file {file_path}: {e}")
            return ""
            
    elif ext == ".pdf":
        try:
            import importlib
            pypdf_mod = importlib.import_module("pypdf")
            reader = pypdf_mod.PdfReader(file_path)
            pages = []
            for p in reader.pages:
                t = p.extract_text()
                if t:
                    pages.append(t)
            return "\n\n".join(pages).strip()
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
            return ""

    elif ext == ".docx":
        try:
            import importlib
            docx_mod = importlib.import_module("docx")
            doc = docx_mod.Document(file_path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n".join(paragraphs).strip()
        except Exception as e:
            print(f"Error reading DOCX {file_path}: {e}")
            return ""
            
    return ""


# Settings persistence
CONFIG_PATH = os.getenv("BLUJAY_CONFIG_PATH", os.path.expanduser("~/.blujay_config.json"))
LEGACY_CONFIG_PATH = os.getenv("PARAKIT_CONFIG_PATH", os.path.expanduser("~/.parakit_config.json"))


def load_user_config() -> Dict[str, Any]:
    """Loads persistent user preferences."""
    for path in (CONFIG_PATH, LEGACY_CONFIG_PATH):
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            print("Config load notice:", e)
    return {"auto_restore_on_question": True}


def save_user_config(cfg: Dict[str, Any]):
    """Saves user preferences to persistent storage."""
    try:
        data = load_user_config()
        data.update(cfg)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print("Config save notice:", e)
