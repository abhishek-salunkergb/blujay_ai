"""
=============================================================================
BluJay AI Dedicated Answer Formatter & Syntax-Highlighted Renderer
=============================================================================
Formats, normalizes, and renders raw AI answer streams into ultra-modern,
scannable, syntax-highlighted responses with Time/Space complexity badges,
clean bullet points, and code block containers.
=============================================================================
"""

import re
from typing import Dict, Any, List, Optional, Tuple


# Known section header names across all question strategies
KNOWN_SECTION_HEADERS = [
    "ANSWER", "APPROACH", "WHY", "KEY POINT", "KEY POINTS", "KEY",
    "SITUATION", "TASK", "ACTION", "RESULT",
    "HOW IT WORKED", "PROJECT OVERVIEW",
    "PROBLEM", "CAUSE", "FIX", "ROOT CAUSE",
    "SQL", "SQL QUERY", "EXPLANATION", "PERFORMANCE", "PERFORMANCE & INDEXING",
    "CODE", "SOLUTION", "COMPLEXITY", "TIME COMPLEXITY", "SPACE COMPLEXITY", "EDGE CASES",
    "REQUIREMENTS", "ARCHITECTURE", "COMPONENTS", "SCALABILITY", "TRADE-OFFS",
    "IN THIS PROJECT", "IN THIS CONTEXT", "OPTIMIZATION", "BEST PRACTICES"
]

HEADER_REGEX = re.compile(
    r"^(?:[\#\*\d\.\s\-\–]*)(?:🎯|💡|⚙️|💻|⏱️|💾|🛡️|🧪|🐛|🔍|⚡|🧠|📋|📊|🏛️|🧩|🔌|🗄️|📬|⚖️|📈|🔒|🌄)?\s*"
    r"(\b(?:" + "|".join(re.escape(h) for h in sorted(KNOWN_SECTION_HEADERS, key=len, reverse=True)) + r")\b)(?:\:|\*\*|\s*[-–])?",
    re.IGNORECASE
)

# Syntax Highlighting Keyword sets
PYTHON_KEYWORDS = {
    "def", "class", "return", "if", "else", "elif", "for", "in", "while",
    "break", "continue", "import", "from", "as", "try", "except", "finally",
    "with", "lambda", "self", "None", "True", "False", "yield", "async",
    "await", "global", "nonlocal", "pass", "raise", "assert", "del", "and",
    "or", "not", "is"
}

BUILTIN_FUNCS = {
    "len", "range", "enumerate", "min", "max", "sum", "zip", "dict",
    "list", "set", "tuple", "str", "int", "float", "bool", "print",
    "sorted", "map", "filter", "any", "all", "abs", "round", "reversed"
}


def parse_structured_answer(raw_answer: str, question_type: str = "general") -> Dict[str, Any]:
    """
    Parses a raw generated answer string into a structured internal representation.
    """
    if not raw_answer:
        return {
            "raw": "",
            "question_type": question_type,
            "summary": "",
            "sections": [],
            "code_blocks": [],
            "complexity": {"time": "", "space": ""}
        }

    lines = raw_answer.strip().split("\n")
    sections = []
    current_title = "ANSWER"
    current_lines = []
    code_blocks = []
    time_comp = ""
    space_comp = ""

    in_code_block = False
    current_code_lang = "python"
    current_code_lines = []

    for line in lines:
        stripped = line.strip()

        # Check for code fence
        if stripped.startswith("```"):
            if not in_code_block:
                in_code_block = True
                lang = stripped.lstrip("`").strip() or "python"
                current_code_lang = lang
                current_code_lines = []
            else:
                in_code_block = False
                code_text = "\n".join(current_code_lines)
                code_blocks.append({"language": current_code_lang, "code": code_text})
                if current_lines:
                    sections.append({
                        "title": current_title,
                        "content": "\n".join(current_lines).strip(),
                        "type": _detect_section_type(current_title, "\n".join(current_lines))
                    })
                    current_lines = []
                
                code_section_title = current_title if current_title in ["CODE", "FIX", "SOLUTION", "ALGORITHM", "SQL", "QUERY", "IMPLEMENTATION"] else "SOLUTION"
                sections.append({
                    "title": code_section_title,
                    "content": code_text,
                    "language": current_code_lang,
                    "type": "code"
                })
                current_title = "COMPLEXITY" if "CODING" in question_type.upper() else "EXPLANATION"
            continue

        if in_code_block:
            current_code_lines.append(line)
            continue

        # Check for Big-O Complexity extraction
        t_match = re.search(r"\bTime(?:\s*Complexity)?\s*:\s*(O\([^\)]+\)[^\n]*)", line, re.IGNORECASE)
        if t_match:
            time_comp = t_match.group(1).strip()
        s_match = re.search(r"\bSpace(?:\s*Complexity)?\s*:\s*(O\([^\)]+\)[^\n]*)", line, re.IGNORECASE)
        if s_match:
            space_comp = s_match.group(1).strip()

        # Check if line is a Section Header
        header_match = HEADER_REGEX.match(stripped)
        if header_match and len(stripped.split()) <= 4:
            matched_header = header_match.group(1).upper()
            if current_lines:
                sections.append({
                    "title": current_title,
                    "content": "\n".join(current_lines).strip(),
                    "type": _detect_section_type(current_title, "\n".join(current_lines))
                })
                current_lines = []
            current_title = matched_header
            remaining = stripped[header_match.end():].strip(" :*-–#")
            if remaining:
                current_lines.append(remaining)
            continue

        current_lines.append(line)

    if current_lines:
        sections.append({
            "title": current_title,
            "content": "\n".join(current_lines).strip(),
            "type": _detect_section_type(current_title, "\n".join(current_lines))
        })

    summary = ""
    for sec in sections:
        if sec["type"] != "code" and sec["content"]:
            summary = sec["content"].split("\n")[0]
            break

    return {
        "raw": raw_answer,
        "question_type": question_type,
        "summary": summary,
        "sections": sections,
        "code_blocks": code_blocks,
        "complexity": {"time": time_comp, "space": space_comp}
    }


def _detect_section_type(title: str, content: str) -> str:
    """Classifies a section into text, bullets, key_point, or code."""
    t_up = title.upper()
    if t_up in {"KEY", "KEY POINT", "KEY POINTS", "RESULT"}:
        return "key_point"
    if t_up in {"CODE", "SOLUTION", "SQL", "SQL QUERY", "FIX"}:
        return "code"
    
    lines = [l.strip() for l in content.split("\n") if l.strip()]
    if lines and any(l.startswith(("•", "-", "*", "Time:", "Space:", "Time Complexity:", "Space Complexity:")) for l in lines):
        return "bullets"
    return "text"


def highlight_code_tokens(code_str: str) -> List[Tuple[str, str]]:
    """
    Tokenizes code and assigns tags:
    ('code_kw', 'code_builtin', 'code_str', 'code_num', 'code_comment', 'code_body')
    """
    tokens = []
    lines = code_str.split("\n")

    for line_idx, line in enumerate(lines):
        # 1. Handle Comment
        comment_idx = line.find("#")
        comment_part = ""
        code_part = line
        if comment_idx != -1:
            code_part = line[:comment_idx]
            comment_part = line[comment_idx:]

        # 2. Tokenize code part with regex for strings, keywords, words, numbers, symbols
        pattern = re.compile(r'("(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|\b[A-Za-z_]\w*\b|\b\d+\b|[^\w\s]|\s+)')
        matches = pattern.findall(code_part)

        for token in matches:
            if not token:
                continue
            if token.startswith(('"', "'")):
                tokens.append((token, "code_str"))
            elif token in PYTHON_KEYWORDS:
                tokens.append((token, "code_kw"))
            elif token in BUILTIN_FUNCS:
                tokens.append((token, "code_builtin"))
            elif token.isdigit():
                tokens.append((token, "code_num"))
            elif token.isidentifier():
                tokens.append((token, "code_ident"))
            else:
                tokens.append((token, "code_body"))

        if comment_part:
            tokens.append((comment_part, "code_comment"))

        if line_idx < len(lines) - 1:
            tokens.append(("\n", "code_body"))

    return tokens


def format_interview_answer(raw_answer: str, question_type: str = "general") -> str:
    """
    Cleans and standardizes raw answer string, preserving natural paragraphs for conversational
    questions and structured code for technical problems.
    """
    if not raw_answer:
        return ""
    q_type = (question_type or "general").lower()
    cleaned = raw_answer.strip()
    
    # Conversational / HR / Resume / Follow-up: return pure natural text without synthetic headers
    if q_type in ("hr", "resume", "behavioral", "general", "tell_me_about_yourself", "follow_up"):
        cleaned = re.sub(r"^(?:[✦#\*\-\s]*ANSWER:?\s*\n+)", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"\n+(?:[✦#\*\-\s]*KEY(?:\s*POINT)?:?\s*\n+)", "\n\n", cleaned, flags=re.IGNORECASE).strip()
        return cleaned

    return cleaned


def render_structured_answer_to_text_widget(
    text_widget: Any,
    raw_answer: str,
    question_type: str = "general"
) -> None:
    """
    Renders a structured, styled interview answer with syntax-highlighted code blocks,
    bullet points (Time/Space complexity), and solution containers.
    For conversational / HR / Resume questions, renders natural spoken text without rigid section headers.
    """
    try:
        text_widget.delete("1.0", "end")
        q_type = (question_type or "general").lower()
        
        parsed = parse_structured_answer(raw_answer, question_type=q_type)
        sections = parsed.get("sections", [])
        code_blocks = parsed.get("code_blocks", [])
        
        # Pure conversational rendering if HR/Resume/Behavioral/General and no code
        if q_type in ("hr", "resume", "behavioral", "general", "tell_me_about_yourself") and not code_blocks:
            clean_text = format_interview_answer(raw_answer, question_type=q_type)
            text_widget.insert("end", clean_text, "body")
            text_widget.see("1.0")
            return
        
        if not sections:
            text_widget.insert("end", raw_answer, "body")
            text_widget.see("1.0")
            return

        for idx, sec in enumerate(sections):
            title = sec.get("title", "")
            content = sec.get("content", "").strip()
            sec_type = sec.get("type", "text")

            # 1. Complexity / Bullet Points (e.g. • Time: O(...), • Space: O(...))
            if sec_type == "bullets" or "COMPLEXITY" in title.upper():
                lines = content.split("\n")
                for l in lines:
                    l_clean = l.strip()
                    if not l_clean:
                        continue
                    if not l_clean.startswith(("•", "-", "*", "1.", "2.")):
                        l_clean = f"• {l_clean}"
                    else:
                        l_clean = re.sub(r"^[\-\*]\s*", "• ", l_clean)

                    # Highlight Time/Space in bold cyan/amber
                    if "Time:" in l_clean or "Time Complexity:" in l_clean:
                        text_widget.insert("end", "• ", "bullet_symbol")
                        text_widget.insert("end", "Time: ", "complexity_time_tag")
                        rem = re.sub(r"^•\s*Time(?:\s*Complexity)?:\s*", "", l_clean)
                        text_widget.insert("end", f"{rem}\n", "bullet_body")
                    elif "Space:" in l_clean or "Space Complexity:" in l_clean:
                        text_widget.insert("end", "• ", "bullet_symbol")
                        text_widget.insert("end", "Space: ", "complexity_space_tag")
                        rem = re.sub(r"^•\s*Space(?:\s*Complexity)?:\s*", "", l_clean)
                        text_widget.insert("end", f"{rem}\n", "bullet_body")
                    else:
                        text_widget.insert("end", f"{l_clean}\n", "bullet_body")
                text_widget.insert("end", "\n")
                continue

            # 2. Code / Solution Block
            if sec_type == "code" or title.upper() in {"CODE", "SOLUTION"}:
                text_widget.insert("end", "\n🌄 Solution:\n", "solution_header")
                tokens = highlight_code_tokens(content)
                text_widget.insert("end", "\n", "code_body")
                for text, tag in tokens:
                    text_widget.insert("end", text, tag)
                text_widget.insert("end", "\n\n", "code_body")
                continue

            # 3. Standard Section Headers (Skip for pure single-section or conversational answers)
            if title and not (title in ["ANSWER", "GENERAL"] and (len(sections) <= 2 or q_type in ["hr", "resume", "general"])):
                header_display = f"✦ {title.upper()}\n"
                tag_name = "key_header" if sec_type == "key_point" else "section_header"
                text_widget.insert("end", header_display, tag_name)

            # 4. Section Content
            if sec_type == "key_point":
                text_widget.insert("end", f"💡 {content}\n\n", "key_body")
            else:
                text_widget.insert("end", f"{content}\n\n", "body")

        text_widget.see("1.0")

    except Exception as e:
        try:
            text_widget.delete("1.0", "end")
            text_widget.insert("end", raw_answer, "body")
        except Exception:
            pass
