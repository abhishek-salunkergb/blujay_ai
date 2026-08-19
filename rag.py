import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# KNOWLEDGE FOLDER & IN-MEMORY CACHE
# ============================================================

KNOWLEDGE_FOLDER = "knowledge"
_DOCUMENTS_CACHE = []
_CHUNKS_CACHE = []
_LAST_MTIME = 0.0


# ============================================================
# LOAD KNOWLEDGE DOCUMENTS (WITH IN-MEMORY CACHING)
# ============================================================

def load_documents():
    global _DOCUMENTS_CACHE, _CHUNKS_CACHE, _LAST_MTIME

    if not os.path.exists(KNOWLEDGE_FOLDER):
        return []

    try:
        current_mtime = os.path.getmtime(KNOWLEDGE_FOLDER)
        if _DOCUMENTS_CACHE and current_mtime == _LAST_MTIME:
            return _DOCUMENTS_CACHE
        _LAST_MTIME = current_mtime
    except Exception:
        pass

    documents = []

    for filename in os.listdir(KNOWLEDGE_FOLDER):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in [".txt", ".md", ".pdf", ".docx"]:
            continue

        path = os.path.join(
            KNOWLEDGE_FOLDER,
            filename
        )

        text = ""
        try:
            if ext in [".txt", ".md"]:
                with open(
                    path,
                    "r",
                    encoding="utf-8",
                    errors="ignore"
                ) as file:
                    text = file.read().strip()
            elif ext == ".pdf":
                try:
                    import importlib
                    pypdf_mod = importlib.import_module("pypdf")
                    reader = pypdf_mod.PdfReader(path)
                    pages = [p.extract_text() for p in reader.pages if p.extract_text()]
                    text = "\n\n".join(pages).strip()
                except Exception as pe:
                    print(f"Error reading PDF {filename}: {pe}")
            elif ext == ".docx":
                try:
                    import importlib
                    docx_mod = importlib.import_module("docx")
                    doc = docx_mod.Document(path)
                    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
                    text = "\n".join(paragraphs).strip()
                except Exception as de:
                    print(f"Error reading DOCX {filename}: {de}")

        except Exception as e:
            print(
                f"Error reading {filename}: {e}"
            )
            continue

        if text:
            documents.append(
                {
                    "filename": filename,
                    "text": text
                }
            )

    _DOCUMENTS_CACHE = documents
    return documents


# ============================================================
# CREATE TEXT CHUNKS
# ============================================================

def create_chunks(text):

    chunks = []

    sections = text.split("\n\n")

    for section in sections:

        section = section.strip()

        if section:

            chunks.append(section)

    return chunks


# ============================================================
# SEARCH KNOWLEDGE
#
# IMPORTANT:
#
# We still use TF-IDF for relevance.
#
# BUT we ALSO return the complete verified candidate
# information.
#
# This prevents an important candidate fact from being
# lost simply because TF-IDF gave it a low similarity score.
# ============================================================

def search_knowledge(
    question,
    conversation_history=None,
    top_k=8
):

    documents = load_documents()

    if not documents:

        return ""

    # --------------------------------------------------------
    # BUILD CHUNKS FOR RELEVANCE SEARCH
    # --------------------------------------------------------

    chunks = []

    for document in documents:

        sections = create_chunks(
            document["text"]
        )

        for section in sections:

            chunks.append(
                {
                    "filename": document["filename"],
                    "text": section
                }
            )

    if not chunks:

        return ""

    # --------------------------------------------------------
    # BUILD CONTEXT FROM PREVIOUS CONVERSATION
    #
    # This helps with questions such as:
    #
    # "What about the database?"
    # "How did you implement that?"
    # "Why did you choose it?"
    # --------------------------------------------------------

    history_text = ""

    if conversation_history:

        recent_messages = conversation_history[-6:]

        history_parts = []

        for message in recent_messages:

            content = message.get(
                "content",
                ""
            )

            if content:

                history_parts.append(
                    content
                )

        history_text = "\n".join(
            history_parts
        )

    # --------------------------------------------------------
    # CREATE RETRIEVAL QUERY
    # --------------------------------------------------------

    retrieval_query = f"""
Current interview question:

{question}

Previous conversation:

{history_text}
"""

    # --------------------------------------------------------
    # TF-IDF SEARCH
    # --------------------------------------------------------

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    all_texts = texts + [
        retrieval_query
    ]

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )

        vectors = vectorizer.fit_transform(
            all_texts
        )

        similarities = cosine_similarity(
            vectors[-1],
            vectors[:-1]
        )[0]

        ranked_indexes = similarities.argsort()[::-1]

    except Exception as e:

        print(
            f"TF-IDF search error: {e}"
        )

        ranked_indexes = []

    # --------------------------------------------------------
    # GET MOST RELEVANT CHUNKS
    # --------------------------------------------------------

    relevant_results = []

    for index in ranked_indexes[:top_k]:

        if similarities[index] <= 0:

            continue

        relevant_results.append(
            f"""
--- {chunks[index]["filename"]} ---

{chunks[index]["text"]}
"""
        )

    # --------------------------------------------------------
    # COMPLETE VERIFIED INFORMATION
    #
    # THIS IS THE IMPORTANT PART.
    #
    # The LLM always receives the complete candidate
    # information, regardless of TF-IDF ranking.
    # --------------------------------------------------------

    verified_information = []

    for document in documents:

        verified_information.append(
            f"""
--- {document["filename"]} ---

{document["text"]}
"""
        )

    # --------------------------------------------------------
    # FINAL CONTEXT
    # --------------------------------------------------------

    final_context = f"""
==================================================
VERIFIED CANDIDATE INFORMATION
==================================================

{"".join(verified_information)}

==================================================
MOST RELEVANT INFORMATION FOR THIS QUESTION
==================================================

{"".join(relevant_results)}
"""

    return final_context