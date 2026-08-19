

Pasted code.py
Python
there are many indentation error write the whole main.py clearly

Yes, bro. This output shows the same problem clearly.

The first half is a good answer to “What is Django?”. But the section:

“Why I chose Django for my project”

contains unsupported personal claims such as registration, login, profile management, authentication system usage, product models, RESTful endpoints, and form validation.

Those aren't currently confirmed by the facts you gave me.

What this tells us
Your system is currently doing this:

Question: What is Django?
        ↓
General Django knowledge
        ↓
Correct technical explanation
        ↓
Then it sees Farmer Marketplace context
        ↓
It tries to connect Django to the project
        ↓
It invents implementation details
For “What is Django?”, it should not automatically add a long project-specific section.

It should answer Django generally, and only add your project experience if the question asks for it or if a very brief connection is genuinely useful.

So our next instruction should be
Add this rule to your system_prompt, preferably under TECHNICAL QUESTIONS:

IMPORTANT TECHNICAL QUESTION RULE:

When the interviewer asks a general technical question such as:

"What is Django?"
"What is React?"
"What is MySQL?"
"What is RAG?"

Answer the technical question using general technical
knowledge.

Do NOT automatically create a "Why I chose it", "How I used it",
or project-specific section unless the question specifically
asks about the candidate's project or experience.

If you mention the candidate's project, only use facts explicitly
confirmed in CANDIDATE INFORMATION.

Never invent project implementation details just to connect a
general technical explanation to the candidate's project.

For example, if the question is:

"What is Django?"

Do not automatically say:

"I used Django for authentication..."
"I used Django ORM..."
"I created REST APIs..."
"I used Django admin..."

unless those exact experiences are confirmed.

Instead, explain Django as a technical concept.

If the question is:

"How did you use Django in your Farmer Marketplace?"

then use the confirmed candidate information and answer in
first person.
This gives us the behavior we actually want
Question:

What is Django?

➡️ General Django explanation.

Question:

How did you use Django?

➡️ Your confirmed Django experience.

Question:

Why did you choose Django?

➡️ Only your confirmed reason for choosing Django. If you haven't provided that reason, it must say it doesn't have that information.

Question:

Explain your Farmer Marketplace.

➡️ Project information + your confirmed role.

This is the distinction we were missing. Don't change the model. Don't rewrite the whole ask_ai() yet. Add this rule, test these four questions, and then we'll move to the next instruction.


Answer**

RAG stands for Retrieval‑Augmented Generation, a pattern that combines a large language model (LLM) with an external information‑retrieval component. The core idea is to let the generative model “look up” relevant facts or documents at inference time, rather than relying solely on the knowledge that was baked into its parameters during pre‑training. This approach helps mitigate two common limitations of pure LLMs: (1) outdated or incomplete knowledge, and (2) the tendency to hallucinate details that aren’t supported by any source.

How it works

Query formulation – When a user asks a question or provides a prompt, the system first extracts a concise query that captures the information need.
Retrieval – That query is sent to a search engine or vector‑based similarity store (e.g., Elasticsearch, FAISS, or a dense passage retriever). The retrieval module returns a set of top‑k documents or passages that are most relevant.
Context construction – The retrieved snippets are concatenated with the original prompt, often with a short instruction reminding the model to ground its answer in the provided sources.
Generation – The combined input is fed to the LLM, which produces a response that explicitly incorporates the retrieved content. Some implementations also use a post‑processing step to cite sources or verify consistency.
Why it’s useful

Up‑to‑date knowledge – Because the retrieval component can query a live knowledge base, the system can answer questions about recent events or domain‑specific data that the LLM never saw during training.
Improved factual accuracy – By grounding the generation in concrete documents, the likelihood of hallucinations drops dramatically, which is crucial for applications like medical advice, legal assistance, or customer support.
Scalability – You can keep the LLM fixed while simply expanding or updating the underlying corpus, avoiding costly re‑training.
Typical use cases

Open‑domain question answering where the answer must be supported by citations.
Enterprise knowledge‑base assistants that pull internal documents, manuals, or policy files to help employees.
Chatbots for product support that need to reference up‑to‑date specifications or troubleshooting guides.
My experience

In my AI/ML background I have worked with Retrieval‑Augmented Generation concepts, especially in the context of natural‑language processing and large language models. I understand how to set up the retrieval pipeline, choose appropriate vector encoders, and design prompts that encourage the model to cite the retrieved material. This knowledge allows me to build systems that combine the expressive power of LLMs with the precision of a searchable knowledge source, delivering answers that are both fluent and trustworthy.


Yes, bro. This output exposes another specific issue.

The general explanation of RAG is fine, but the last section says:

“In my AI/ML background I have worked with Retrieval-Augmented Generation concepts…”

and then claims:

you set up retrieval pipelines

chose vector encoders

designed prompts

built systems

used citations

Those are not confirmed by the candidate information you gave me.

You have only confirmed:

AI/ML:
Machine Learning, NLP, RAG, LLMs
That means the model can say you have RAG knowledge, but it cannot invent specific RAG implementation experience.

Add this rule to the prompt
Put this immediately after the technical-question rule we added:

IMPORTANT GENERAL KNOWLEDGE VS EXPERIENCE RULE:

A technology appearing in the candidate's Skills or AI/ML
section means that the candidate has listed knowledge or
familiarity with that technology.

It does NOT automatically mean that the candidate has
implemented every feature, architecture, algorithm, tool,
library, or technique associated with that technology.

For example:

If the candidate information says:

AI/ML:
Machine Learning, NLP, RAG, LLMs

You may say:

"I have knowledge of RAG and understand how retrieval and
generation work together."

You MUST NOT automatically say:

"I built a RAG pipeline."
"I used FAISS."
"I created vector embeddings."
"I implemented semantic search."
"I designed a retrieval system."

unless those specific experiences are explicitly confirmed.

When answering a general technical question, explain the
technology normally using general technical knowledge.

When describing the candidate's experience, use ONLY the
specific experience confirmed in the candidate information.
Then your RAG answer should behave like this
For:

What is RAG?

The AI can give you a detailed technical answer:

RAG stands for Retrieval-Augmented Generation. It is a technique that combines information retrieval with a large language model. Instead of relying only on the knowledge stored in the model, the system first retrieves relevant information from an external knowledge source and provides that information to the model as context. The model then uses that context to generate the answer. A typical RAG pipeline consists of retrieving relevant documents, adding them to the prompt, and generating a response based on that context. RAG is useful when an application needs to work with domain-specific or frequently changing information. I have knowledge of RAG as part of my AI/ML background, and I understand its role in combining retrieval with LLM-based generation.

That last sentence is supported by your information.

But it should not suddenly claim:

"I built a FAISS vector database..."

unless you actually add that to your knowledge file.

So we now have 3 separate rules
1. General technical question

What is Django?

→ Explain Django. Don't invent project experience.

2. Technology + candidate experience

How did you use Django?

→ Use confirmed Django/project facts and speak in first person.

3. Technology listed as a skill

What is RAG?

→ Explain RAG technically. You can say you have knowledge of RAG, but don't invent implementation experience.

This is the behavior we want. We don't need to change the model. We need to keep tightening the distinction between knowledge, confirmed experience, and general technical knowledge.


before going to next can you make readme.md 

Done, bro. I created the README.md based on the architecture and rules we've established so far.


It covers the current structure, RAG flow, candidate-fact rules, AI answer behavior, technologies, setup, and future development. 


okay so i 

AI Interview Assistant
An AI-powered interview practice assistant that listens to interview
questions, converts speech to text, retrieves relevant candidate
information, and generates natural interview-style answers.

Overview
The project is designed to help a candidate practice answering
technical, project, coding, HR, and follow-up interview questions.

The system combines:

Speech-to-text

Retrieval-Augmented Generation (RAG)

Candidate knowledge files

LLM-based answer generation

Conversation history

A Tkinter desktop interface

Groq API

Current Architecture
Interviewer / User
       |
       v
Audio or Typed Question
       |
       v
Speech-to-Text
(Whisper)
       |
       v
Question Text
       |
       v
RAG / Knowledge Retrieval
       |
       v
Relevant Candidate Information
       |
       v
LLM
       |
       v
Interview-Style Answer
       |
       v
Tkinter UI
Main Components
1. main.py
The main application controller.

Responsibilities:

Load environment variables

Initialize the Groq client

Maintain conversation history

Process typed questions

Process speech questions

Retrieve candidate information

Send prompts to the LLM

Stream generated answers to the UI

2. ui.py
Contains the Tkinter user interface.

The interface includes:

Application title

Status indicator

Question display

AI answer display

Voice input

Typed question input

3. speech.py
Responsible for speech processing.

The current system uses speech-to-text to convert recorded audio into
text before sending the question to the LLM.

4. audio.py
Responsible for computer/system audio recording.

The project is being developed to support capturing audio from the
computer so that spoken interview questions can be converted into text.

5. rag.py
Responsible for retrieving relevant information from the candidate's
knowledge files.

The current implementation uses:

TF-IDF

Cosine similarity

Text chunks

Top-k retrieval

The retrieved information is passed to the LLM as candidate context.

6. knowledge/
Contains candidate-specific .txt files.

Example information includes:

Education

Programming languages

Frameworks

Databases

AI/ML knowledge

Cloud knowledge

Projects

Confirmed project contributions

The knowledge files are treated as the source of truth for
candidate-specific claims.

Candidate Knowledge Rules
A core design principle of this project is:

The AI must never invent candidate experience.

The model may explain general technical concepts using its general
knowledge, but candidate-specific claims must be supported by the
candidate knowledge.

For example, if the knowledge says:

I used Django for the backend of the Farmer Marketplace,
particularly for product listing and user-related functionality.
The assistant may explain that experience naturally.

However, it should not automatically claim that the candidate:

Used Django ORM

Built REST APIs

Implemented JWT authentication

Used Django Admin

Implemented permissions

Used a specific Django package

unless those facts are explicitly added to the candidate knowledge.

Knowledge vs Experience
The system distinguishes between general technical knowledge and
confirmed experience.

For example:

AI/ML:
Machine Learning, NLP, RAG, LLMs
This confirms that these technologies are part of the candidate's stated
knowledge.

It does not automatically confirm that the candidate:

Built a RAG pipeline

Used FAISS

Created embeddings

Implemented vector search

Built a production LLM application

Those details must be explicitly confirmed before the assistant presents
them as personal experience.

Answer Types
General Technical Questions
Example:

What is Django?
The assistant should explain Django as a technical concept.

It should not automatically create a project-specific explanation unless
the question asks about the candidate's experience.

Candidate Experience Questions
Example:

How did you use Django in the Farmer Marketplace?
The assistant should answer in first person and use only confirmed
project information.

Project Questions
Example:

Explain your Farmer Marketplace project.
The assistant should use confirmed information about:

Project purpose

Technologies

Features

Candidate contribution

Missing information should not be invented.

Why Questions
Example:

Why did you choose Django?
The assistant should only provide a personal reason if that reason has
been explicitly added to the candidate knowledge.

If the reason is not known, the assistant should not invent one.

Coding Questions
The assistant should:

Explain the approach.

Provide clean code.

Explain important parts.

Include complexity when useful.

HR / Behavioral Questions
Answers should be:

Natural

Conversational

Based on confirmed candidate information

Free from invented achievements or experience

Follow-up Questions
The assistant uses conversation history to understand what the
interviewer is referring to while still following the candidate-facts
rule.

Technologies
Current project technologies include:

Python

Tkinter

Groq API

Whisper

TF-IDF

Cosine Similarity

RAG

LLMs

React / Django / MySQL as technologies represented in candidate
knowledge

API Configuration
Create a .env file in the project root:

GROQ_API_KEY=your_groq_api_key
Do not commit the .env file to Git.

A .gitignore file should include:

.env
.venv/
__pycache__/
*.pyc
Installation
Create and activate a Python virtual environment:

python -m venv .venv
Windows:

.venv\Scripts\activate
Install the required dependencies used by the project.

Example:

pip install groq python-dotenv scikit-learn soundcard soundfile
Additional dependencies may be required depending on the current audio
implementation.

Running the Application
After configuring the environment:

python main.py
The Tkinter application should open.

From the interface, the user can:

Type an interview question.

Use voice input.

Convert speech to text.

Retrieve relevant candidate information.

Generate an AI answer.

View the answer in the UI.

RAG Flow
The current retrieval process works approximately as follows:

Candidate .txt files
        |
        v
Load documents
        |
        v
Split into text sections
        |
        v
TF-IDF vectorization
        |
        v
Compare question with sections
        |
        v
Cosine similarity
        |
        v
Top relevant sections
        |
        v
LLM context
The purpose of retrieval is to provide the LLM with the candidate
information that is most relevant to the current question.

Conversation Memory
The application maintains recent conversation history.

This allows follow-up questions to retain context.

The current implementation limits stored conversation history to recent
messages rather than keeping the entire conversation indefinitely.

Important Design Principle
The assistant should be good at:

Turning confirmed candidate facts into natural interview answers.

It should not be good at:

Making the candidate appear more experienced by inventing details.

Therefore:

Confirmed fact
     |
     v
Retrieve
     |
     v
Explain naturally
not:

Missing fact
     |
     v
LLM assumption
     |
     v
Present assumption as candidate experience
Current Development Status
Implemented / working:

Tkinter desktop UI

Typed question input

Voice input

Speech-to-text

Groq LLM integration

Streaming AI responses

Candidate knowledge retrieval

TF-IDF-based RAG

Conversation history

Candidate-specific answer rules

System audio recording development

Future Improvements
Potential future improvements include:

Better audio capture and handling

Improved retrieval and document chunking

Better candidate knowledge organization

Stronger question classification

Improved follow-up question handling

Better answer formatting

More reliable context selection

Testing across technical, project, coding, HR, and behavioral
questions

Automated evaluation of factual consistency

Project Philosophy
The goal is not simply to generate impressive answers.

The goal is to generate:

Accurate answers

Natural answers

Speakable answers

Technically useful answers

Candidate-specific answers when supported

No fabricated candidate experience

Library
/
README.md


AI Interview Assistant
An AI-powered interview practice assistant that listens to interview
questions, converts speech to text, retrieves relevant candidate
information, and generates natural interview-style answers.

Overview
The project is designed to help a candidate practice answering
technical, project, coding, HR, and follow-up interview questions.

The system combines:

Speech-to-text

Retrieval-Augmented Generation (RAG)

Candidate knowledge files

LLM-based answer generation

Conversation history

A Tkinter desktop interface

Groq API

Current Architecture
Interviewer / User
       |
       v
Audio or Typed Question
       |
       v
Speech-to-Text
(Whisper)
       |
       v
Question Text
       |
       v
RAG / Knowledge Retrieval
       |
       v
Relevant Candidate Information
       |
       v
LLM
       |
       v
Interview-Style Answer
       |
       v
Tkinter UI
Main Components
1. main.py
The main application controller.

Responsibilities:

Load environment variables

Initialize the Groq client

Maintain conversation history

Process typed questions

Process speech questions

Retrieve candidate information

Send prompts to the LLM

Stream generated answers to the UI

2. ui.py
Contains the Tkinter user interface.

The interface includes:

Application title

Status indicator

Question display

AI answer display

Voice input

Typed question input

3. speech.py
Responsible for speech processing.

The current system uses speech-to-text to convert recorded audio into
text before sending the question to the LLM.

4. audio.py
Responsible for computer/system audio recording.

The project is being developed to support capturing audio from the
computer so that spoken interview questions can be converted into text.

5. rag.py
Responsible for retrieving relevant information from the candidate's
knowledge files.

The current implementation uses:

TF-IDF

Cosine similarity

Text chunks

Top-k retrieval

The retrieved information is passed to the LLM as candidate context.

6. knowledge/
Contains candidate-specific .txt files.

Example information includes:

Education

Programming languages

Frameworks

Databases

AI/ML knowledge

Cloud knowledge

Projects

Confirmed project contributions

The knowledge files are treated as the source of truth for
candidate-specific claims.

Candidate Knowledge Rules
A core design principle of this project is:

The AI must never invent candidate experience.

The model may explain general technical concepts using its general
knowledge, but candidate-specific claims must be supported by the
candidate knowledge.

For example, if the knowledge says:

I used Django for the backend of the Farmer Marketplace,
particularly for product listing and user-related functionality.
The assistant may explain that experience naturally.

However, it should not automatically claim that the candidate:

Used Django ORM

Built REST APIs

Implemented JWT authentication

Used Django Admin

Implemented permissions

Used a specific Django package

unless those facts are explicitly added to the candidate knowledge.

Knowledge vs Experience
The system distinguishes between general technical knowledge and
confirmed experience.

For example:

AI/ML:
Machine Learning, NLP, RAG, LLMs
This confirms that these technologies are part of the candidate's stated
knowledge.

It does not automatically confirm that the candidate:

Built a RAG pipeline

Used FAISS

Created embeddings

Implemented vector search

Built a production LLM application

Those details must be explicitly confirmed before the assistant presents
them as personal experience.

Answer Types
General Technical Questions
Example:

What is Django?
The assistant should explain Django as a technical concept.

It should not automatically create a project-specific explanation unless
the question asks about the candidate's experience.

Candidate Experience Questions
Example:

How did you use Django in the Farmer Marketplace?
The assistant should answer in first person and use only confirmed
project information.

Project Questions
Example:

Explain your Farmer Marketplace project.
The assistant should use confirmed information about:

Project purpose

Technologies

Features

Candidate contribution

Missing information should not be invented.

Why Questions
Example:

Why did you choose Django?
The assistant should only provide a personal reason if that reason has
been explicitly added to the candidate knowledge.

If the reason is not known, the assistant should not invent one.

Coding Questions
The assistant should:

Explain the approach.

Provide clean code.

Explain important parts.

Include complexity when useful.

HR / Behavioral Questions
Answers should be:

Natural

Conversational

Based on confirmed candidate information

Free from invented achievements or experience

Follow-up Questions
The assistant uses conversation history to understand what the
interviewer is referring to while still following the candidate-facts
rule.

Technologies
Current project technologies include:

Python

Tkinter

Groq API

Whisper

TF-IDF

Cosine Similarity

RAG

LLMs

React / Django / MySQL as technologies represented in candidate
knowledge

API Configuration
Create a .env file in the project root:

GROQ_API_KEY=your_groq_api_key
Do not commit the .env file to Git.

A .gitignore file should include:

.env
.venv/
__pycache__/
*.pyc
Installation
Create and activate a Python virtual environment:

python -m venv .venv
Windows:

.venv\Scripts\activate
Install the required dependencies used by the project.

Example:

pip install groq python-dotenv scikit-learn soundcard soundfile
Additional dependencies may be required depending on the current audio
implementation.

Running the Application
After configuring the environment:

python main.py
The Tkinter application should open.

From the interface, the user can:

Type an interview question.

Use voice input.

Convert speech to text.

Retrieve relevant candidate information.

Generate an AI answer.

View the answer in the UI.

RAG Flow
The current retrieval process works approximately as follows:

Candidate .txt files
        |
        v
Load documents
        |
        v
Split into text sections
        |
        v
TF-IDF vectorization
        |
        v
Compare question with sections
        |
        v
Cosine similarity
        |
        v
Top relevant sections
        |
        v
LLM context
The purpose of retrieval is to provide the LLM with the candidate
information that is most relevant to the current question.

Conversation Memory
The application maintains recent conversation history.

This allows follow-up questions to retain context.

The current implementation limits stored conversation history to recent
messages rather than keeping the entire conversation indefinitely.

Important Design Principle
The assistant should be good at:

Turning confirmed candidate facts into natural interview answers.

It should not be good at:

Making the candidate appear more experienced by inventing details.

Therefore:

Confirmed fact
     |
     v
Retrieve
     |
     v
Explain naturally
not:

Missing fact
     |
     v
LLM assumption
     |
     v
Present assumption as candidate experience
Current Development Status
Implemented / working:

Tkinter desktop UI

Typed question input

Voice input

Speech-to-text

Groq LLM integration

Streaming AI responses

Candidate knowledge retrieval

TF-IDF-based RAG

Conversation history

Candidate-specific answer rules

System audio recording development

Future Improvements
Potential future improvements include:

Better audio capture and handling

Improved retrieval and document chunking

Better candidate knowledge organization

Stronger question classification

Improved follow-up question handling

Better answer formatting

More reliable context selection

Testing across technical, project, coding, HR, and behavioral
questions

Automated evaluation of factual consistency

Project Philosophy
The goal is not simply to generate impressive answers.

The goal is to generate:

Accurate answers

Natural answers

Speakable answers

Technically useful answers

Candidate-specific answers when supported

No fabricated candidate experience

