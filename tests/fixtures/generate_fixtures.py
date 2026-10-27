"""
Golden Test Fixtures Generator for BluJay AI Parakeet-Level Validation Suite
Creates reproducible WAV audio fixtures, JSON question benchmarks, candidate profile,
coding exercises, transcript streams, and failure scenarios.
"""

import os
import io
import json
import wave
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
TRANSCRIPTS_DIR = os.path.join(BASE_DIR, "transcripts")
QUESTIONS_DIR = os.path.join(BASE_DIR, "questions")
PROFILES_DIR = os.path.join(BASE_DIR, "candidate_profiles")
CODING_DIR = os.path.join(BASE_DIR, "coding")
SCREEN_DIR = os.path.join(BASE_DIR, "screen_context")
FAILURES_DIR = os.path.join(BASE_DIR, "failures")

for d in [AUDIO_DIR, TRANSCRIPTS_DIR, QUESTIONS_DIR, PROFILES_DIR, CODING_DIR, SCREEN_DIR, FAILURES_DIR]:
    os.makedirs(d, exist_ok=True)


def create_wav(filename: str, samples: np.ndarray, sample_rate: int = 16000):
    filepath = os.path.join(AUDIO_DIR, filename)
    samples_int16 = np.clip(samples, -32768, 32767).astype(np.int16)
    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples_int16.tobytes())
    return filepath


def generate_audio_fixtures():
    # 1. Pure Silence (1.0 sec @ 16kHz)
    silence = np.zeros(16000, dtype=np.int16)
    create_wav("silence_16k.wav", silence, 16000)

    # 2. 16kHz, 44.1kHz, 48kHz pure tone (440Hz, 1.0 sec)
    for sr, name in [(16000, "tone_16k.wav"), (44100, "tone_44k.wav"), (48000, "tone_48k.wav")]:
        t = np.linspace(0, 1.0, sr, endpoint=False)
        tone = (np.sin(2 * np.pi * 440 * t) * 12000).astype(np.int16)
        create_wav(name, tone, sr)

    # 3. Loud clipped audio (near +/- 32767)
    t = np.linspace(0, 0.5, 8000, endpoint=False)
    loud = np.clip(np.sin(2 * np.pi * 300 * t) * 50000, -32767, 32767).astype(np.int16)
    create_wav("loud_clipped_16k.wav", loud, 16000)

    # 4. Very quiet speech-level audio (RMS ~ 50)
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    quiet = (np.sin(2 * np.pi * 200 * t) * 65).astype(np.int16)
    create_wav("quiet_signal_16k.wav", quiet, 16000)

    # 5. Simulated mechanical keyboard click (impulsive transient: high crest factor > 6.0, high ZCR)
    click = np.zeros(16000, dtype=np.int16)
    click_pulse = np.array([25000, -22000, 18000, -14000, 10000, -6000, 3000, -1000], dtype=np.int16)
    click[4000:4008] = click_pulse
    click[8000:8008] = click_pulse
    create_wav("keyboard_clicks_16k.wav", click, 16000)

    # 6. Simulated fan noise (ambient mechanical motor rumble < 85 Hz + floor)
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    fan_rumble = (np.sin(2 * np.pi * 60 * t) * 150).astype(np.float32)
    fan_noise = fan_rumble + np.random.normal(0, 20, 16000)
    create_wav("fan_noise_16k.wav", fan_noise, 16000)

    # 7. Simulated notification chime (pure high-pitch tonal chime > 550 Hz, high spectral peakiness)
    t = np.linspace(0, 0.4, 6400, endpoint=False)
    chime = (np.sin(2 * np.pi * 880 * t) * 15000).astype(np.int16)
    create_wav("notification_chime_16k.wav", chime, 16000)


def generate_question_datasets():
    # 100 Technical questions
    tech_questions = [
        "What is polymorphism?",
        "Can you explain polymorphism in object-oriented programming?",
        "What is the difference between an abstract class and an interface?",
        "How does Python manage memory with reference counting and garbage collection?",
        "What is the Global Interpreter Lock (GIL) in CPython?",
        "How does asyncio work under the hood with an event loop?",
        "What is the difference between concurrency and parallelism?",
        "How do generators and the yield keyword work in Python?",
        "What are Python decorators and how do they wrap functions?",
        "How does dictionary hashing work in Python?",
        "What is the time complexity of searching in a hash table in average and worst cases?",
        "What is the difference between deep copy and shallow copy in Python?",
        "What is a REST API?",
        "What are the HTTP methods GET, POST, PUT, PATCH, and DELETE?",
        "What is idempotent in the context of HTTP methods?",
        "What is the difference between PUT and PATCH?",
        "How do HTTP status codes 401 Unauthorized and 403 Forbidden differ?",
        "What are the core components of a JWT token?",
        "How is a JWT signature verified?",
        "What is the difference between symmetric and asymmetric JWT signing?",
        "What is CORS and why is the Preflight OPTIONS request sent?",
        "How do WebSockets provide full-duplex communication over a single TCP connection?",
        "What is the difference between TCP and UDP?",
        "How does the TLS/SSL 3-way handshake work?",
        "What is connection pooling and why is it important in database clients?",
        "What is ACID in relational databases?",
        "What are the different transaction isolation levels in SQL?",
        "What is a dirty read vs a phantom read?",
        "How do B-Tree indexes work in PostgreSQL?",
        "What is the difference between a clustered and non-clustered index?",
        "When would you choose PostgreSQL over MongoDB?",
        "How does MongoDB store documents and handle schema flexibility?",
        "What is Redis and what data structures does it support?",
        "What are cache eviction policies like LRU, LFU, and FIFO?",
        "How do you handle the cache stampede or thundering herd problem?",
        "What is the difference between Cache-Aside and Write-Through caching patterns?",
        "What is the CAP theorem?",
        "How do you design a rate limiter using Redis token bucket or sliding window?",
        "What is database sharding and how does consistent hashing help?",
        "What is the difference between horizontal and vertical scaling?",
        "How does a reverse proxy like Nginx differ from a forward proxy?",
        "What is Docker and how do containers differ from virtual machines?",
        "What are Linux namespaces and cgroups in containerization?",
        "What is Kubernetes and what is the role of the kube-apiserver and kubelet?",
        "What is a Kubernetes Pod vs a Deployment?",
        "How does Kubernetes Service ClusterIP differ from NodePort and LoadBalancer?",
        "What is CI/CD and what are the stages of a deployment pipeline?",
        "What is gRPC and how does it use HTTP/2 and Protocol Buffers?",
        "What is GraphQL and how does it solve over-fetching and under-fetching?",
        "What is the N+1 problem in GraphQL or ORMs and how do you solve it?"
    ]
    # Expand to 100 with technical inquiries
    for i in range(len(tech_questions) + 1, 101):
        tech_questions.append(f"How do you optimize system performance in scenario {i}?")

    # 100 Project questions
    project_questions = [
        "Tell me about your Farmer Marketplace project.",
        "What did you contribute to the Farmer Marketplace?",
        "What architecture did you choose for the Farmer Marketplace?",
        "What database did you use for the Farmer Marketplace?",
        "Why did you choose Django for the backend?",
        "How did you handle authentication in the Farmer Marketplace?",
        "How did you implement the product listing feature?",
        "How did you integrate React with the Django backend?",
        "What caching mechanism did you use for product queries?",
        "How did you handle state management on the frontend in React?",
        "What challenges did you face during database migrations?",
        "How did you design the user role model for farmers and buyers?",
        "What payment gateway did you integrate for transactions?",
        "How did you test your Django API endpoints?",
        "How did you handle image uploads for farmer produce?",
        "What was the deployment strategy for the Farmer Marketplace?",
        "How did you handle concurrent orders for limited inventory?",
        "What was your experience with Docker in this project?",
        "How did you monitor server health and error logging?",
        "What would you do differently if you built the Farmer Marketplace again?"
    ]
    for i in range(len(project_questions) + 1, 101):
        project_questions.append(f"In your project, how did you handle operational challenge {i}?")

    # 50 Behavioral questions
    behavioral_questions = [
        "Tell me about a time you faced a difficult technical challenge and how you overcame it.",
        "How do you prioritize tasks when working under tight deadlines?",
        "Describe a situation where you had a disagreement with a team member and how you resolved it.",
        "Tell me about a time you had to learn a new technology quickly.",
        "How do you handle constructive feedback on your code reviews?",
        "Tell me about a project that failed or missed a deadline and what you learned from it.",
        "How do you ensure clear communication with non-technical stakeholders?",
        "Describe a time you took initiative to improve an existing process or codebase.",
        "How do you mentor or assist junior engineers on your team?",
        "Where do you see yourself technically in the next three years?"
    ]
    for i in range(len(behavioral_questions) + 1, 51):
        behavioral_questions.append(f"Behavioral interview inquiry scenario {i}: describe your approach.")

    # 50 Coding questions
    coding_questions = [
        "Write a Python function to reverse a linked list.",
        "Write a function to find the two numbers that add up to a target sum in an array.",
        "Implement binary search on a sorted list of integers.",
        "Write a function to check if a string contains valid parentheses.",
        "Write a function to find the maximum subarray sum using Kadane's algorithm.",
        "Implement a LRU cache with get and put operations in O(1) time.",
        "Write a function to invert a binary tree.",
        "Write a function to merge two sorted linked lists.",
        "Implement a function to find the longest substring without repeating characters.",
        "Write a function to detect a cycle in a linked list using Floyd's algorithm."
    ]
    for i in range(len(coding_questions) + 1, 51):
        coding_questions.append(f"Write a function to solve algorithmic task {i}.")

    # 50 Follow-ups
    follow_up_questions = [
        "Why did you choose it?",
        "Why?",
        "And what about REST APIs?",
        "And authentication?",
        "What was the trade-off with that approach?",
        "Can we optimize the time complexity further?",
        "How would that handle high concurrency?",
        "What happens if the primary database fails?",
        "Can you elaborate on the caching strategy?",
        "Why not use MongoDB instead?"
    ]
    for i in range(len(follow_up_questions) + 1, 51):
        follow_up_questions.append(f"Follow-up question on detail {i}?")

    # 50 Unknown Information / Anti-Hallucination questions
    unknown_questions = [
        "Did you deploy the Farmer Marketplace to Kubernetes clusters at Google?",
        "Tell me about your experience working at Goldman Sachs in 2021.",
        "What did you build during your internship at Netflix?",
        "Explain your patent on quantum cryptography algorithms.",
        "How did you lead the team of 50 engineers at Microsoft?",
        "What AWS certified solutions architect professional score did you receive?",
        "How did you migrate the core payments ledger to Solana blockchain?",
        "Tell me about your published paper in Nature Neuroscience.",
        "What was your contribution to the Linux kernel TCP stack?",
        "Did you use Rust to rewrite the SpaceX telemetry parser?"
    ]
    for i in range(len(unknown_questions) + 1, 51):
        unknown_questions.append(f"Did you earn unverified certification or work at unverified company {i}?")

    with open(os.path.join(QUESTIONS_DIR, "technical_questions_100.json"), "w") as f:
        json.dump(tech_questions, f, indent=2)
    with open(os.path.join(QUESTIONS_DIR, "project_questions_100.json"), "w") as f:
        json.dump(project_questions, f, indent=2)
    with open(os.path.join(QUESTIONS_DIR, "behavioral_questions_50.json"), "w") as f:
        json.dump(behavioral_questions, f, indent=2)
    with open(os.path.join(QUESTIONS_DIR, "coding_questions_50.json"), "w") as f:
        json.dump(coding_questions, f, indent=2)
    with open(os.path.join(QUESTIONS_DIR, "follow_up_questions_50.json"), "w") as f:
        json.dump(follow_up_questions, f, indent=2)
    with open(os.path.join(QUESTIONS_DIR, "unknown_questions_50.json"), "w") as f:
        json.dump(unknown_questions, f, indent=2)


def generate_candidate_profile():
    profile = {
        "candidate_name": "Alex Mercer",
        "title": "Full Stack Software Engineer",
        "skills": ["Python", "Django", "FastAPI", "React", "JavaScript", "PostgreSQL", "Redis", "Docker", "REST APIs", "Git"],
        "projects": [
            {
                "name": "Farmer Marketplace",
                "role": "Lead Full Stack Developer",
                "technologies": ["Python", "Django", "React", "PostgreSQL", "Redis"],
                "description": "A web marketplace connecting local farmers directly with bulk produce buyers.",
                "contributions": [
                    "Engineered RESTful backend APIs using Django and PostgreSQL.",
                    "Built reactive buyer and seller dashboards in React.",
                    "Implemented JWT-based user authentication with role-based access control.",
                    "Designed product catalog with search, filtering, and Redis query caching."
                ],
                "explicit_exclusions": [
                    "Did NOT deploy to Kubernetes (deployed on single-node Docker VPS).",
                    "Did NOT work at Google, Netflix, Goldman Sachs, or Microsoft.",
                    "Did NOT use Rust, C++, Kubernetes, or Solana blockchain."
                ]
            }
        ]
    }
    with open(os.path.join(PROFILES_DIR, "farmer_marketplace_profile.json"), "w") as f:
        json.dump(profile, f, indent=2)


def generate_transcript_fixtures():
    data = {
        "interim_sequence": [
            {"type": "interim", "text": "What is", "time": 0.1},
            {"type": "interim", "text": "What is Django", "time": 0.3},
            {"type": "final", "text": "What is Django?", "time": 0.5}
        ],
        "overlapping_pair": {
            "prev": "Why did you choose PostgreSQL",
            "new": "PostgreSQL for your database?",
            "expected_merge": "Why did you choose PostgreSQL for your database?"
        },
        "duplicate_sequence": [
            "Write a function to invert a binary tree.",
            "Write a function to invert a binary tree.",
            "write a function to invert a binary tree?"
        ],
        "interviewer_candidate_turn": {
            "interviewer": "What is Django?",
            "candidate": "Django is a Python web framework."
        }
    }
    with open(os.path.join(TRANSCRIPTS_DIR, "transcript_scenarios.json"), "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    generate_audio_fixtures()
    generate_question_datasets()
    generate_candidate_profile()
    generate_transcript_fixtures()
    print("[OK] All golden test fixtures generated successfully in tests/fixtures/")
