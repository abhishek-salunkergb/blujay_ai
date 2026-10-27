"""
Real-time Interview Copilot Engine (Public Architecture Facade)
===============================================================
Exposes the complete public API for backward compatibility across all existing
modules (main.py, web_copilot_view.py, vision.py, tests/*).

Internally, this facade delegates to the decoupled domain submodules under `engine/`:
- engine.utils: Document extractors, client persistence, user config
- engine.session_models: InterviewSession, InterviewTurnState, latency tracing
- engine.relationship_classifier: Parakeet follow-up fusion, entity extraction
- engine.turn_taking: Turn detection, utterance segmentation
- engine.prompt_builder: Dynamic prompt assembly, knowledge grounding
- engine.llm_orchestrator: Streaming answer generation, reasoning filters
- engine.transcriber: Continuous audio loopback & microphone transcription
"""

from engine import (
    get_openai_client,
    extract_text_from_file,
    load_user_config,
    save_user_config,
    CONFIG_PATH,
    LEGACY_CONFIG_PATH,
    InterviewTurnState,
    ComponentHealthState,
    categorize_api_error,
    SessionController,
    AudioFrame,
    AudioSegment,
    QuestionLatencyTrace,
    InterviewSession,
    calculate_session_analytics,
    QuestionStabilityState,
    QUESTION_STARTERS,
    INTERVIEW_QUESTION_PATTERNS,
    ASR_CORRECTIONS,
    WHISPER_HALLUCINATIONS,
    _STANDALONE_GHOST_WORDS,
    _is_whisper_hallucination,
    clean_asr_transcript,
    compute_question_confidence,
    is_likely_question,
    refine_interview_question,
    PRIMARY_TECH_PLATFORMS,
    extract_primary_tech_entity,
    classify_answer_relationship,
    analyze_conversation_flow,
    get_normalized_semantic_key,
    evaluate_question_completeness,
    compute_question_likelihood,
    detect_speaker_turn,
    is_conversational_eligible_answer,
    classify_and_understand_utterance,
    TECH_TOOL_WHITELIST,
    determine_knowledge_source,
    build_interview_system_prompt,
    validate_candidate_grounding,
    get_adaptive_max_tokens,
    parse_retry_after_seconds,
    sanitize_generated_answer,
    StreamingReasoningFilter,
    validate_and_clean_answer,
    generate_interview_answer,
    predict_likely_followups,
    prewarm_pipeline_connections,
    generate_mock_interview_question,
    evaluate_mock_interview_answer,
    generate_mock_interview_final_report,
    generate_session_summary,
    STTProvider,
    GroqWhisperProvider,
    ResilientMultiSTTProvider,
    ContinuousTranscriber,
    __all__
)
