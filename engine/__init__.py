"""
BluJay AI - Real-time Interview Copilot Engine Package
======================================================
Modular engine architecture for audio transcription, turn-taking,
question relationship intelligence, dynamic prompt assembly, and answer streaming.
"""

from engine.utils import (
    get_openai_client,
    extract_text_from_file,
    load_user_config,
    save_user_config,
    CONFIG_PATH,
    LEGACY_CONFIG_PATH
)

from engine.session_models import (
    InterviewTurnState,
    ComponentHealthState,
    categorize_api_error,
    SessionController,
    AudioFrame,
    AudioSegment,
    QuestionLatencyTrace,
    InterviewSession,
    calculate_session_analytics,
    QuestionStabilityState
)

from engine.relationship_classifier import (
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
    compute_question_likelihood
)

from engine.turn_taking import (
    detect_speaker_turn,
    is_conversational_eligible_answer,
    classify_and_understand_utterance
)

from engine.prompt_builder import (
    TECH_TOOL_WHITELIST,
    determine_knowledge_source,
    build_interview_system_prompt,
    validate_candidate_grounding
)

from engine.llm_orchestrator import (
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
    generate_session_summary
)

from engine.transcriber import (
    STTProvider,
    GroqWhisperProvider,
    ResilientMultiSTTProvider,
    ContinuousTranscriber
)

__all__ = [
    # Utils
    "get_openai_client",
    "extract_text_from_file",
    "load_user_config",
    "save_user_config",
    "CONFIG_PATH",
    "LEGACY_CONFIG_PATH",
    # Session Models
    "InterviewTurnState",
    "ComponentHealthState",
    "categorize_api_error",
    "SessionController",
    "AudioFrame",
    "AudioSegment",
    "QuestionLatencyTrace",
    "InterviewSession",
    "calculate_session_analytics",
    "QuestionStabilityState",
    # Relationship & Intelligence
    "QUESTION_STARTERS",
    "INTERVIEW_QUESTION_PATTERNS",
    "ASR_CORRECTIONS",
    "WHISPER_HALLUCINATIONS",
    "_STANDALONE_GHOST_WORDS",
    "_is_whisper_hallucination",
    "clean_asr_transcript",
    "compute_question_confidence",
    "is_likely_question",
    "refine_interview_question",
    "PRIMARY_TECH_PLATFORMS",
    "extract_primary_tech_entity",
    "classify_answer_relationship",
    "analyze_conversation_flow",
    "get_normalized_semantic_key",
    "evaluate_question_completeness",
    "compute_question_likelihood",
    # Turn Taking
    "detect_speaker_turn",
    "is_conversational_eligible_answer",
    "classify_and_understand_utterance",
    # Prompt Building
    "TECH_TOOL_WHITELIST",
    "determine_knowledge_source",
    "build_interview_system_prompt",
    "validate_candidate_grounding",
    # LLM Orchestrator
    "get_adaptive_max_tokens",
    "parse_retry_after_seconds",
    "sanitize_generated_answer",
    "StreamingReasoningFilter",
    "validate_and_clean_answer",
    "generate_interview_answer",
    "predict_likely_followups",
    "prewarm_pipeline_connections",
    "generate_mock_interview_question",
    "evaluate_mock_interview_answer",
    "generate_mock_interview_final_report",
    "generate_session_summary",
    # Transcriber
    "STTProvider",
    "GroqWhisperProvider",
    "ResilientMultiSTTProvider",
    "ContinuousTranscriber",
]
