# nemo_forced_aligner/constants.py

# Supported model names for different languages
SUPPORTED_MODELS = {
    "en": "stt_en_fastconformer_hybrid_large_pc",
    "fr": "stt_fr_conformer_ctc_large",
    "de": "stt_de_fastconformer_hybrid_large_pc",
    "es": "stt_es_fastconformer_hybrid_large_pc",
    "zh": "stt_zh_citrinet_1024_gamma_0_25",
}

# CTC models for long audio (better memory efficiency)
CTC_MODELS = {
    "en": "stt_en_conformer_ctc_small",
    "fr": "stt_fr_conformer_ctc_large",
    "de": "stt_de_conformer_ctc_large",
    "es": "stt_es_conformer_ctc_large",
}

# Default settings
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHUNK_LEN = 1.6
DEFAULT_TOTAL_BUFFER = 4.0
DEFAULT_CHUNK_BATCH_SIZE = 32

# Long audio threshold (in seconds)
LONG_AUDIO_THRESHOLD = 7200  # 2 hours