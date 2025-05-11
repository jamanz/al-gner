# nemo_forced_aligner/aligner.py

import os
import tempfile
import logging
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Union, List

import torch
import librosa
import soundfile as sf
from omegaconf import OmegaConf

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.models.ctc_models import EncDecCTCModel
from nemo.collections.asr.models.hybrid_rnnt_ctc_models import EncDecHybridRNNTCTCModel
from nemo.core.config import hydra_runner
from nemo.utils import logging as nemo_logging

from .constants import (
    SUPPORTED_MODELS, CTC_MODELS, DEFAULT_SAMPLE_RATE,
    DEFAULT_CHUNK_LEN, DEFAULT_TOTAL_BUFFER, DEFAULT_CHUNK_BATCH_SIZE,
    LONG_AUDIO_THRESHOLD
)
from .utils.data_prep import get_batch_variables, get_manifest_lines_batch, get_batch_starts_ends
from .utils.viterbi_decoding import viterbi_decoding
from .utils.srt_utils import generate_srt_from_alignment


@dataclass
class NeMoAlignmentConfig:
    """Configuration for NeMo forced alignment"""

    # Required parameters
    manifest_filepath: str
    output_dir: str

    # Model parameters
    pretrained_name: Optional[str] = None
    model_path: Optional[str] = None

    # Device parameters
    transcribe_device: Optional[str] = None
    viterbi_device: Optional[str] = None

    # Long audio support
    use_buffered_chunked_streaming: bool = False
    chunk_len_in_secs: float = DEFAULT_CHUNK_LEN
    total_buffer_in_secs: float = DEFAULT_TOTAL_BUFFER
    chunk_batch_size: int = DEFAULT_CHUNK_BATCH_SIZE
    use_local_attention: bool = True

    # Other parameters
    batch_size: int = 1
    audio_filepath_parts_in_utt_id: int = 1
    additional_segment_grouping_separator: Optional[str] = "|"


class NeMoForcedAligner:
    """Simple forced aligner for integration into data processing pipelines"""

    def __init__(
            self,
            model_name: Optional[str] = None,
            device: Optional[str] = None,
            logger: Optional[logging.Logger] = None,
            # Long audio support
            use_buffered_streaming: bool = True,
            chunk_len_in_secs: float = DEFAULT_CHUNK_LEN,
            total_buffer_in_secs: float = DEFAULT_TOTAL_BUFFER,
            chunk_batch_size: int = DEFAULT_CHUNK_BATCH_SIZE,
            use_local_attention: bool = True,
            # Device separation
            transcribe_device: Optional[str] = None,
            viterbi_device: Optional[str] = None,
    ):
        """
        Initialize the aligner

        Args:
            model_name: Model name from constants or custom model path
            device: Default device if transcribe/viterbi not specified
            logger: Optional logger instance
            use_buffered_streaming: Enable chunk streaming for long audio
            chunk_len_in_secs: Chunk length for streaming
            total_buffer_in_secs: Total buffer size
            chunk_batch_size: Batch size for chunks
            use_local_attention: Enable local attention for memory efficiency
            transcribe_device: Device for transcription step
            viterbi_device: Device for viterbi decoding
        """
        # Setup logger
        self.logger = logger or self._setup_default_logger()

        # Suppress NeMo logging in notebooks
        nemo_logging.setLevel(logging.ERROR)

        # Handle device selection
        self.default_device = self._select_device(device)
        self.transcribe_device = self._select_device(transcribe_device or device)
        self.viterbi_device = self._select_device(viterbi_device or device)

        # Store configuration
        self.config = NeMoAlignmentConfig(
            manifest_filepath="",  # Will be set per alignment
            output_dir="",  # Will be set per alignment
            use_buffered_chunked_streaming=use_buffered_streaming,
            chunk_len_in_secs=chunk_len_in_secs,
            total_buffer_in_secs=total_buffer_in_secs,
            chunk_batch_size=chunk_batch_size,
            use_local_attention=use_local_attention,
            transcribe_device=str(self.transcribe_device),
            viterbi_device=str(self.viterbi_device),
        )

        # Load model
        self.model = self._load_model(model_name)
        self._temp_files = []

    def _setup_default_logger(self) -> logging.Logger:
        """Setup default logger"""
        logger = logging.getLogger('NeMoForcedAligner')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _select_device(self, device: Optional[str] = None) -> torch.device:
        """Auto-select best available device"""
        if device:
            return torch.device(device)

        if torch.cuda.is_available():
            # Select GPU with most free memory
            max_free_memory = 0
            best_device = 0

            for i in range(torch.cuda.device_count()):
                torch.cuda.set_device(i)
                free_memory = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)

                if free_memory > max_free_memory:
                    max_free_memory = free_memory
                    best_device = i

            device_name = f"cuda:{best_device}"
            self.logger.info(f"Auto-selected device: {device_name} ({max_free_memory / 1e9:.1f}GB free)")
            return torch.device(device_name)

        self.logger.info("CUDA not available, using CPU")
        return torch.device("cpu")

    def _detect_language_from_model(self, model_name: str) -> str:
        """Detect language from model name"""
        for lang, model in SUPPORTED_MODELS.items():
            if model == model_name:
                return lang

        # Fallback to finding in model name
        for lang in SUPPORTED_MODELS.keys():
            if f"_{lang}_" in model_name:
                return lang

        return "en"  # Default to English

    def _select_best_model_for_duration(self, duration: float, default_model: str) -> str:
        """Select appropriate model based on audio duration"""
        if duration > LONG_AUDIO_THRESHOLD:
            lang = self._detect_language_from_model(default_model)

            if lang in CTC_MODELS:
                ctc_model = CTC_MODELS[lang]
                self.logger.info(f"Long audio detected ({duration / 60:.1f} min), using CTC model: {ctc_model}")
                return ctc_model

        return default_model

    def _load_model(self, model_name: Optional[str] = None) -> ASRModel:
        """Load and configure NeMo model"""
        if model_name is None:
            model_name = SUPPORTED_MODELS["fr"]  # Default to French
        elif model_name in SUPPORTED_MODELS:
            model_name = SUPPORTED_MODELS[model_name]

        self.logger.info(f"Loading model: {model_name}")

        model = ASRModel.from_pretrained(model_name, map_location=self.default_device)
        model.eval()

        # Configure model for CTC if hybrid
        if isinstance(model, EncDecHybridRNNTCTCModel):
            model.change_decoding_strategy(decoder_type="ctc")

        # Configure local attention if supported
        if self.config.use_local_attention:
            try:
                model.change_attention_model(
                    self_attention_model="rel_pos_local_attn",
                    att_context_size=[64, 64]
                )
                self.logger.info("Enabled local attention for memory efficiency")
            except:
                self.logger.warning("Local attention not supported by this model")

        return model

    def _process_audio(self, audio_path: str) -> str:
        """Convert any audio format to WAV if needed"""
        # Check if already WAV at correct sample rate
        _, ext = os.path.splitext(audio_path)

        if ext.lower() == '.wav':
            # Check sample rate and channels
            try:
                info = sf.info(audio_path)
                if info.samplerate == DEFAULT_SAMPLE_RATE and info.channels == 1:
                    return audio_path
            except:
                pass

        # Need to convert - load and resample
        self.logger.info(f"Converting {audio_path} to WAV format")

        try:
            # Load with librosa (handles most formats)
            y, sr = librosa.load(audio_path, sr=DEFAULT_SAMPLE_RATE, mono=True)

            # Create temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp_path = tmp.name

            # Save as WAV
            sf.write(tmp_path, y, DEFAULT_SAMPLE_RATE)

            # Track for cleanup
            self._temp_files.append(tmp_path)

            return tmp_path

        except Exception as e:
            self.logger.error(f"Failed to process audio: {e}")
            raise

    def _process_text_file(self, text_path: str) -> str:
        """Process text file with line-by-line segments"""
        with open(text_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        # Add '|' delimiters if not present
        if '|' not in ' '.join(lines):
            text = ' | '.join(lines)
        else:
            text = ' '.join(lines)

        self.logger.info(f"Processed {len(lines)} segments from text file")
        return text

    def _create_manifest(self, audio_path: str, text: str) -> str:
        """Create NeMo manifest file"""
        manifest_data = {
            "audio_filepath": audio_path,
            "text": text
        }

        # Create temporary manifest file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            tmp_path = tmp.name
            json.dump(manifest_data, tmp)
            tmp.write('\n')

        self._temp_files.append(tmp_path)
        return tmp_path

    def _check_audio_duration(self, audio_path: str) -> float:
        """Check audio duration and optionally adjust configuration"""
        y, sr = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)

        self.logger.info(f"Audio duration: {duration:.2f} seconds ({duration / 60:.2f} minutes)")

        # Auto-adjust for long audio
        if duration > LONG_AUDIO_THRESHOLD:
            self.logger.info("Long audio detected, enabling streaming configuration")
            self.config.use_buffered_chunked_streaming = True
            self.config.chunk_len_in_secs = 1.6
            self.config.total_buffer_in_secs = 4.0
            self.config.chunk_batch_size = 16

            # Consider using CPU for viterbi if GPU memory is limited
            if self.transcribe_device.type == 'cuda' and self.viterbi_device.type == 'cuda':
                self.logger.info("Moving viterbi decoding to CPU for memory efficiency")
                self.viterbi_device = torch.device('cpu')
                self.config.viterbi_device = 'cpu'

        return duration

    def _run_alignment(self, config: NeMoAlignmentConfig) -> dict:
        """Run the actual alignment process"""
        # Get batch starts and ends
        starts, ends = get_batch_starts_ends(config.manifest_filepath, config.batch_size)

        # Initialize variables
        output_timestep_duration = None
        alignments = []

        # Process batches
        for start, end in zip(starts, ends):
            manifest_lines_batch = get_manifest_lines_batch(config.manifest_filepath, start, end)

            # Get batch variables
            (log_probs_batch, y_batch, T_batch, U_batch, utt_obj_batch,
             output_timestep_duration,) = get_batch_variables(
                manifest_lines_batch,
                self.model,
                config.additional_segment_grouping_separator,
                False,  # align_using_pred_text
                config.audio_filepath_parts_in_utt_id,
                output_timestep_duration,
                False,  # simulate_cache_aware_streaming
                config.use_buffered_chunked_streaming,
                {
                    "delay": None,
                    "model_stride_in_secs": None,
                    "tokens_per_chunk": None
                } if config.use_buffered_chunked_streaming else {}
            )

            # Run viterbi decoding
            alignments_batch = viterbi_decoding(
                log_probs_batch,
                y_batch,
                T_batch,
                U_batch,
                self.viterbi_device
            )

            # Store results
            for utt_obj, alignment in zip(utt_obj_batch, alignments_batch):
                alignments.append({
                    'utt_obj': utt_obj,
                    'alignment': alignment,
                    'output_timestep_duration': output_timestep_duration
                })

        return alignments

    def align(
            self,
            audio_path: str,
            text_path: Optional[str] = None,
            text: Optional[str] = None,
            output_srt_path: Optional[str] = None
    ) -> Union[str, Path]:
        """
        Align audio with text and generate SRT file

        Args:
            audio_path: Path to audio file (mp3, wav, etc.)
            text_path: Path to text file with line-by-line segments
            text: Direct text input (alternative to text_path)
            output_srt_path: Where to save SRT (optional, returns content if None)

        Returns:
            SRT content if output_srt_path is None, else path to saved file
        """
        try:
            # Validate inputs
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            if text_path and not os.path.exists(text_path):
                raise FileNotFoundError(f"Text file not found: {text_path}")

            if not text_path and not text:
                raise ValueError("Either text_path or text must be provided")

            # Check audio duration and adjust configuration
            audio_duration = self._check_audio_duration(audio_path)

            # Potentially adjust model for long audio
            original_model_name = self.model.name if hasattr(self.model, 'name') else None
            if original_model_name:
                best_model_name = self._select_best_model_for_duration(audio_duration, original_model_name)
                if best_model_name != original_model_name:
                    self.model = self._load_model(best_model_name)

            # Process audio
            processed_audio = self._process_audio(audio_path)

            # Process text
            if text_path:
                transcript = self._process_text_file(text_path)
            else:
                transcript = text

            # Create manifest
            manifest_path = self._create_manifest(processed_audio, transcript)

            # Create output directory
            with tempfile.TemporaryDirectory() as output_dir:
                # Configure alignment
                config = self.config
                config.manifest_filepath = manifest_path
                config.output_dir = output_dir

                # Run alignment
                alignments = self._run_alignment(config)

                # Generate SRT
                srt_content = generate_srt_from_alignment(alignments)

                # Save or return
                if output_srt_path:
                    with open(output_srt_path, 'w', encoding='utf-8') as f:
                        f.write(srt_content)
                    self.logger.info(f"Saved alignment to: {output_srt_path}")
                    return output_srt_path

                return srt_content

        except Exception as e:
            self.logger.error(f"Alignment failed: {e}")
            raise

        finally:
            # Cleanup temporary files
            self._cleanup_temp_files()

    def _cleanup_temp_files(self):
        """Clean up temporary files"""
        for temp_file in self._temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                self.logger.warning(f"Failed to delete temp file {temp_file}: {e}")

        self._temp_files = []

    def __del__(self):
        """Cleanup on deletion"""
        self._cleanup_temp_files()