# nemo_forced_aligner/utils/data_prep.py

import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

from nemo.collections.asr.parts.utils.streaming_utils import FrameBatchASR
from nemo.utils import logging
import librosa

# Constants for alignment
BLANK_TOKEN = '<blank>'
SPACE_TOKEN = ' '
V_NEGATIVE_NUM = -10.0


def get_batch_starts_ends(manifest_filepath: str, batch_size: int) -> Tuple[List[int], List[int]]:
    """Get start and end line numbers for each batch"""
    with open(manifest_filepath, 'r') as f:
        lines = f.readlines()

    starts = list(range(0, len(lines), batch_size))
    ends = [min(start + batch_size, len(lines)) for start in starts]

    return starts, ends


def get_manifest_lines_batch(manifest_filepath: str, start: int, end: int) -> List[Dict]:
    """Get a batch of manifest lines"""
    with open(manifest_filepath, 'r') as f:
        lines = f.readlines()

    batch_lines = []
    for i in range(start, end):
        if i < len(lines):
            batch_lines.append(json.loads(lines[i]))

    return batch_lines


def get_utt_obj(
        manifest_line: Dict,
        utt_id: str,
        segment_grouping_separator: Optional[str],
        model,
        log_probs: torch.Tensor,
        output_timestep_duration: float
) -> Dict:
    """Create utterance object with alignment information following NeMo's structure"""

    text = manifest_line.get('text', '').strip()
    audio_filepath = manifest_line.get('audio_filepath', '')

    # Create basic utterance object
    utt_obj = {
        'utt_id': utt_id,
        'text': text,
        'audio_filepath': audio_filepath,
        'output_timestep_duration': output_timestep_duration,
        'token_ids_with_blanks': [],
        'segments_and_tokens': []
    }

    # Process segments if separator is provided
    if segment_grouping_separator is None:
        segments = [text]
    else:
        segments = text.split(segment_grouping_separator)

    # Remove empty segments and strip whitespace
    segments = [seg.strip() for seg in segments if seg.strip()]

    # Build up token IDs with blanks for Viterbi alignment
    if hasattr(model, 'tokenizer'):
        # Use model's tokenizer
        if hasattr(model, 'blank_id'):
            BLANK_ID = model.blank_id
        else:
            BLANK_ID = len(model.tokenizer.vocab)

        utt_obj['token_ids_with_blanks'] = [BLANK_ID]
        
        # Check for empty text
        if len(text) == 0:
            return utt_obj

        # Process segments and create tokens
        for segment in segments:
            segment_tokens = model.tokenizer.text_to_ids(segment)
            
            # Build token structure with blanks
            for token_id in segment_tokens:
                utt_obj['token_ids_with_blanks'].extend([token_id, BLANK_ID])

        # Create simplified structure for alignment
        utt_obj['tokens'] = [t for t in utt_obj['token_ids_with_blanks'] if t != BLANK_ID]
        
    else:
        # Character-based tokenization fallback
        BLANK_ID = len(model.decoder.vocabulary)
        utt_obj['token_ids_with_blanks'] = [BLANK_ID]
        
        if len(text) == 0:
            return utt_obj

        tokens = []
        for char in text:
            if hasattr(model.decoder, 'vocabulary') and char in model.decoder.vocabulary:
                token_id = model.decoder.vocabulary.index(char)
            else:
                token_id = BLANK_ID
            tokens.append(token_id)
            utt_obj['token_ids_with_blanks'].extend([token_id, BLANK_ID])

        utt_obj['tokens'] = tokens

    return utt_obj


def process_long_audio_with_buffered_streaming(
        model,
        audio_filepath: str,
        chunk_size_seconds: float = 30.0,
        overlap_seconds: float = 2.0,
        sample_rate: int = 16000,
) -> Tuple[torch.Tensor, int, str]:
    """
    Process a long audio file using buffered chunked streaming to avoid memory issues.

    Args:
        model: NeMo ASR model (CTC or Hybrid with CTC mode)
        audio_filepath: Path to the audio file
        chunk_size_seconds: Size of each processing chunk in seconds
        overlap_seconds: Overlap between chunks in seconds
        sample_rate: Sample rate to use for audio processing

    Returns:
        log_probs: Log probabilities tensor
        T: Length of the log_probs tensor
        transcription: Full text transcription
    """
    logging.info(f"Processing long audio file: {audio_filepath}")

    # Get audio duration without loading entire file
    info = librosa.get_duration(path=audio_filepath)
    total_duration = info

    # Calculate number of chunks
    effective_chunk_size = chunk_size_seconds - overlap_seconds
    num_chunks = max(1, int(np.ceil(total_duration / effective_chunk_size)))

    logging.info(f"Audio duration: {total_duration:.2f} seconds")
    logging.info(f"Processing in {num_chunks} chunks of {chunk_size_seconds}s with {overlap_seconds}s overlap")

    # Process audio in chunks
    all_logits = []
    all_transcriptions = []

    # Create progress bar
    progress_bar = tqdm(range(num_chunks), desc="Processing chunks")

    # Calculate output timestep duration based on model
    if hasattr(model, 'name'):
        model_name = model.name.lower()
        if "citrinet" in model_name or "_fastconformer_" in model_name:
            output_timestep_duration = 0.08
        elif "_conformer_" in model_name:
            output_timestep_duration = 0.04
        elif "quartznet" in model_name:
            output_timestep_duration = 0.02
        else:
            output_timestep_duration = 0.04  # Default value
    else:
        output_timestep_duration = 0.04  # Default fallback

    # Calculate frames per chunk for logits
    frames_per_second = 1.0 / output_timestep_duration
    overlap_frames = int(overlap_seconds * frames_per_second)

    for i in progress_bar:
        # Calculate chunk boundaries
        start_time = max(0, i * effective_chunk_size)
        end_time = min(total_duration, start_time + chunk_size_seconds)
        chunk_length = end_time - start_time

        # Update progress bar
        progress_bar.set_description(f"Chunk {i + 1}/{num_chunks} ({start_time:.1f}s - {end_time:.1f}s)")

        try:
            # Load just this chunk of audio
            audio, sr = librosa.load(
                audio_filepath,
                sr=sample_rate,
                offset=start_time,
                duration=chunk_length
            )

            # Convert to mono if needed and ensure float32
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            audio = audio.astype(np.float32)

            # Convert to torch tensor
            audio_signal = torch.FloatTensor(audio).unsqueeze(0).to(model.device)
            audio_signal_length = torch.LongTensor([len(audio)]).to(model.device)

            # Get log probabilities using the model's forward method
            with torch.no_grad():
                log_probs, encoded_len, _ = model.forward(
                    input_signal=audio_signal,
                    input_signal_length=audio_signal_length
                )

            # Get text transcription for this chunk
            chunk_transcription = model.transcribe([audio_filepath],
                                                   offset=start_time,
                                                   duration=chunk_length)[0]
            all_transcriptions.append(chunk_transcription)

            # Remove batch dimension from log_probs
            log_probs = log_probs.squeeze(0)

            # Handle overlap - trim beginning for all chunks except first
            if i > 0 and overlap_frames > 0:
                log_probs = log_probs[overlap_frames // 2:]

            # Handle overlap - trim end for all chunks except last
            if i < num_chunks - 1 and overlap_frames > 0:
                log_probs = log_probs[:-overlap_frames // 2]

            # Add to the accumulated logits
            all_logits.append(log_probs)

        except Exception as e:
            logging.error(f"Error processing chunk {i + 1}: {str(e)}")
            # Continue with next chunk

    # Concatenate all logits
    if len(all_logits) > 0:
        try:
            full_log_probs = torch.cat(all_logits, dim=0)
            T = full_log_probs.shape[0]

            # Join transcriptions with spaces
            full_transcription = " ".join([t.strip() for t in all_transcriptions])

            logging.info(f"Successfully processed audio with {T} frames")
            return full_log_probs, T, full_transcription

        except Exception as e:
            logging.error(f"Error concatenating logits: {str(e)}")
            raise
    else:
        raise RuntimeError("Failed to process any chunks of the audio file")


def get_batch_variables(
        manifest_lines_batch: List[Dict],
        model,
        additional_segment_grouping_separator: Optional[str],
        align_using_pred_text: bool,
        audio_filepath_parts_in_utt_id: int,
        output_timestep_duration: Optional[float],
        simulate_cache_aware_streaming: bool = False,
        use_buffered_chunked_streaming: bool = False,
        buffered_chunk_params: Optional[Dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[Dict], float]:
    """
    Get batch variables for alignment with enhanced support for very long audio files.

    Args:
        manifest_lines_batch: List of manifest lines with audio paths and text
        model: NeMo ASR model
        additional_segment_grouping_separator: Optional separator for text segments
        align_using_pred_text: Whether to use ASR-predicted text for alignment
        audio_filepath_parts_in_utt_id: How many parts of audio path to use for utt_id
        output_timestep_duration: Duration of each output timestep
        simulate_cache_aware_streaming: Whether to simulate cache-aware streaming
        use_buffered_chunked_streaming: Whether to use buffered chunked streaming
        buffered_chunk_params: Parameters for buffered chunked streaming

    Returns:
        log_probs_batch: Log probabilities tensor for each utterance
        y_batch: Target token sequences
        T_batch: Sequence lengths
        U_batch: Target sequence lengths
        utt_obj_batch: Utterance objects
        output_timestep_duration: Duration per timestep
    """
    # Get audio filepaths for batch
    audio_filepaths_batch = [line["audio_filepath"] for line in manifest_lines_batch]
    B = len(audio_filepaths_batch)
    log_probs_list_batch = []
    T_list_batch = []
    pred_text_batch = []

    # Get default buffered chunk parameters if not provided
    if use_buffered_chunked_streaming and buffered_chunk_params is None:
        buffered_chunk_params = {
            "chunk_size_seconds": 30.0,  # Process 30 seconds at a time
            "overlap_seconds": 2.0,  # 2 second overlap between chunks
            "sample_rate": 16000  # Default sample rate
        }

    # Process audio files
    if not use_buffered_chunked_streaming:
        # Standard processing for short audio files
        if not simulate_cache_aware_streaming:
            with torch.no_grad():
                hypotheses = model.transcribe(audio_filepaths_batch, return_hypotheses=True, batch_size=B)
        else:
            # Simulate cache-aware streaming if requested
            with torch.no_grad():
                hypotheses = model.transcribe_simulate_cache_aware_streaming(
                    audio_filepaths_batch, return_hypotheses=True, batch_size=B
                )

        # Handle Hybrid models that return tuples
        if type(hypotheses) == tuple and len(hypotheses) == 2:
            hypotheses = hypotheses[0]

        # Extract log probabilities and predictions from hypotheses
        for hypothesis in hypotheses:
            log_probs_list_batch.append(hypothesis.y_sequence)
            T_list_batch.append(hypothesis.y_sequence.shape[0])
            pred_text_batch.append(hypothesis.text)
    else:
        # Buffered chunked streaming for long audio files
        logging.info("Using buffered chunked streaming for long audio files")

        # Get parameters from buffered_chunk_params
        chunk_size_seconds = buffered_chunk_params.get("chunk_size_seconds", 30.0)
        overlap_seconds = buffered_chunk_params.get("overlap_seconds", 2.0)
        sample_rate = buffered_chunk_params.get("sample_rate", 16000)

        for audio_filepath in audio_filepaths_batch:
            # Process long audio file with efficient buffered streaming
            full_log_probs, T, full_transcription = process_long_audio_with_buffered_streaming(
                model=model,
                audio_filepath=audio_filepath,
                chunk_size_seconds=chunk_size_seconds,
                overlap_seconds=overlap_seconds,
                sample_rate=sample_rate
            )

            # Store results
            log_probs_list_batch.append(full_log_probs)
            T_list_batch.append(T)
            pred_text_batch.append(full_transcription)

    # Process each manifest line with extracted log probabilities
    y_list_batch = []
    U_list_batch = []
    utt_obj_batch = []

    for i_line, line in enumerate(manifest_lines_batch):
        # Determine text to use for alignment
        if align_using_pred_text:
            gt_text_for_alignment = " ".join(pred_text_batch[i_line].split())
        else:
            gt_text_for_alignment = line["text"]

        # Generate utterance ID from audio filepath
        audio_path = Path(audio_filepaths_batch[i_line])
        utt_id = str(audio_path.stem).replace(' ', '_')

        # Calculate output timestep duration if not provided
        if output_timestep_duration is None:
            # Estimate based on model type
            model_name = model.name if hasattr(model, 'name') else ""

            if "citrinet" in model_name or "_fastconformer_" in model_name:
                output_timestep_duration = 0.08
            elif "_conformer_" in model_name:
                output_timestep_duration = 0.04
            elif "quartznet" in model_name:
                output_timestep_duration = 0.02
            else:
                output_timestep_duration = 0.04  # Default value

        # Create utterance object using helper function
        utt_obj = get_utt_obj(
            {"text": gt_text_for_alignment, "audio_filepath": audio_filepaths_batch[i_line]},
            utt_id,
            additional_segment_grouping_separator,
            model,
            log_probs_list_batch[i_line],
            output_timestep_duration
        )

        # Store additional information
        if align_using_pred_text:
            utt_obj['pred_text'] = pred_text_batch[i_line]
            if "text" in line:
                utt_obj['text'] = line["text"]
        else:
            utt_obj['text'] = line["text"]

        # Get token sequence for alignment
        y = utt_obj['tokens']

        # Add to batch
        y_list_batch.append(y)
        U_list_batch.append(len(y))
        utt_obj_batch.append(utt_obj)

    # Convert to proper tensors for Viterbi decoding
    T_max = max(T_list_batch)
    U_max = max(U_list_batch)
    V = len(model.tokenizer.vocab) + 1 if hasattr(model, 'tokenizer') else len(model.decoder.vocabulary) + 1

    T_batch = torch.tensor(T_list_batch)
    U_batch = torch.tensor(U_list_batch)

    # Create log_probs_batch tensor of shape (B x T_max x V)
    V_NEGATIVE_NUM = -10.0  # Large negative number for padding
    log_probs_batch = V_NEGATIVE_NUM * torch.ones((B, T_max, V))
    for b, log_probs_utt in enumerate(log_probs_list_batch):
        t = log_probs_utt.shape[0]
        log_probs_batch[b, :t, :] = log_probs_utt

    # Create y tensor of shape (B x U_max)
    y_batch = V * torch.ones((B, U_max), dtype=torch.int64)
    for b, y_utt in enumerate(y_list_batch):
        U_utt = U_batch[b]
        y_batch[b, :U_utt] = torch.tensor(y_utt)

    return (
        log_probs_batch,
        y_batch,
        T_batch,
        U_batch,
        utt_obj_batch,
        output_timestep_duration,
    )


def add_t_start_end_to_utt_obj(
        utt_obj: Dict,
        alignment: List[int],
        output_timestep_duration: float
) -> Dict:
    """Add start and end times to utterance object based on alignment"""
    
    # Simplified approach - just store the alignment for later use in SRT generation
    # The actual timing will be calculated in the SRT generation function
    utt_obj['alignment'] = alignment
    utt_obj['output_timestep_duration'] = output_timestep_duration
    
    return utt_obj