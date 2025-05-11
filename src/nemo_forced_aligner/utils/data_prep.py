# nemo_forced_aligner/utils/data_prep.py

import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

from nemo.collections.asr.parts.utils.streaming_utils import FrameBatchASR
from nemo.utils import logging

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
) -> Tuple[List[torch.Tensor], List[List[int]], List[int], List[int], List[Dict], float]:
    """
    Get batch variables for alignment

    Returns:
        log_probs_batch: List of log probability tensors
        y_batch: List of token sequences
        T_batch: List of time steps
        U_batch: List of token lengths
        utt_obj_batch: List of utterance objects
        output_timestep_duration: Duration per timestep
    """

    # Get audio filepaths for batch
    audio_filepaths_batch = [line["audio_filepath"] for line in manifest_lines_batch]
    B = len(audio_filepaths_batch)
    log_probs_list_batch = []
    T_list_batch = []
    pred_text_batch = []

    # Get log probabilities and hypotheses
    if not use_buffered_chunked_streaming:
        if not simulate_cache_aware_streaming:
            with torch.no_grad():
                hypotheses = model.transcribe(audio_filepaths_batch, return_hypotheses=True, batch_size=B)
        else:
            # Simulate cache-aware streaming if needed
            with torch.no_grad():
                # This is a placeholder - implement the actual cache-aware streaming method
                hypotheses = model.transcribe(audio_filepaths_batch, return_hypotheses=True, batch_size=B)

        # Handle Hybrid models that return tuples
        if type(hypotheses) == tuple and len(hypotheses) == 2:
            hypotheses = hypotheses[0]

        # Extract log probabilities and predictions from hypotheses
        for hypothesis in hypotheses:
            log_probs_list_batch.append(hypothesis.y_sequence)
            T_list_batch.append(hypothesis.y_sequence.shape[0])
            pred_text_batch.append(hypothesis.text)
    else:
        # Buffered chunked streaming logic
        delay = buffered_chunk_params.get("delay", 0.4)
        model_stride_in_secs = buffered_chunk_params.get("model_stride_in_secs", 0.04)
        tokens_per_chunk = buffered_chunk_params.get("tokens_per_chunk", 10)
        
        for audio_filepath in audio_filepaths_batch:
            if hasattr(model, 'reset'):
                model.reset()
            if hasattr(model, 'read_audio_file'):
                model.read_audio_file(audio_filepath, delay, model_stride_in_secs)
            
            # For buffered streaming, transcribe and get logits
            if hasattr(model, 'transcribe') and callable(model.transcribe):
                hyp, logits = model.transcribe(tokens_per_chunk, delay, keep_logits=True)
                log_probs_list_batch.append(logits)
                T_list_batch.append(logits.shape[0])
                pred_text_batch.append(hyp)

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

        # Create utterance object
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