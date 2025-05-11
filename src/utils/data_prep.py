# nemo_forced_aligner/utils/data_prep.py

import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from nemo.collections.asr.parts.utils.streaming_utils import FrameBatchASR
from nemo.utils import logging


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
    """Create utterance object with alignment information"""

    text = manifest_line.get('text', '').strip()
    audio_filepath = manifest_line.get('audio_filepath', '')

    # Create basic utterance object
    utt_obj = {
        'utt_id': utt_id,
        'text': text,
        'audio_filepath': audio_filepath,
        'log_probs': log_probs,
        'output_timestep_duration': output_timestep_duration
    }

    # Process segments if separator is provided
    if segment_grouping_separator:
        segments = [s.strip() for s in text.split(segment_grouping_separator) if s.strip()]
    else:
        segments = [text]

    # Create segment information
    segments_info = []
    for i, segment_text in enumerate(segments):
        segment_info = {
            'segment_id': i,
            'text': segment_text,
            'segment_token_span': (0, 0)  # Will be calculated later
        }
        segments_info.append(segment_info)

    utt_obj['segments_utt_info'] = segments_info

    # Tokenize text for alignment
    if hasattr(model, 'tokenizer'):
        # Use model's tokenizer
        tokens = model.tokenizer.text_to_ids(text)
    else:
        # Fallback to character-level tokenization
        tokens = []
        for char in text:
            if hasattr(model.decoder, 'vocabulary') and char in model.decoder.vocabulary:
                tokens.append(model.decoder.vocabulary.index(char))
            else:
                # Use blank token for unknown characters
                tokens.append(len(model.decoder.vocabulary))

    utt_obj['tokens'] = tokens

    # Create token-level details
    token_details = []
    for i, token in enumerate(tokens):
        token_info = {
            'token_index': i,
            'token': token,
            'aligned_w_token': None  # Will be filled by alignment
        }
        token_details.append(token_info)

    utt_obj['token_level_details'] = token_details

    return utt_obj


def get_batch_variables(
        manifest_lines_batch: List[Dict],
        model,
        additional_segment_grouping_separator: Optional[str],
        align_using_pred_text: bool,
        audio_filepath_parts_in_utt_id: int,
        output_timestep_duration: Optional[float],
        simulate_cache_aware_streaming: bool = False,
        use_buffered_infer: bool = False,
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

    log_probs_batch = []
    y_batch = []
    T_batch = []
    U_batch = []
    utt_obj_batch = []

    # Process each manifest line
    for manifest_line in manifest_lines_batch:
        audio_filepath = manifest_line['audio_filepath']

        # Generate utterance ID from audio filepath
        audio_path = Path(audio_filepath)
        parts = audio_path.parts
        utt_id = str(audio_path.stem).replace(' ', '_')

        # Get log probabilities from model
        if use_buffered_infer and isinstance(model, FrameBatchASR):
            # Use buffered streaming inference
            log_probs = model.transcribe([audio_filepath], return_hypotheses=True)[0].log_probs
        else:
            # Standard inference
            logits = model.transcribe([audio_filepath], logprobs=True, return_hypotheses=True)[0].logprobs
            log_probs = torch.tensor(logits)

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
            manifest_line,
            utt_id,
            additional_segment_grouping_separator,
            model,
            log_probs,
            output_timestep_duration
        )

        # Get token sequence for alignment
        y = utt_obj['tokens']

        # Add to batch
        log_probs_batch.append(log_probs)
        y_batch.append(y)
        T_batch.append(log_probs.shape[0])
        U_batch.append(len(y))
        utt_obj_batch.append(utt_obj)

    return log_probs_batch, y_batch, T_batch, U_batch, utt_obj_batch, output_timestep_duration


def add_t_start_end_to_utt_obj(
        utt_obj: Dict,
        alignment: List[int],
        output_timestep_duration: float
) -> Dict:
    """Add start and end times to utterance object based on alignment"""

    # Process token alignments
    for i, token_info in enumerate(utt_obj['token_level_details']):
        if i < len(alignment):
            # Find token boundaries in alignment
            start_idx = alignment.index(i) if i in alignment else None

            if start_idx is not None:
                # Find end of this token
                end_idx = start_idx
                while end_idx < len(alignment) and alignment[end_idx] == i:
                    end_idx += 1

                # Store alignment info
                token_info['aligned_w_token'] = (i, start_idx, end_idx)

    # Process segment alignments
    for segment_info in utt_obj.get('segments_utt_info', []):
        # Find segment boundaries based on tokens
        segment_start = float('inf')
        segment_end = 0

        start_token, end_token = segment_info.get('segment_token_span', (0, 0))

        for token_idx in range(start_token, end_token):
            if token_idx < len(utt_obj['token_level_details']):
                token_data = utt_obj['token_level_details'][token_idx]
                if 'aligned_w_token' in token_data:
                    _, t_start, t_end = token_data['aligned_w_token']
                    segment_start = min(segment_start, t_start)
                    segment_end = max(segment_end, t_end)

        if segment_start < float('inf'):
            segment_info['aligned_w_segment'] = (segment_info['segment_id'], segment_start, segment_end)

    return utt_obj