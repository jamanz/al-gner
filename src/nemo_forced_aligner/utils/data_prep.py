# nemo_forced_aligner/utils/data_prep.py

import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

import soundfile as sf

from nemo.collections.asr.parts.utils.streaming_utils import FrameBatchASR
from nemo.utils import logging
import librosa
from nemo.collections.asr.models.ctc_models import EncDecCTCModel

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
    manifest_lines_batch,
    model: EncDecCTCModel,
    separator: str,
    align_using_pred_text: bool,
    audio_filepath_parts_in_utt_id,
    output_timestep_duration,
    simulate_cache_aware_streaming: bool = False,
    use_buffered_chunked_streaming: bool = False,
    buffered_chunk_params: dict = None,
):
    """
    Returns:
        log_probs_batch, y_batch, T_batch, U_batch, utt_obj_batch, output_timestep_duration
    """

    buffered_chunk_params = buffered_chunk_params or {}
    audio_paths = [line["audio_filepath"] for line in manifest_lines_batch]
    B = len(audio_paths)
    log_probs_list, T_list, pred_text = [], [], []

    # 1) Collect log-probs & (optionally) ASR text
    if not use_buffered_chunked_streaming:
        with torch.no_grad():
            if not simulate_cache_aware_streaming:
                hyps = model.transcribe(audio_paths, return_hypotheses=True, batch_size=B)
            else:
                hyps = model.transcribe_simulate_cache_aware_streaming(
                    audio_paths, return_hypotheses=True, batch_size=B
                )
        # Hybrid models return (best, nbest), so unpack if needed
        if isinstance(hyps, tuple) and len(hyps) == 2:
            hyps = hyps[0]
        for h in hyps:
            log_probs_list.append(h.y_sequence)       # Tensor: (T_i, V)
            T_list.append(h.y_sequence.shape[0])
            pred_text.append(h.text)
    else:
        delay  = buffered_chunk_params["delay"]
        stride = buffered_chunk_params["model_stride_in_secs"]
        toks   = buffered_chunk_params["tokens_per_chunk"]
        for path in tqdm(audio_paths, desc="Streaming chunks"):
            model.reset()
            model.read_audio_file(path, delay, stride)
            hyp, logits = model.transcribe(toks, delay, keep_logits=True)
            log_probs_list.append(logits)             # Tensor: (T_i, V)
            T_list.append(logits.shape[0])
            pred_text.append(hyp)

    # 2) Build y/U and Utterance objects
    y_list, U_list, utt_objs = [], [], []
    for i, line in enumerate(manifest_lines_batch):
        gt = pred_text[i] if align_using_pred_text else line["text"]
        utt = get_utt_obj(
            gt,
            model,
            separator,
            T_list[i],
            audio_paths[i],
            _get_utt_id(audio_paths[i], audio_filepath_parts_in_utt_id),
        )
        # preserve original text if aligning on predictions
        if align_using_pred_text:
            utt.pred_text = pred_text[i]
            utt.text = line.get("text", "")
        else:
            utt.text = line.get("text", "")
        y_list.append(utt.token_ids_with_blanks)
        U_list.append(len(utt.token_ids_with_blanks))
        utt_objs.append(utt)

    # 3) Pad into batch tensors
    T_max = max(T_list)
    U_max = max(U_list)
    V = len(getattr(model, "tokenizer", model.decoder).vocab) + 1
    T_batch = torch.tensor(T_list)
    U_batch = torch.tensor(U_list)

    # log_probs: (B, T_max, V)
    log_probs_batch = V_NEGATIVE_NUM * torch.ones((B, T_max, V))
    for b, lp in enumerate(log_probs_list):
        log_probs_batch[b, : lp.shape[0], :] = lp

    # y_batch: (B, U_max)
    y_batch = V * torch.ones((B, U_max), dtype=torch.int64)
    for b, y in enumerate(y_list):
        y_batch[b, : len(y)] = torch.tensor(y, dtype=torch.int64)

    # 4) Compute timestep duration if needed
    if output_timestep_duration is None:
        pre = model.cfg.preprocessor
        if "window_stride" not in pre or "sample_rate" not in pre:
            raise ValueError("Cannot compute timestep duration without window_stride and sample_rate")
        with sf.SoundFile(audio_paths[0]) as f:
            audio_dur = f.frames / f.samplerate
        n_in = audio_dur / pre.window_stride
        down = round(n_in / int(T_batch[0]))
        hop = model.preprocessor.featurizer.hop_length
        sr  = pre.sample_rate
        output_timestep_duration = (hop * down) / sr
        logging.info(f"Computed timestep duration: {output_timestep_duration}s")

    return (
        log_probs_batch,
        y_batch,
        T_batch,
        U_batch,
        utt_objs,
        output_timestep_duration,
    )


def _get_utt_id(audio_filepath, audio_filepath_parts_in_utt_id):
    fp_parts = Path(audio_filepath).parts[-audio_filepath_parts_in_utt_id:]
    utt_id = Path("_".join(fp_parts)).stem
    utt_id = utt_id.replace(" ", "-")  # replace any spaces in the filepath with dashes
    return utt_id


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