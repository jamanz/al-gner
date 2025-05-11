# nemo_forced_aligner/utils/srt_utils.py

import numpy as np
from typing import List, Dict, Any


def format_srt_time(seconds: float) -> str:
    """Format seconds to SRT time format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds - int(seconds)) * 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"


def extract_text_segments_from_alignment(utt_obj: Dict, alignment: List[int], output_timestep_duration: float) -> List[
    Dict]:
    """Extract text segments with timestamps from alignment"""
    segments = []

    # Use entire text as a single segment for simplified output
    text = utt_obj.get('text', '')
    
    if len(alignment) > 0 and len(text.strip()) > 0:
        # Find first and last non-blank token in alignment
        start_time = 0
        end_time = len(alignment) * output_timestep_duration
        
        # If the text contains segment separators, split into multiple segments
        if '|' in text:
            parts = [p.strip() for p in text.split('|') if p.strip()]
            # Estimate segment boundaries based on text length proportions
            total_length = sum(len(p) for p in parts)
            current_time = 0
            
            for part in parts:
                # Estimate timing based on text length proportion
                part_duration = (len(part) / total_length) * end_time
                
                segments.append({
                    'text': part,
                    'start': current_time,
                    'end': current_time + part_duration
                })
                current_time += part_duration
        else:
            # Single segment for entire text
            segments.append({
                'text': text,
                'start': start_time,
                'end': end_time
            })
    
    return segments


def merge_adjacent_segments(segments: List[Dict], max_gap: float = 0.5) -> List[Dict]:
    """Merge adjacent segments that are close together"""
    if not segments:
        return []

    merged = []
    current_segment = segments[0].copy()

    for i in range(1, len(segments)):
        next_segment = segments[i]

        # Check if segments should be merged
        gap = next_segment['start'] - current_segment['end']

        if gap <= max_gap:
            # Merge segments
            current_segment['text'] += ' ' + next_segment['text']
            current_segment['end'] = next_segment['end']
        else:
            # Save current and start new
            merged.append(current_segment)
            current_segment = next_segment.copy()

    # Don't forget the last segment
    merged.append(current_segment)

    return merged


def generate_srt_from_alignment(alignments: List[Dict]) -> str:
    """
    Convert alignment results to SRT format

    Args:
        alignments: List of alignment results from NeMo forced aligner

    Returns:
        SRT content as string
    """
    all_segments = []

    # Process each alignment result
    for align_data in alignments:
        utt_obj = align_data['utt_obj']
        alignment = align_data['alignment']
        output_timestep_duration = align_data['output_timestep_duration']

        # Extract segments
        segments = extract_text_segments_from_alignment(utt_obj, alignment, output_timestep_duration)
        all_segments.extend(segments)

    # Sort segments by start time
    all_segments.sort(key=lambda x: x['start'])

    # Merge adjacent segments for better readability
    merged_segments = merge_adjacent_segments(all_segments)

    # Generate SRT content
    srt_lines = []

    for i, segment in enumerate(merged_segments, 1):
        start_time = format_srt_time(segment['start'])
        end_time = format_srt_time(segment['end'])
        text = segment['text']

        # Skip empty segments
        if not text.strip():
            continue

        srt_lines.append(f"{i}")
        srt_lines.append(f"{start_time} --> {end_time}")
        srt_lines.append(text.strip())
        srt_lines.append("")  # Empty line between entries

    return "\n".join(srt_lines)


def create_simple_srt(text_segments: List[str], duration: float) -> str:
    """
    Create a simple SRT by dividing text evenly across duration
    (fallback when alignment fails)
    """
    if not text_segments:
        return ""

    srt_lines = []
    segment_duration = duration / len(text_segments)

    for i, text in enumerate(text_segments, 1):
        start_time = format_srt_time(i * segment_duration - segment_duration)
        end_time = format_srt_time(i * segment_duration)

        srt_lines.append(f"{i}")
        srt_lines.append(f"{start_time} --> {end_time}")
        srt_lines.append(text.strip())
        srt_lines.append("")

    return "\n".join(srt_lines)