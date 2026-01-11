from __future__ import annotations

import numpy as np
import librosa
from dataclasses import dataclass

@dataclass(frozen=True)
class IntensityProfile:
    """Beat-level and section-level intensity curves (higher = more activity)"""
    beat_intensities: list[float]
    section_intensities: list[float]

def analyze_song_intensity(audio_path: str, bpm: int, beats_per_section: int = 16) -> IntensityProfile:
    y, sr = librosa.load(audio_path, sr=None)

    tempo, beat_frames = librosa.beat.beat_track(y, sr=sr, start_bpm=bpm)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    onset_env = librosa.onset.onset_strength(y, sr=sr)
    onset_times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr)
   
    # FIND AVERAGE (UNTESTED)
    beat_intensities = []
    for i in range(len(beat_times) - 1):
        start_t, end_t = beat_times[i], beat_times[i+1]
        mask = (onset_times >= start_t) & (onset_times < end_t)
        avg_strength = onset_env[mask].mean() if np.any(mask) else 0.0
        beat_intensities.append(avg_strength)
    
    #group intensities to sections
    section_intensities: list[float] = []
    num_sections = len(beat_intensities) // beats_per_section

    for sec_idx in range(num_sections):
        a = sec_idx * beats_per_section
        b = a + beats_per_section
        sec_vals = beat_intensities[a:b]
        section_intensities.append(float(np.mean(sec_vals)) if sec_vals else 0.0)
    
    return IntensityProfile(
        beat_intensities=beat_intensities, 
        section_intensities=section_intensities
    )

