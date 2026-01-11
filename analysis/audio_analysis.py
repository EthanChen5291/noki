import numpy as np
import librosa
from dataclasses import dataclass

@dataclass
class IntensityProfile:
    """A data type holding beat and section intensity profiles"""
    beat_intensities: list[float]
    section_intensities: list[float]

def analyze_song_intensity(audio_path: str, bpm: int, beats_per_section: int = 16) -> IntensityProfile:
    y, sr = librosa.load(audio_path, sr=None)
    tempo, beat_frames = librosa.beat.beat_track(y, sr=sr, start_bpm=bpm)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    onset_env = librosa.onset.onset_strength(y, sr=sr)
    onset_times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr)
   
    # FIND AVERAGE (UNTESTED)
    intensities = []
    for i in range(len(beat_times) - 1):
        start_t, end_t = beat_times[i], beat_times[i+1]

        mask = (onset_times >= start_t) & (onset_times < end_t)
        avg_strength = onset_env[mask].mean() if np.any(mask) else 0.0
        intensities.append(avg_strength)
    
    #group intensities to sections
    section_intensities = []
    beats_per_sec = bpm / 60.0
    section_length_sec = (beats_per_section / beats_per_sec) #idk if i need 

    for sec_idx in range(int(len(intensities) / beats_per_section)):
        sec_vals = intensities[sec_idx * beats_per_section : (sec_idx+1) * beats_per_section]
        section_intensities.append(np.mean(sec_vals) if sec_vals else 0.0)
    return IntensityProfile(intensities, section_intensities)

