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

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, start_bpm=bpm)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
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

#def get_energy_trend(profile: IntensityProfile) -> list[int]:
# - take intensity profile, section the beat_intensities into measures (4 beats), 
# - calculate average measure intensity, set it to 1
# - normalize all the measure densities to avg and output a list of ratios

def get_sub_beat_times(beat_times : list[float], n : int = 4) -> None | list[float]:
    """Returns a list of times of beats which each beat cut into 'n' subdivisions. 
    
    If subdivisions == 4, a list with four values per beat for each beat in beat_times
    would be given.
    """
    #separates onset_times into sixteenth note intervals 
    if not beat_times:
        return None

    sub_beat_times = []
    
    for i in range(len(beat_times) - 1):
        t0, t1 = beat_times[i], beat_times[i + 1]
        for k in range(n):
            sub_t = t0 + (k / n) * (t1 - t0)
            sub_beat_times.append(sub_t)
    
    return sub_beat_times
    
def beat_intensity_at_time(t, onset_env, onset_times, window=0.05):
    "Fetches the beat intensity around interval 'window' of time 't'"
    mask = (onset_times >= t - window) & (onset_times <= t + window)
    return float(onset_env[mask].mean()) if np.any(mask) else 0.0

def get_sub_beat_intensities(beat_times, onset_env, onset_times, n: int = 4, window : float = 0.05) -> None | list:
    "Returns the all sub_beat_intensities around 'window' of each sub-beat. "
    "A sub-beat has each beat in 'beat_times' cut into 'n' subdivisions"
    sub_beat_times = get_sub_beat_times(beat_times, n)

    sb_intensities : list[float] = []

    if not sub_beat_times:
        return None
    
    for t in sub_beat_times:
        intensity = beat_intensity_at_time(t, onset_env, onset_times, window)
        sb_intensities.append(intensity)
    
    return sb_intensities
    
# def get_peaks() -> finds onset peaks at every sixteenth note that exceed a certain threshold 
# (threshold should be calculated based off avg onset difference i think)

# def get_frequencies(raw_sound: , ) -> 
# finds audio frequencies and lists the frequencies in terms of sixteenth notes
# how to normalize frequency ? do we normalize or just categorize (low, medium, high) by raw values?

# i aim to figure out the time slots where words CAN go in the regular beatmap_generator math stuff,
# then use this stuff to kinda map out WHERE in those time slots i can put chars
# this constrains the chars to recommended CPS while ensuring musical feel

# get silence

