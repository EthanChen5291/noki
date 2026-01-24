from __future__ import annotations

import numpy as np
import librosa
import game.constants as C
import game.models as M
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


STRONG_INTENSITY_THRESHOLD = 70 # melodies / strong beats, louder than 70%
MEDIUM_INTENSITY_THRESHOLD = 40 # louder than 40% 
@dataclass(frozen=True)
class SubBeatInfo():
    """Beat-level timestamps, raw intensities, and normalized intensity levels"""
    time: float
    raw_intensity: float
    level: SubBeatIntensity

class SubBeatIntensity(Enum):
    """Normalized beat-level intensity relative to each beat's respective measure intensity"""
    WEAK = auto()
    MEDIUM = auto()
    STRONG = auto()

@dataclass(frozen=True)
class IntensityProfile:
    """Beat-level and section-level intensity curves (higher = more activity)"""
    beat_intensities: list[float]
    section_intensities: list[float]

def get_bpm(audio_path: str, expected_bpm: Optional[int] = None) -> float:
    """Returns the tempo in BPM. Not fully reliable yet."""
    y, sr = librosa.load(audio_path, sr=None)

    tempo, beat_frames = librosa.beat.beat_track(
        y=y,
        sr=sr,
        start_bpm=expected_bpm if expected_bpm else 120,
        tightness=200
    )

    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    if len(beat_times) > 1:
        intervals = np.diff(beat_times)
        median_interval = np.median(intervals)
        bpm = float(60 / median_interval)
    else:
        bpm = float(tempo[0] if isinstance(tempo, np.ndarray) else tempo)

    print(f"Raw BPM (beat_times): {bpm:.3f}")
    
    return bpm

def normalize_bpm(bpm: float) -> int:
    """Normalizes bpm to a playable-rhythm BPM. Found through testing.
    
    Librosa typically interprets BPM in a triplet ambiguity. This detects against that
    and provides the appropriate correction."""
    if bpm <= 0:
        raise ValueError("Invalid BPM")

    candidates = [
        bpm,
        bpm * 2,
        bpm / 2,
        bpm * 3/2,
        bpm * 2/3,
    ]

    playable = [c for c in candidates if 90 <= c <= 220]

    if playable:
        best = min(playable, key=lambda x: abs(x - 150))
    else:
        best = min(candidates, key=lambda x: abs(x - bpm))

    rounded = int(round(best))

    print(f"Final BPM: {rounded}")
    return int(round(best))

def get_duration(audio_path: str) -> int:
    """Returns the song duration in seconds"""
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    return int(duration)

def get_song_info(audio_path: str, expected_bpm: Optional[int], *, normalize: Optional[bool] =True) -> M.Song:
    """Gets the song at 'audio_path's duration and tempo (BPM).
    
    Can normalize bpm (check for triplet ambiguities) if doing raw BPM detection (no expected_bpm) with 'normalize'."""
    if expected_bpm:
        bpm = get_bpm(audio_path, expected_bpm)
    else:
        bpm = get_bpm(audio_path)
        if normalize:
            bpm = normalize_bpm(bpm)
    
    duration = get_duration(audio_path)
    return M.Song(bpm, duration, audio_path)

def analyze_song_intensity(audio_path: str, bpm: float, beats_per_section: int = 16) -> IntensityProfile:
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

def get_sb_times(beat_times : list[float], n : int = 4) -> list[float]:
    """Returns a list of times of beats which each beat cut into 'n' subdivisions. 
    
    If subdivisions == 4, a list with four values per beat for each beat in beat_times
    would be given.
    """
    #separates onset_times into sixteenth note intervals 
    if not beat_times:
        return []

    sb_times = []
    
    for i in range(len(beat_times) - 1):
        t0, t1 = beat_times[i], beat_times[i + 1]
        for k in range(n):
            sub_t = t0 + (k / n) * (t1 - t0)
            sb_times.append(sub_t)
    
    return sb_times
    
def beat_intensity_at_time(t, onset_env, onset_times, window=0.05):
    "Fetches the beat intensity around interval 'window' of time 't'"
    mask = (onset_times >= t - window) & (onset_times <= t + window)
    return float(onset_env[mask].mean()) if np.any(mask) else 0.0
    
def convert_to_measure_intensities(beat_intensities: list[float]) -> list[float]:
    "Returns average measure intensities for all measures (BEATS_PER_MEASURE) in beat_intensities."
    "Discards remainder."
    if not beat_intensities:
        return []
    
    measure_intensities : list[float] = []
    current_measure_intensity = 0

    for i, intensity in enumerate(beat_intensities):
        if i % C.BEATS_PER_MEASURE == 0:
            measure_intensities.append(current_measure_intensity / C.BEATS_PER_MEASURE)
            current_measure_intensity = 0
        
        current_measure_intensity += intensity

    return measure_intensities

    # what if there is an incomplete measure

def get_sb_intensities(sb_times, onset_env, onset_times, window : float = 0.05) -> list[float]:
    "Returns the all sub_beat_intensities around 'window' of each sub-beat. "
    "A sub-beat has each beat in 'beat_times' cut into 'n' subdivisions"
    sb_intensities : list[float] = []

    if not sb_times:
        return []
    
    for t in sb_times:
        intensity = beat_intensity_at_time(t, onset_env, onset_times, window)
        sb_intensities.append(intensity)
    
    return sb_intensities

def group_sb_intensities(beat_intensities: list[float], n: int) -> list[list[float]]:
    "Groups subbeats into groups of size n. Discards remainder, if any."
    if not beat_intensities:
        return []
    
    group_intensities: list[list[float]] = []
    current_group_intensity: list[float] = []

    for i, intensity in enumerate(beat_intensities):
        if i % n == 0 and i != 0:
            group_intensities.append(current_group_intensity)
            current_group_intensity: list[float] = []
        
        current_group_intensity.append(intensity)

    return group_intensities

def first_greater_than_percentile(lst: list[float], p: float) -> float:
    "Returns minimum value in 'lst' >= the 'p'th percentile. "
    "Does not preserve order."
    if not lst:
        raise ValueError("Cannot compute percentile of an empty list")
    
    sorted_lst = sorted(lst)
    
    idx = int(len(lst) * (p/100))
    idx = min(idx, len(lst) - 1)

    return sorted_lst[idx]

def normalize_sb_intensities(
        beat_times: list[float], 
        beat_intensities: list[float], 
        onset_env: list[float], 
        onset_times: list[float],
        n: int = 4 # sixteen notes
    ) -> list[SubBeatInfo]:
    """
    Splits each beat_times into n slices (sub-beats). 
    
    Returns a list for all sub-beats, each with:
    - sub-beat timestamp
    - raw intensity
    - normalized level (e.g WEAK, MEDIUM, STRONG)
    """
    if not beat_intensities:
        return []
    
    sb_times = get_sb_times(beat_times, n)
    sb_intensities = get_sb_intensities(sb_times, onset_env, onset_times)
    group_intensities = group_sb_intensities(sb_intensities, C.BEATS_PER_MEASURE * n) # group into measures

    all_sb_info: list[SubBeatInfo] = []
    sb_index = 0

    for i, group in enumerate(group_intensities):
        if not group:
            continue
        
        strong_threshold = first_greater_than_percentile(group, STRONG_INTENSITY_THRESHOLD)
        medium_threshold = first_greater_than_percentile(group, MEDIUM_INTENSITY_THRESHOLD)

        for intensity in group:
            curr_measure_subbeats = group_intensities[i]

            if intensity >= strong_threshold:
                level = SubBeatIntensity.STRONG
            elif intensity >= medium_threshold:
                level = SubBeatIntensity.MEDIUM
            else:
                level = SubBeatIntensity.WEAK # add empty for silence?
            
            time = sb_times[sb_index]
            sb_index += 1

            sb_info = SubBeatInfo(time, intensity, level)
            all_sb_info.append(sb_info)
    
    return all_sb_info

def filter_sb_info(sb_info: list[SubBeatInfo], level: SubBeatIntensity) -> list[SubBeatInfo]:
    """Returns all intensities in sb_info that matches level"""
    return [i for i in sb_info if i.level == level]

#
    
# THEN, DO FREQUENCY CHECK ON THE NORMALIZED SB INTENSITIES. maybe split into categories based off wavelength ranges?
    

    


    
# check if difference between last exceeds THRESHOLD. how to calculate threshold?
# check average measure beat intensity through beat_intensities. 

# def get_peaks() -> finds onset peaks at every sixteenth note that exceed a certain threshold 
# (threshold should be calculated based off avg onset difference i think)

# def get_frequencies(raw_sound: , ) -> 
# finds audio frequencies and lists the frequencies in terms of sixteenth notes
# how to normalize frequency ? do we normalize or just categorize (low, medium, high) by raw values?

# i aim to figure out the time slots where words CAN go in the regular beatmap_generator math stuff,
# then use this stuff to kinda map out WHERE in those time slots i can put chars
# this constrains the chars to recommended CPS while ensuring musical feel

# get silence

