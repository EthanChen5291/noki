from __future__ import annotations

import numpy as np
import librosa
import game.constants as C
import game.models as M
from typing import Optional

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
    """
    Normalizes bpm to a playable-rhythm BPM. Found through testing.
    
    Librosa typically interprets BPM in a triplet ambiguity. This detects against that
    and provides the appropriate correction.
    """
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
    return rounded

def get_duration(audio_path: str) -> int:
    """Returns the song duration in seconds"""
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    return int(duration)

def find_downbeat_offset(beat_times: np.ndarray, onset_env: np.ndarray, onset_times: np.ndarray) -> int:
    """
    Find which beat position (0-3) is consistently the downbeat across multiple measures.
    Uses statistical analysis over multiple measures for more reliable detection.
    Returns the offset (0-3) to align beats to measures.
    """
    if len(beat_times) < 8:
        return 0

    # Analyze multiple measures (up to 8) to find consistent downbeat pattern
    num_measures = min(8, len(beat_times) // 4)

    # Accumulate intensity for each beat position (0-3) across measures
    position_intensities = [[] for _ in range(4)]

    for measure_idx in range(num_measures):
        for beat_pos in range(4):
            beat_idx = measure_idx * 4 + beat_pos
            if beat_idx >= len(beat_times):
                break

            bt = beat_times[beat_idx]
            # Find onset intensity near this beat
            mask = (onset_times >= bt - 0.05) & (onset_times <= bt + 0.05)
            if np.any(mask):
                intensity = float(np.max(onset_env[mask]))
            else:
                intensity = 0.0
            position_intensities[beat_pos].append(intensity)

    # Calculate mean intensity for each beat position
    mean_intensities = []
    for pos in range(4):
        if position_intensities[pos]:
            mean_intensities.append(np.mean(position_intensities[pos]))
        else:
            mean_intensities.append(0.0)

    # The position with highest mean intensity is likely the downbeat
    downbeat_pos = int(np.argmax(mean_intensities))

    # The offset is how many beats to skip so that beat 1 aligns to index 0
    # If downbeat is at position 2, we need to skip 2 beats
    return downbeat_pos


def get_song_info(audio_path: str, expected_bpm: Optional[int], *, normalize: Optional[bool] =True) -> M.Song:
    """Gets the song at 'audio_path's duration and tempo (BPM).

    Can normalize bpm (check for triplet ambiguities) if doing raw BPM detection (no expected_bpm) with 'normalize'.
    Also extracts actual beat timestamps from librosa for drift-free rendering.
    Aligns beats to measures by finding the downbeat."""
    if expected_bpm:
        bpm = get_bpm(audio_path, expected_bpm)
    else:
        bpm = get_bpm(audio_path)
        if normalize:
            bpm = normalize_bpm(bpm)

    # Extract actual beat times from librosa (drift-free)
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)

    _, beat_frames = librosa.beat.beat_track(y=y, sr=sr, start_bpm=bpm, tightness=200)
    beat_times_raw = librosa.frames_to_time(beat_frames, sr=sr)

    # Find downbeat alignment
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr)
    downbeat_offset = find_downbeat_offset(beat_times_raw, onset_env, onset_times)

    # Offset beat times so measure lines align correctly
    # Skip the first `downbeat_offset` beats so index 0 is a downbeat
    aligned_beat_times = beat_times_raw[downbeat_offset:].tolist()

    return M.Song(bpm, int(duration), audio_path, aligned_beat_times)

def get_beat_intensities(beat_times: np.ndarray, onset_times: np.ndarray, onset_env: np.ndarray) -> list[float]:
    beat_intensities = []
    for i in range(len(beat_times) - 1):
        start_t, end_t = beat_times[i], beat_times[i+1]
        mask = (onset_times >= start_t) & (onset_times < end_t)
        avg_strength = onset_env[mask].mean() if np.any(mask) else 0.0
        beat_intensities.append(avg_strength)
    
    return beat_intensities

def get_section_intensities(beat_intensities : list[float], beats_per_section: int = 16) -> list[float]:
    section_intensities: list[float] = []
    num_sections = len(beat_intensities) // beats_per_section

    for sec_idx in range(num_sections):
        a = sec_idx * beats_per_section
        b = a + beats_per_section
        sec_vals = beat_intensities[a:b]
        section_intensities.append(float(np.mean(sec_vals)) if sec_vals else 0.0)
    
    return section_intensities
    
def analyze_song_intensity(audio_path: str, bpm: float, beats_per_section: int = 16) -> M.IntensityProfile:
    """Given a song's file path, returns the intensity at each beat and
    average intensity per section (defined by "beats_per_section")"""
    y, sr = librosa.load(audio_path, sr=None)

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, start_bpm=bpm)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr)

    beat_intensities = get_beat_intensities(beat_times, onset_times, onset_env)

    #group intensities to sections
    section_intensities: list[float] = get_section_intensities(beat_intensities, beats_per_section)

    return M.IntensityProfile(
        beat_intensities=beat_intensities,
        section_intensities=section_intensities
    )

def get_measure_intensities(beats: list[float], beats_per_measure: int) -> list[float]:
    """Calculates the average measure intensity (defined by 'beats_per_measure')."""
    return [
        sum(beats[i:i + beats_per_measure]) / len(beats[i:i + beats_per_measure])
        for i in range(0, len(beats), beats_per_measure)
    ]

def detect_drops(
    beat_times: list[float],
    audio_path: str,
    bpm: float,
    beats_per_section: int = 8
) -> list[M.DropEvent]:
    """
    Detect multiple drops/climaxes using accurate beat-frame-based section intensities.
    Returns all drops that exceed a dynamic threshold based on the song's intensity profile.
    """
    if len(beat_times) < beats_per_section * 2:
        return []

    y, sr = librosa.load(audio_path, sr=None)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr)

    # Calculate beat intensities using actual beat times
    beat_intensities = get_aggro_beat_intensities(
        np.array(beat_times), onset_times, onset_env
    )

    # Group into sections using actual beat times
    num_sections = len(beat_intensities) // beats_per_section
    section_intensities: list[float] = []
    section_start_times: list[float] = []

    for sec_idx in range(num_sections):
        beat_start = sec_idx * beats_per_section
        beat_end = beat_start + beats_per_section

        sec_beats = beat_intensities[beat_start:beat_end]
        section_intensities.append(float(np.max(sec_beats)) if sec_beats else 0.0)

        # Use actual beat time for section start
        if beat_start < len(beat_times):
            section_start_times.append(beat_times[beat_start])

    if len(section_intensities) < 2:
        return []

    avg_intensity = float(np.mean(section_intensities))
    max_intensity = float(np.max(section_intensities))
    intensity_range = max_intensity - avg_intensity

    if avg_intensity <= 1e-6 or intensity_range <= 1e-6:
        return []

    # Dynamic threshold based on song's intensity range
    # Threshold is relative to the song's own dynamics
    drop_threshold = avg_intensity + (intensity_range * 0.4)

    drops: list[M.DropEvent] = []
    cooldown_sections = 3  # Minimum sections between drops
    last_drop_idx = -cooldown_sections

    for i in range(1, len(section_intensities)):
        current = section_intensities[i]
        previous = section_intensities[i - 1]

        # Check for significant intensity jump
        jump = current - previous
        jump_ratio = jump / (avg_intensity + 1e-6)

        # Trigger if: big jump AND current exceeds threshold AND not in cooldown
        if (jump_ratio > 0.25 and
            current > drop_threshold and
            i - last_drop_idx >= cooldown_sections):

            intensity_ratio = current / (avg_intensity + 1e-6)

            drops.append(M.DropEvent(
                timestamp=section_start_times[i] if i < len(section_start_times) else 0.0,
                intensity=current,
                intensity_ratio=intensity_ratio,
                section_idx=i
            ))

            last_drop_idx = i

    return drops


def detect_loudest_drop(audio_path: str, bpm: float, subdivisions: int = 4) -> Optional[M.DropEvent]:
    """
    Legacy function - detects single loudest drop.
    Use detect_drops() for multiple drops.
    """
    y, sr = librosa.load(audio_path, sr=None)

    _, beat_frames = librosa.beat.beat_track(y=y, sr=sr, start_bpm=bpm)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

    drops = detect_drops(beat_times, audio_path, bpm)
    return drops[0] if drops else None

def get_sb_times(beat_times : np.ndarray, n : int = 4) -> list[float]:
    """
    Returns a list of times of beats which each beat cut into 'n' subdivisions. 
    
    If subdivisions == 4, a list with four values per beat for each beat in beat_times
    would be given.
    """
    #separates onset_times into sixteenth note intervals 
    if not beat_times.any():
        return []

    sb_times = []
    
    for i in range(len(beat_times) - 1):
        t0, t1 = beat_times[i], beat_times[i + 1]
        for k in range(n):
            sub_t = t0 + (k / n) * (t1 - t0)
            sb_times.append(sub_t)
    
    return sb_times
    
def beat_intensity_at_time(t, onset_env, onset_times, window=0.05):
    """Fetches the beat intensity around interval 'window' of time 't'"""
    mask = (onset_times >= t - window) & (onset_times <= t + window)
    return float(onset_env[mask].mean()) if np.any(mask) else 0.0
    
def convert_to_measure_intensities(beat_intensities: list[float]) -> list[float]:
    """
    Returns average measure intensities for all measures (BEATS_PER_MEASURE) 
    in beat_intensities. Discards remainder.
    """
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
    """
    Returns the all sub_beat_intensities around 'window' of each sub-beat. 
    A sub-beat has each beat in 'beat_times' cut into 'n' subdivisions
    """
    sb_intensities : list[float] = []

    if not sb_times:
        return []
    
    for t in sb_times:
        intensity = beat_intensity_at_time(t, onset_env, onset_times, window)
        sb_intensities.append(intensity)
    
    return sb_intensities

def group_sb_intensities(beat_intensities: list[float], n: int) -> list[list[float]]:
    """Groups subbeats into groups of size n. Discards remainder, if any."""
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
    """Returns minimum value in 'lst' >= the 'p'th percentile.
    Does not preserve order."""
    if not lst:
        raise ValueError("Cannot compute percentile of an empty list")
    
    sorted_lst = sorted(lst)
    
    idx = int(len(lst) * (p/100))
    idx = min(idx, len(lst) - 1)

    return sorted_lst[idx]

def normalize_sb_intensities(
        beat_times: np.ndarray, 
        beat_intensities: list[float], 
        onset_env: np.ndarray, 
        onset_times: np.ndarray,
        n: int = 4 # sixteen notes
    ) -> list[M.SubBeatInfo]:
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

    all_sb_info: list[M.SubBeatInfo] = []
    sb_index = 0

    for i, group in enumerate(group_intensities):
        if not group:
            continue
        
        strong_threshold = first_greater_than_percentile(group, C.STRONG_INTENSITY_THRESHOLD)
        medium_threshold = first_greater_than_percentile(group, C.MEDIUM_INTENSITY_THRESHOLD)

        for intensity in group:
            curr_measure_subbeats = group_intensities[i]

            if intensity >= strong_threshold:
                level = M.SubBeatIntensity.STRONG
            elif intensity >= medium_threshold:
                level = M.SubBeatIntensity.MEDIUM
            else:
                level = M.SubBeatIntensity.WEAK # add empty for silence?
            
            time = sb_times[sb_index]
            sb_index += 1

            sb_info = M.SubBeatInfo(time, intensity, level)
            all_sb_info.append(sb_info)
    
    return all_sb_info

def get_sb_info(song: M.Song, subdivisions: int) -> list[M.SubBeatInfo]:
    """Returns (timestamp, raw_intensity, level) for each sub_beat in 'song'.
    Each sub-beat is 1/subdivisions of a beat."""
    y, sr = librosa.load(song.file_path, sr=None)

    _, beat_frames = librosa.beat.beat_track(
        y=y,
        sr=sr,
        start_bpm=song.bpm if song.bpm else 120,
        tightness=200
    )

    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr)

    beat_intensities = get_beat_intensities(beat_times, onset_times, onset_env)

    return normalize_sb_intensities(beat_times, beat_intensities, onset_env, onset_times, subdivisions)

def group_info_by_section(sb_info: list[M.SubBeatInfo], subdivisions: int = 4, beats_per_section: int = 16) -> list[list[M.SubBeatInfo]]:
    """Groups sb_info by sections (defined by beats_per_section), given 'subdivisions' per beat.
    Defaults to sixteen notes (4 subdivisions per beat) and 4/4 time signature (16 beats per section)"""
    if not sb_info: 
        return []
    
    total_per_section = subdivisions * beats_per_section

    sections_sb_info: list[list[M.SubBeatInfo]] = []
    current_section: list[M.SubBeatInfo] = []
    current_idx = 0

    for sb in sb_info:
        if current_idx >= total_per_section:
            sections_sb_info.append(current_section)
            current_idx = 0
            current_section = []
        
        current_section.append(sb)
        current_idx += 1
    
    if current_section:
        sections_sb_info.append(current_section)

    return sections_sb_info

def filter_sb_info(sb_info: list[M.SubBeatInfo], level: M.SubBeatIntensity) -> list[M.SubBeatInfo]:
    """Returns all intensities in sb_info that matches level"""
    return [i for i in sb_info if i.level == level]

def num_of_level(sb_info: list[M.SubBeatInfo], level: M.SubBeatIntensity) -> int:
    """Returns the number of sub-beats in sb_info that match 'level'"""
    return len(filter_sb_info(sb_info, level))

def classify_pace(audio_path: str, bpm: float) -> M.PaceProfile:
    """
    Analyze audio features to classify song pace and compute scroll multiplier.

    Uses:
    - BPM
    - onset density (rhythmic activity)
    - intensity (loudness)
    - intensity variance (dynamic range)

    Returns PaceProfile with pace_score in [0, 1].
    """
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=onset_env)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    onset_density = len(onset_times) / duration if duration > 0 else 0.0

    avg_intensity = float(np.mean(onset_env)) if len(onset_env) > 0 else 0.0
    intensity_variance = float(np.var(onset_env)) if len(onset_env) > 0 else 0.0

    bpm_score = np.clip((bpm - 70) / 150, 0, 1)
    density_score = np.clip((onset_density - 1.2) / 6.0, 0, 1)
    intensity_score = np.clip(avg_intensity / 40.0, 0, 1)
    variance_score = np.clip(intensity_variance / 300.0, 0, 1)

    pace_score = float(
        bpm_score * 0.25 +
        density_score * 0.45 +
        intensity_score * 0.15 +
        variance_score * 0.15
    )

    scroll_multiplier = 0.75 + (pace_score * 0.75)

    return M.PaceProfile(
        bpm=bpm,
        onset_density=onset_density,
        avg_intensity=avg_intensity,
        intensity_variance=intensity_variance,
        pace_score=pace_score,
        scroll_multiplier=scroll_multiplier
    )

def get_aggro_beat_intensities(beat_times, onset_times, onset_env):
    intensities = []

    for i in range(len(beat_times) - 1):
        start_t, end_t = beat_times[i], beat_times[i + 1]
        mask = (onset_times >= start_t) & (onset_times < end_t)

        if np.any(mask):
            val = float(np.max(onset_env[mask]))
        else:
            val = 0.0

        intensities.append(val)

    return intensities

def calculate_energy_shifts(
    audio_path: str,
    bpm: float,
    pace_score: float,
    aligned_beat_times: list[float],
    beats_per_section: int = 8
) -> list[M.SectionEnergyShift]:
    """
    Calculate dynamic scroll speed shifts using aligned beat times.
    Shifts persist until intensity drops noticeably below the trigger level.
    Uses the same beat_times as measure lines for perfect sync.
    """
    if len(aligned_beat_times) < beats_per_section * 2:
        return []

    y, sr = librosa.load(audio_path, sr=None)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr)

    beat_times = np.array(aligned_beat_times)
    beat_intensities = get_aggro_beat_intensities(beat_times, onset_times, onset_env)

    num_sections = len(beat_intensities) // beats_per_section
    section_intensities: list[float] = []
    section_times: list[tuple[float, float]] = []

    for sec_idx in range(num_sections):
        beat_start = sec_idx * beats_per_section
        beat_end = beat_start + beats_per_section

        sec_beats = beat_intensities[beat_start:beat_end]
        section_intensities.append(float(np.mean(sec_beats)) if sec_beats else 0.0)

        start_t = float(beat_times[beat_start]) if beat_start < len(beat_times) else 0.0
        end_t = float(beat_times[min(beat_end, len(beat_times) - 1)])
        section_times.append((start_t, end_t))

    if len(section_intensities) < 2:
        return []

    avg_intensity = float(np.mean(section_intensities))
    if avg_intensity <= 1e-6:
        return []

    base_threshold = 0.10 - (pace_score * 0.04)
    base_threshold = max(0.04, base_threshold)

    drop_threshold = 0.85

    shifts: list[M.SectionEnergyShift] = []
    in_shift = False
    shift_start_idx = -1
    shift_trigger_intensity = 0.0
    shift_modifier = 1.0
    shift_delta = 0.0

    for i in range(1, len(section_intensities)):
        current = section_intensities[i]
        previous = section_intensities[i - 1]

        delta = (current - previous) / avg_intensity
        ratio = current / (avg_intensity + 1e-6)

        strong_jump = abs(delta) > base_threshold * 1.2
        medium_jump = abs(delta) > base_threshold
        climax = ratio > (1.10 + 0.15 * pace_score)

        trigger = strong_jump or climax or (medium_jump and ratio > 1.0)

        if not in_shift:
            if trigger and delta > 0:
                in_shift = True
                shift_start_idx = i
                shift_trigger_intensity = current
                shift_delta = delta

                magnitude = max(abs(delta), ratio - 1.0)
                magnitude = magnitude ** (1.2 + 0.6 * pace_score)

                shift_modifier = 1.0 + magnitude * (1.5 + 1.2 * pace_score)
                shift_modifier = float(np.clip(shift_modifier, 1.1, 3.0))
        else:
            intensity_ratio = current / (shift_trigger_intensity + 1e-6)

            if intensity_ratio < drop_threshold or delta < -base_threshold * 1.5:
                start_time = section_times[shift_start_idx][0]
                end_time = section_times[i - 1][1]

                shifts.append(M.SectionEnergyShift(
                    section_idx=shift_start_idx,
                    start_time=start_time,
                    end_time=end_time,
                    energy_delta=float(shift_delta),
                    scroll_modifier=shift_modifier
                ))

                in_shift = False

                if delta < -base_threshold:
                    slow_magnitude = abs(delta) ** 1.2
                    slow_modifier = 1.0 - slow_magnitude * (0.6 + 0.4 * pace_score)
                    slow_modifier = float(np.clip(slow_modifier, 0.5, 0.95))

                    shifts.append(M.SectionEnergyShift(
                        section_idx=i,
                        start_time=section_times[i][0],
                        end_time=section_times[i][1],
                        energy_delta=float(delta),
                        scroll_modifier=slow_modifier
                    ))

    if in_shift and shift_start_idx >= 0:
        start_time = section_times[shift_start_idx][0]
        end_time = section_times[-1][1]

        shifts.append(M.SectionEnergyShift(
            section_idx=shift_start_idx,
            start_time=start_time,
            end_time=end_time,
            energy_delta=float(shift_delta),
            scroll_modifier=shift_modifier
        ))

    return shifts





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