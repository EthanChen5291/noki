import random
from typing import Optional
from analysis.audio_analysis import (
    analyze_song_intensity, 
    get_sb_info, 
    group_info_by_section, 
    filter_sb_info
)
from . import constants as C
from . import models as M

# ==================== SLOT-BASED RHYTHM GENERATION ====================

def build_rhythm_slots(sb_info: list[M.SubBeatInfo], song: M.Song) -> list[M.RhythmSlot]:
    """
    Build rhythm slots from audio analysis.
    A slot is a potential character placement based on musical features.
    """
    slots: list[M.RhythmSlot] = []
    beat_duration = 60 / song.bpm
    
    for i, sb in enumerate(sb_info):
        # Determine if this is a good slot for a character
        is_note_slot = False
        priority = 0
        
        if sb.level == M.SubBeatIntensity.STRONG:
            is_note_slot = True
            priority = 3  # High priority
        elif sb.level == M.SubBeatIntensity.MEDIUM:
            is_note_slot = True
            priority = 2  # Medium priority
        else:
            # Weak beats - only use if needed for rhythm variety
            if i > 0 and slots and (sb.time - slots[-1].time) > 0.4:
                # Fill long gaps with weak beats
                is_note_slot = True
                priority = 1
        
        if is_note_slot:
            slots.append(M.RhythmSlot(
                time=sb.time,
                intensity=sb.raw_intensity,
                priority=priority,
                is_filled=False,
                beat_position=sb.time / beat_duration
            ))
    
    return slots


def filter_slots_for_playability(slots: list[M.RhythmSlot], min_spacing: float = 0.12) -> list[M.RhythmSlot]:
    """
    Remove slots that are too close together for comfortable typing.
    Modern rhythm games maintain ~120-150ms minimum spacing.
    """
    if not slots:
        return []
    
    filtered = [slots[0]]
    
    for slot in slots[1:]:
        if slot.time - filtered[-1].time >= min_spacing:
            filtered.append(slot)
    
    return filtered


def group_slots_by_measure(slots: list[M.RhythmSlot], beat_duration: float) -> list[list[M.RhythmSlot]]:
    """Group rhythm slots into measures (4 beats each)"""
    if not slots:
        return []
    
    measure_duration = beat_duration * C.BEATS_PER_MEASURE
    measures: list[list[M.RhythmSlot]] = []
    current_measure: list[M.RhythmSlot] = []
    current_measure_start = 0.0
    
    for slot in slots:
        measure_num = int(slot.time / measure_duration)
        expected_measure = int(current_measure_start / measure_duration)
        
        if measure_num > expected_measure and current_measure:
            measures.append(current_measure)
            current_measure = []
            current_measure_start = measure_num * measure_duration
        
        current_measure.append(slot)
    
    if current_measure:
        measures.append(current_measure)
    
    return measures


# ==================== SMART WORD ASSIGNMENT ====================

def get_words_with_rhythm_info(words: list[str], beat_duration: float) -> list[M.Word]:
    """Enhanced word creation with better rhythm properties"""
    return [
        M.Word(
            text=word,
            rest_type=None,
            ideal_beats=(ideal := len(word) / C.TARGET_CPS / beat_duration),
            snapped_beats=(snapped := snap_to_grid(ideal, C.SNAP_GRID)),
            snapped_cps=len(word) / (snapped * beat_duration)
        )
        for word in words
    ]


def select_word_for_measure(
    available_slots: int,
    remaining_words: list[M.Word],
    word_bank: list[M.Word],
    measure_intensity: float = 1.0
) -> Optional[M.Word]:
    """
    Select the best word for a measure based on:
    - Number of available rhythm slots
    - Measure intensity (loud = more chars, quiet = fewer)
    - Word variety
    """
    if not remaining_words:
        return None
    
    # Adjust target based on intensity
    target_chars = int(available_slots * measure_intensity)
    target_chars = max(2, min(target_chars, available_slots))
    
    # Find words matching target length ±1
    candidates = [
        w for w in remaining_words 
        if abs(len(w.text) - target_chars) <= 1
    ]
    
    if not candidates:
        # Fallback: any word that fits
        candidates = [w for w in remaining_words if len(w.text) <= available_slots]
    
    if not candidates:
        # Last resort: shortest word
        candidates = [min(remaining_words, key=lambda w: len(w.text))]
    
    # Prefer words with good CPS
    viable = [w for w in candidates if C.MIN_CPS <= w.snapped_cps <= C.MAX_CPS]
    
    return random.choice(viable if viable else candidates)


# ==================== SLOT ASSIGNMENT ====================

def assign_words_to_slots(
    measures: list[list[M.RhythmSlot]],
    word_bank: list[M.Word],
    intensity_profile: Optional[M.IntensityProfile] = None
) -> list[M.CharEvent]:
    """
    Assign characters to rhythm slots measure-by-measure.
    This creates a natural, musical rhythm flow.
    """
    events: list[M.CharEvent] = []
    remaining_words = word_bank.copy()
    section_idx = 0
    
    for measure_idx, measure_slots in enumerate(measures):
        if not measure_slots or not remaining_words:
            continue
        
        # Determine section (every 4 measures = 1 section)
        section_idx = measure_idx // 4
        
        # Get measure intensity if available
        measure_intensity = 1.0
        if intensity_profile and section_idx < len(intensity_profile.section_intensities):
            avg_intensity = sum(intensity_profile.section_intensities) / len(intensity_profile.section_intensities)
            measure_intensity = intensity_profile.section_intensities[section_idx] / (avg_intensity + 1e-6)
        
        # Select word for this measure
        word = select_word_for_measure(
            len(measure_slots),
            remaining_words,
            word_bank,
            measure_intensity
        )
        
        if not word or not word.text:
            continue
        
        # Remove word from remaining
        if word in remaining_words:
            remaining_words.remove(word)
        
        # Assign characters to slots
        chars_to_place = len(word.text)
        
        # Strategy: Use high-priority slots first
        sorted_slots = sorted(measure_slots, key=lambda s: s.priority, reverse=True)
        selected_slots = sorted_slots[:chars_to_place]
        
        # Re-sort by time for proper sequence
        selected_slots.sort(key=lambda s: s.time)
        
        # Create character events
        for char_idx, (char, slot) in enumerate(zip(word.text, selected_slots)):
            events.append(M.CharEvent(
                char=char,
                timestamp=slot.time,
                word_text=word.text,
                char_idx=char_idx,
                beat_position=slot.beat_position,
                section=section_idx,
                is_rest=False
            ))
            slot.is_filled = True
        
        # Add automatic rest after word (except last measure)
        if measure_idx < len(measures) - 1:
            # Rest duration = time until next measure
            next_measure_start = measures[measure_idx + 1][0].time if measure_idx + 1 < len(measures) else selected_slots[-1].time + 1.0
            rest_duration = next_measure_start - selected_slots[-1].time
            
            events.append(M.CharEvent(
                char="",
                timestamp=selected_slots[-1].time,
                word_text="",
                char_idx=-1,
                beat_position=selected_slots[-1].beat_position,
                section=section_idx,
                is_rest=True
            ))
        
        # Recycle words if running low
        if len(remaining_words) < len(word_bank) * 0.3:
            # Add least-used words back
            for w in word_bank:
                if w not in remaining_words:
                    remaining_words.append(w)
    
    return events


# ==================== RHYTHM VARIATIONS ====================

def add_rhythm_variations(events: list[M.CharEvent], song: M.Song) -> list[M.CharEvent]:
    """
    Add modern rhythm game elements:
    - Occasional bursts (fast typing sections)
    - Syncopation (off-beat emphasis)
    - Call-and-response patterns
    """
    if not events:
        return events
    
    # Group by section
    sections = group_events_by_section(events)
    enhanced: list[M.CharEvent] = []
    
    for section in sections:
        # 20% chance for "burst section" - slightly faster, more intense
        if random.random() < 0.2:
            # Compress timing slightly for burst effect
            for e in section:
                if not e.is_rest:
                    e_copy = M.CharEvent(
                        char=e.char,
                        timestamp=e.timestamp * 0.95,  # 5% faster
                        word_text=e.word_text,
                        char_idx=e.char_idx,
                        beat_position=e.beat_position,
                        section=e.section,
                        is_rest=e.is_rest
                    )
                    enhanced.append(e_copy)
                else:
                    enhanced.append(e)
        else:
            enhanced.extend(section)
    
    return enhanced


# ==================== UTILITY FUNCTIONS ====================

def snap_to_grid(beats: float, grid: float = 0.5) -> float:
    """Snap beats to nearest musical grid interval"""
    return round(beats / grid) * grid


def group_events_by_section(events: list[M.CharEvent]) -> list[list[M.CharEvent]]:
    """Groups CharEvents by section"""
    if not events:
        return []
    
    sections: list[list[M.CharEvent]] = []
    current = [events[0]]
    
    for e in events[1:]:
        if e.section != current[-1].section:
            sections.append(current)
            current = []
        current.append(e)
    
    if current:
        sections.append(current)
    
    return sections


# ==================== MAIN GENERATION ====================

def generate_beatmap(word_list: list[str], song: M.Song) -> list[M.CharEvent]:
    """
    Generate an engaging, playable beatmap using slot-based rhythm generation.
    
    This modern approach:
    1. Analyzes audio to find musical moments (slots)
    2. Filters for playability (no impossible speeds)
    3. Assigns words to measures intelligently
    4. Creates natural rhythm patterns
    """
    beat_duration = 60 / song.bpm
    
    sb_info = get_sb_info(song, subdivisions=4)  # 16th notes
    
    # build rhythm slots from audio
    slots = build_rhythm_slots(sb_info, song)
    
    slots = filter_slots_for_playability(slots, min_spacing=0.12)
    
    measures = group_slots_by_measure(slots, beat_duration)
    
    intensity_profile = None
    if song.file_path:
        path = C._to_abs_path(song.file_path)
        if path:
            intensity_profile = analyze_song_intensity(path, song.bpm)
    
    # prepare word bank
    word_bank = get_words_with_rhythm_info(word_list, beat_duration)
    
    events = assign_words_to_slots(measures, word_bank, intensity_profile)
    
    # for engagement
    events = add_rhythm_variations(events, song)
    
    return events