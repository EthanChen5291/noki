import random
from dataclasses import dataclass
from typing import Optional
from enum import Enum, auto
from analysis.audio_analysis import analyze_song_intensity, IntensityProfile

# NOT YET FULLY INTEGRATED YET

class RestType(Enum):
     PAUSE = auto()
     FILL = auto()

@dataclass
class Word:
    """Represents a word with rhythm timing info"""
    text: str
    rest_type: RestType
    ideal_beats: float
    snapped_beats: float
    snapped_cps: float

@dataclass
class CharEvent:
    char: str
    timestamp: float
    word_text: str
    beat_position: float
    #section_idx: int

# NEED TO INCORPORATE CHAR EVENT INTO RHYTHM/BEATMAP.PY

TARGET_CPS = 3.5
MIN_CPS = 3
MAX_CPS = 4.5 # need to test cps values
CPS_TOLERANCE = 0.5

BEATS_PER_MEASURE = 4
BEATS_PER_SECTION = 16
#USABLE_SECTION_BEATS = BEATS_PER_SECTION - BEATS_PER_MEASURE 
MIN_PAUSE = 0.5 #pause between each word
IDEAL_PAUSE = 1.0
MAX_PAUSE = 1.5
# Intervals of 0.5 here to stick to eight notes (because 0.5 of a beat is a half beat) for playability 
# since triplets (0.33) are weird and sixteenth notes (0.25) are prob too fast
SECTION_ANCHOR_COST = 1.0 # idk if I need this

PAUSE_ROUND_THRESHOLD = 0.5
MIN_BEAT_GAP = 0.25  # gap between each beat. want to incorporate later as a guard/check
SNAP_GRID = 0.5

BUILD_UP_WORDS = ["rush", "hope", "more", "next"]

def snap_to_grid(beats: float, grid: float = 0.5) -> float:
    """Snap beats to nearest musical grid interval"""
    return round(beats / grid) * grid

def get_beat_duration(bpm: int) -> float:
     return 60 / bpm

def get_words_with_snapped_durations(words: list[str], beat_duration: float) -> list[Word]:
	"""Returns (word, ideal_beats, snapped_beats, snapped_cps)"""
	return [
		Word(
		    text=word,
            rest_type=None,
		    ideal_beats=(ideal_beats :=  len(word) / TARGET_CPS / beat_duration),
		    snapped_beats=(snapped := snap_to_grid(ideal_beats, grid=0.5)),
		    snapped_cps=(len(word) / (snapped * beat_duration))
		    )
        for word in words
    ]

def get_raw_pressure(remaining_words: list[Word], remaining_sections: int, leftover_beats: float):
	"""Calculates pressure WITHOUT considering pauses (Step 1)"""
	if remaining_sections <= 0 or not remaining_words:
		return 0.0

	total_word_beats = sum(w.snapped_beats for w in remaining_words)
	total_avail_beats = (remaining_sections * BEATS_PER_SECTION) + leftover_beats
	return total_word_beats / total_avail_beats

def get_ideal_pause_duration(
    remaining_words: list[Word], 
    remaining_sections: int, 
    section_pressure: float
    ) -> float:
    """
    Calculate ideal pause duration to achieve pressure = 1.0 roughly.
    Returns pause duration clamped to [MIN_PAUSE, MAX_PAUSE] and snapped to 0.5 (eighth note) beat grid
    """
    if remaining_sections <= 0 or len(remaining_words) <= 1:
        return IDEAL_PAUSE
            
    total_avail_beats = remaining_sections * BEATS_PER_SECTION    
    total_word_beats = sum(w.snapped_beats for w in remaining_words)
    space_for_pauses = total_avail_beats - total_word_beats # could be negative

    if space_for_pauses <= 0:
         raise Exception("no pause time") # later, just return -1.0 and have the outside function check

    num_pauses = len(remaining_words) - 1    
    if num_pauses == 0:
        return IDEAL_PAUSE

    ideal_pause = space_for_pauses / num_pauses
    snapped_pause = snap_to_grid(ideal_pause, grid=0.5)
    final_pause = max(MIN_PAUSE, min(MAX_PAUSE, snapped_pause))

    return final_pause

def get_pressure_ratio(remaining_words : list[Word], remaining_sections : int, leftover_beats : float, pause_duration: float) -> float:
    """Calculates pressure of remaining song beats including pauses."""
    if remaining_sections <= 0 or not remaining_words:
        return 0.0

    total_avail_beats = leftover_beats + (remaining_sections * BEATS_PER_SECTION)
    total_word_beats = sum(w.snapped_beats for w in remaining_words)

    num_pauses = max(0, len(remaining_words) - 1)
    total_pause_beats = pause_duration * num_pauses

    return (total_pause_beats + total_word_beats) / total_avail_beats

def get_section_remaining_beats(section_words : list[Word]) -> float:
	"""Returns remaining beats. section_words should include pauses."""
	total_word_duration = sum(word.snapped_beats for word in section_words)
	return BEATS_PER_SECTION - total_word_duration

def select_best_word(remaining_beats: float, words_bank: list[Word], remaining_words: list[Word], 
                     true_pressure : float, prepping_for_buildup: bool) -> Word:
    if prepping_for_buildup:
        remaining_beats = remaining_beats % BEATS_PER_MEASURE # beats left in measure

    candidates = [w for w in remaining_words if w.snapped_beats <= remaining_beats]
            
    if not candidates:
        if random.random() < 0.3 and words_bank:
            return random.choice(words_bank)

        fill = Word("", RestType.FILL, None, remaining_beats, 0.0)
        return fill 

    viable = [w for w in candidates if abs(w.snapped_cps - TARGET_CPS) <= CPS_TOLERANCE]
    
    if viable:
        if true_pressure > 1.2: # 20% over-capacity
            return min(viable, key=lambda w: w.snapped_beats)
        else:
            return max(viable, key=lambda w: w.snapped_beats)
    else:
        return min(candidates, key = lambda w: abs(w.snapped_cps - TARGET_CPS)) 

def get_base_pause(intensity_profile: IntensityProfile, section_intensities: list[float], section_idx: int) -> float:
    """Returns a base pause duration depending on how loud the given section is"""
    section_intensity = section_intensities[section_idx]
    
    if section_intensity is not None:
            avg_intensity = sum(section_intensities) / len(section_intensities) if intensity_profile else section_intensity
            intensity_ratio = section_intensity / (avg_intensity + 1e-6)
            if intensity_ratio < 0.8:
                base_ideal_pause = min(MAX_PAUSE, IDEAL_PAUSE + 0.5)
            elif intensity_ratio > 1.2:
                base_ideal_pause = max(MIN_PAUSE, IDEAL_PAUSE - 0.25)

def get_beat_offset(remaining_beats: float):
    """Returns an offset that rounds the current beat to the start of the measure
    if offset < PAUSE_ROUND_THRESHOLD, else returns 0."""
    current_section_beat = BEATS_PER_SECTION - remaining_beats
    beat_inside_measure = current_section_beat % BEATS_PER_MEASURE

    if beat_inside_measure != 0: # round note to nearest beat
        if (beat_inside_measure < PAUSE_ROUND_THRESHOLD):
            return -beat_inside_measure
            
        elif (BEATS_PER_MEASURE - beat_inside_measure < PAUSE_ROUND_THRESHOLD):
            return beat_inside_measure
    
    return 0

# if is loudest section,add build_up
def is_loudest_section(intensity_profile: IntensityProfile, section_idx: int) -> bool:
    """Returns true if the given section is the loudest section in the song."""
    section_intensities = intensity_profile.section_intensities

    if section_intensities(section_idx) == max(section_intensities):
        return True
    
    return False

def get_current_measure(section_remaining_beats: float) -> int:
    """Calculates current measure from number of remaining beats in section"""
    beat_in_section = BEATS_PER_SECTION - section_remaining_beats
    return int(beat_in_section // BEATS_PER_MEASURE)

def add_build_up(section_words: list[Word], word_bank: list[Word], remaining_words: list[Word]) -> list[Word]:
    """Adds a 4-letter word from remaining_words or word_bank (if none in remaining_words) 
    to section_words set at one letter per beat. If none found, pulls from BUILD_UP_WORDS"""
    remaining_four_letter_words = (w for w in remaining_words if len(w.word_text == 4))
    
    if remaining_four_letter_words:
        word = random.choice(remaining_four_letter_words)
    else: 
        four_letter_words = (w for w in word_bank if len(w.word_text == 4))
        if four_letter_words:
            word = random.choice(four_letter_words)
    
    word = random.choice(BUILD_UP_WORDS)
    
    word.snapped_beats = 4
    section_words.append(word)

def assign_words(word_list: list[str], num_sections: int, beat_duration : float, 
                 intensity_profile: Optional[IntensityProfile] = None) -> list[list[Word]]:
    words_bank = get_words_with_snapped_durations(word_list, beat_duration)
    remaining_words = words_bank.copy()
    sections_words : list[list[Word]] = [[] for _ in range(num_sections)]
    
    beat_intensities = intensity_profile.beat_intensities # DO A CHECK LATER THAT ROUNDS NEXT BEAT IF NEXT BEAT IS BIG BEAT
    section_intensities = intensity_profile.section_intensities
    prep_for_buildup = False #ensures words assigned don't cross into the next measure (leaving all of the last measure for buildup)
    before_loudest_section = False
    previous_measure = -1
    # I wanna have a 50% chance at buildup in the last measure of every section 
    # and 100% chance if it's the last measure before the loudest section
    for section_idx in range(num_sections):
        section_intensity = section_intensities[section_idx] if intensity_profile else None
        base_pause = get_base_pause(section_intensity)
            
        remaining_beats = get_section_remaining_beats(sections_words[section_idx])
        
        if intensity_profile:
            if is_loudest_section(intensity_profile, section_idx+1):
                before_loudest_section = True

        while remaining_beats >= base_pause:
            # recalculate pressure and ideal_pause for every section
            current_measure = get_current_measure(remaining_beats)
            
            if current_measure == 3:
                random_chance = random.randint(1, 2) == 1 #50% chance
                if random_chance or before_loudest_section or (current_measure == previous_measure): #idk if better name for previous measure
                    prep_for_buildup = True
            else:
                prep_for_buildup = False
                previous_measure = current_measure

            raw_pressure = get_raw_pressure(remaining_words, num_sections - section_idx, remaining_beats)
            ideal_pause = get_ideal_pause_duration(remaining_words, num_sections - section_idx, raw_pressure)
            
            if section_intensity is not None:
                ideal_pause = (ideal_pause * 0.5) + (base_pause * 0.5) #mixing. idk if ratios are ok
            
            true_pressure = get_pressure_ratio(remaining_words, num_sections - section_idx, remaining_beats, ideal_pause)
            
            best = select_best_word(remaining_beats, words_bank, remaining_words, true_pressure, prep_for_buildup)
            sections_words[section_idx].append(best)
            
            if best.rest_type is None and best in remaining_words:
                remaining_words.remove(best)

            if (remaining_beats - best.snapped_beats) > 0:
                offset = get_beat_offset(remaining_beats)
                best.snapped_beats += offset
                remaining_beats -= best.snapped_beats 
                
            if (remaining_beats - ideal_pause) > 0:
                offset = get_beat_offset(remaining_beats)
                new_pause = ideal_pause + offset
                remaining_beats -= new_pause

            rest = Word("", RestType.PAUSE, None, new_pause, 0)
            sections_words[section_idx].append(rest)
    
    return sections_words

def vary_pause_duration(sections_words: list[list[Word]]) -> list[list[Word]]:
    """
    Varies pause durations in sections_words based on neighboring word lengths.
    Alternates increments/decrements to maintain section balance.
    
    sections_words: List of sections, each containing Words and Pauses

    CONCERN -> needs to rebalance to prevent spilling into next section by cancelling out changes
    with "should-increment". What if the word_list has two long words in a row?? 
    """
    for section in sections_words:
        pauses_with_indices = [
            (i, word) for i, word in enumerate(section) 
            if word.rest_type == RestType.PAUSE
        ]
        
        if not pauses_with_indices:
            continue
        
        num_pauses = len(pauses_with_indices)
        
        if num_pauses % 2 == 1:
            num_to_change = num_pauses - 1
        else:
            num_to_change = num_pauses
        
        should_increment = True
        
        for change_idx in range(num_to_change):
            pause_idx, pause_word = pauses_with_indices[change_idx]
            
            next_word_idx = pause_idx + 1
            if next_word_idx >= len(section):
                continue  # no word after pause
            
            next_word = section[next_word_idx]
            
            if next_word.rest_type is not None:
                continue
            
            base_increment = get_pause_increment(len(next_word.text))
            
            if should_increment:
                adjustment = base_increment
            else:
                adjustment = -base_increment
            
            pause_word.snapped_beats += adjustment
            
            pause_word.snapped_beats = max(MIN_PAUSE, min(MAX_PAUSE, pause_word.snapped_beats))
            
            pause_word.snapped_beats = snap_to_grid(pause_word.snapped_beats, grid=SNAP_GRID)
            
            should_increment = not should_increment
    
    return sections_words
                  
def get_pause_increment(word_len: int) -> float:
    """Returns a pause duration offset based on following word length."""
    if word_len >= 8: 
        return 0.5
    elif word_len >= 5:
        return 0.25
    else:
        return -0.25 
    
def balance_section_timing(section_words: list[list[Word]], target_beats: float = BEATS_PER_SECTION) -> None:
    """Adjust pauses so section ends exactly on target_beats."""
    for section in section_words:
        current_total = sum(w.snapped_beats for w in section)
        
        beat_error = target_beats - current_total
        
        if abs(beat_error) < 0.1:
            continue #return or continue
        
        # distribute error
        pauses = [w for w in section if w.rest_type == RestType.PAUSE]
        if not pauses:
            continue #return or continue
        
        adjustment_per_pause = beat_error / len(pauses)
        
        for word in pauses:
            word.snapped_beats += adjustment_per_pause
            word.snapped_beats = max(MIN_PAUSE, min(MAX_PAUSE, word.snapped_beats))
            word.snapped_beats = snap_to_grid(word.snapped_beats, grid=0.5)

def create_char_events(section_words : list[Word], beat_duration : float) -> list[CharEvent]:
    #char: str
    #timestamp: float
    #word_text: str
    #beat_position: float
    char_events: list[CharEvent] = []
    curr_beat = 0.0

    for section_idx, section in enumerate(section_words):
        for word in section:
            if word.text != "":
                char_beat_duration = word.snapped_beats / len(word.text)

                for char in word.text:
                    char_events.append(
                        CharEvent(
                            char=char,
                            timestamp=curr_beat * beat_duration,
                            word_text=word.text,
                            beat_position=curr_beat,
                        )
                    )
                    curr_beat += char_beat_duration
            else:
                curr_beat += word.snapped_beats
    
    return char_events

def generate_beatmap(word_list : list[str], bpm : int, song_duration : int, audio_path: Optional[str] = None):
    beat_duration = 60 / bpm
    num_sections = int(song_duration / (beat_duration * BEATS_PER_SECTION))
    intensity_profile = None

    if audio_path:
        beat_intensities, section_intensities = analyze_song_intensity(audio_path, bpm)
        section_intensity_profile = section_intensities
        beat_intensity_profile = beat_intensities

    sections_words = assign_words(word_list, num_sections, beat_duration, intensity_profile=intensity_profile)

    vary_pause_duration(sections_words) # change to produce None

    balance_section_timing(sections_words)

    #should add cps outlier check in here + other tuning stuff 
    # if snapped_cps < MIN_CPS or snapped_cps > MAX_CPS:
    # rebalance()
    return create_char_events(sections_words, beat_duration)

def print_beatmap(events):
    for e in events:
        print(
            f"{e.char or '·'} "
            f"[beat {e.beat_position:5.2f}] "
            f"[{e.timestamp:5.2f}s]"
        )

# when incrementing up/down, could ask "is_close_to_beat" (does this make the next word on a beat)
# with a degree (if distance <= DEGREE then increment.)


# ------- TO-DO LIST -------

# ---- functionality

# vary pauses
# beatmap timeline creation

# integration into rhythm.py

# positional bias
# rebalancing checks
# local cps outlier checks


# ---- quality of life

# chance to make the 4th measure one word that at 1 char / beat for build-up
# triplet events


