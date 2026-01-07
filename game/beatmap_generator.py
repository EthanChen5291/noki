import random
from dataclasses import dataclass
from typing import Optional
from enum import Enum, auto

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
    """
    Calculates pressure of remaining song beats including pauses.
    """
    if remaining_sections <= 0 or not remaining_words:
        return 0.0

    total_avail_beats = leftover_beats + (remaining_sections * BEATS_PER_SECTION)
    total_word_beats = sum(w.snapped_beats for w in remaining_words)

    num_pauses = max(0, len(remaining_words) - 1)
    total_pause_beats = pause_duration * num_pauses

    return (total_pause_beats + total_word_beats) / total_avail_beats

def get_section_remaining_beats(section_words : list[Word]) -> float:
	"""
    Returns remaining beats. section_words should include pauses.
    """
	total_word_duration = sum(word.snapped_beats for word in section_words)
	return BEATS_PER_SECTION - total_word_duration

def select_best_word(remaining_beats: float, words_bank: list[Word], remaining_words: list[Word], true_pressure : float) -> Word:
    candidates = [w for w in remaining_words if w.snapped_beats <= remaining_beats]
            
    if not candidates: # eventually wanna make it repeat words if no words here
        if remaining_beats >= BEATS_PER_MEASURE:
            valid = max((w for w in words_bank if w.snapped_beats <= remaining_beats),
                            default = None
                            )
            if valid is not None:
                return valid # may cause some issues since total_avail_beats is manipulated with redundant word here

        else:
            return Word("", RestType.Fill, None, remaining_beats, 0) # should I make this different from natural pause str message

    viable = [w for w in candidates if abs(w.snapped_cps - TARGET_CPS) <= CPS_TOLERANCE]
    if viable:
        if true_pressure > 1.2: # 20% over-capacity
            return min(viable, key=lambda w: w.snapped_beats)
        else:
            return max(viable, key=lambda w: w.snapped_beats)
    else:
        return min(candidates, key = lambda w: abs(w.snapped_cps - TARGET_CPS)) 

def assign_words(word_list: list[str], pause_beat_duration: float, num_sections: int, beat_duration : float) -> list[list[Word]]:
    words_bank = get_words_with_snapped_durations(word_list, beat_duration)
    remaining_words = words_bank.copy()
    sections_words : list[list[Word]] = [[] for _ in range(num_sections)]

    for section_idx in range(num_sections):
        section = sections_words[section_idx]

        remaining_beats = get_section_remaining_beats(sections_words[section_idx])
        
        while remaining_beats >= IDEAL_PAUSE:
            # recalculate pressure and ideal_pause for every section
            raw_pressure = get_raw_pressure(remaining_words, num_sections - section_idx, remaining_beats)
            ideal_pause = get_ideal_pause_duration(remaining_words, num_sections - section_idx, raw_pressure)
            true_pressure = get_pressure_ratio(remaining_words, num_sections - section_idx, remaining_beats, ideal_pause)

            best = select_best_word(remaining_beats, words_bank, remaining_words, true_pressure)

            sections_words[section_idx].append(best)
            
            if best.rest_type is None:
                remaining_words.remove(best)

            rest = Word("", RestType.PAUSE, None, ideal_pause, 0)
            sections_words[section_idx].append(rest)

            remaining_beats -= (best.snapped_beats + ideal_pause)
            current_pause_duration = ideal_pause
    
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
    """
    Returns a pause duration offset based on following word length.
    """
    if word_len >= 8: 
        return 0.5
    elif word_len >= 5:
        return 0.25
    else:
        return -0.25 
    
def balance_section_timing(section_words: list[list[Word]], target_beats: float = BEATS_PER_SECTION) -> None:
    """
    Adjust pauses so section ends exactly on target_beats.
    """
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

    for section_idx, section in enumerate(section_words):
        curr_beat = section_idx * BEATS_PER_SECTION # maybe later do curr_beat = global_beat_cursor?

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

def generate_beatmap(word_list : list[str], bpm : int, song_duration : int):
    beat_duration = 60 / bpm
    num_sections = int(song_duration / (beat_duration * BEATS_PER_SECTION))

    sections_words = assign_words(word_list, IDEAL_PAUSE, num_sections, beat_duration)

    sections_words = vary_pause_duration(sections_words) # change to produce None

    sections_words = balance_section_timing(sections_words)

    #should add cps outlier check in here + other tuning stuff 
    # if snapped_cps < MIN_CPS or snapped_cps > MAX_CPS:
    # rebalance()
    return create_char_events(sections_words, beat_duration)


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


