import random
from dataclasses import dataclass
from typing import Optional
from enum import Enum, auto
from analysis.audio_analysis import analyze_song_intensity, IntensityProfile
from . import constants as C

# NOT YET FULLY INTEGRATED YET

class RestType(Enum):
     PAUSE = auto()
     FILL = auto()

@dataclass
class Word:
    """Represents a word with rhythm timing info"""
    text: str
    rest_type: Optional[RestType]
    ideal_beats: Optional[float]
    snapped_beats: float
    snapped_cps: float

@dataclass
class CharEvent:
    char: str
    timestamp: float
    word_text: str
    char_idx: int
    beat_position: float
    hit: bool = False
    #section_idx: int

# NEED TO INCORPORATE CHAR EVENT INTO RHYTHM/BEATMAP.PY

def estimate_required_words(
    song_duration_secs: float,
    avg_word_len: float,
    target_cps: float
) -> int:
    """Estimates number of words needed to fill song_duration_secs"""
    total_chars = song_duration_secs * target_cps
    return int(total_chars / avg_word_len)

def expand_word_list(
    base_words: list[str],
    target_word_count: int,
    *,
    shuffle_each_cycle: bool = True
) -> list[str]:
    """Expands the base word list to a target word count. Can shuffle new words."""
    if not base_words:
        raise ValueError("No words provided")

    expanded = []
    while len(expanded) < target_word_count:
        cycle = base_words[:]
        if shuffle_each_cycle:
            random.shuffle(cycle)
        expanded.extend(cycle)

    return expanded[:target_word_count]

def snap_to_grid(beats: float, grid: float = 0.5) -> float:
    """Snap beats to nearest musical grid interval"""
    return round(beats / grid) * grid

def get_beat_duration(bpm: int) -> float:
     if bpm <= 0:
         raise ValueError("Invalid bpm")
     
     return 60 / bpm

def get_words_with_snapped_durations(words: list[str], beat_duration: float) -> list[Word]:
	"""Returns (word, ideal_beats, snapped_beats, snapped_cps)"""
	return [
		Word(
		    text=word,
            rest_type=None,
		    ideal_beats=(ideal_beats :=  len(word) / C.TARGET_CPS / beat_duration),
		    snapped_beats=(snapped := max(C.SNAP_GRID, snap_to_grid(ideal_beats, C.SNAP_GRID))),
		    snapped_cps=(len(word) / (snapped * beat_duration))
		    )
        for word in words
    ]

def get_raw_pressure(remaining_words: list[Word], remaining_sections: int, leftover_beats: float):
	"""Calculates pressure WITHOUT considering pauses (Step 1)"""
	if remaining_sections <= 0 or not remaining_words:
		return 0.0

	total_word_beats = sum(w.snapped_beats for w in remaining_words)
	total_avail_beats = (remaining_sections - 1) * C.BEATS_PER_SECTION + leftover_beats
	return total_word_beats / total_avail_beats

def get_ideal_pause_duration(
    remaining_words: list[Word], 
    remaining_sections: int, 
    leftover_beats: float
    ) -> float:
    """
    Calculate ideal pause duration to achieve pressure = 1.0 roughly.
    Returns pause duration clamped to [MIN_PAUSE, MAX_PAUSE] and snapped to 0.5 (eighth note) beat grid
    """
    if remaining_sections <= 0 or len(remaining_words) <= 1:
        return C.IDEAL_PAUSE
            
    total_avail_beats = (remaining_sections - 1) * C.BEATS_PER_SECTION + leftover_beats
    total_word_beats = sum(w.snapped_beats for w in remaining_words)
    space_for_pauses = total_avail_beats - total_word_beats # could be negative

    if space_for_pauses <= 0:
         raise ValueError("Insufficent space for pauses") # later, just return -1.0 and have the outside function check

    num_pauses = len(remaining_words) - 1    
    if num_pauses == 0:
        return C.IDEAL_PAUSE

    ideal_pause = space_for_pauses / num_pauses
    snapped_pause = snap_to_grid(ideal_pause, grid=0.5)
    final_pause = max(C.MIN_PAUSE, min(C.MAX_PAUSE, snapped_pause))

    return final_pause

def get_pressure_ratio(remaining_words : list[Word], remaining_sections : int, leftover_beats : float, pause_duration: float) -> float:
    """Calculates pressure of remaining song beats including pauses."""
    if remaining_sections <= 0 or not remaining_words:
        return 0.0

    total_avail_beats = (remaining_sections - 1) * C.BEATS_PER_SECTION + leftover_beats
    total_word_beats = sum(w.snapped_beats for w in remaining_words)

    num_pauses = max(0, len(remaining_words) - 1)
    total_pause_beats = pause_duration * num_pauses

    return (total_pause_beats + total_word_beats) / total_avail_beats

def get_section_remaining_beats(section_words : list[Word]) -> float:
	"""Returns remaining beats. section_words should include pauses."""
	total_word_duration = sum(word.snapped_beats for word in section_words)
	return C.BEATS_PER_SECTION - total_word_duration

def select_best_word(remaining_beats: float, words_bank: list[Word], remaining_words: list[Word], 
                     true_pressure : float, prepping_for_build_up: bool) -> Word:
    if prepping_for_build_up:
        remaining_beats = remaining_beats % C.BEATS_PER_MEASURE # beats left in measure

    candidates = [w for w in remaining_words if w.snapped_beats <= remaining_beats]
            
    if not candidates:
        if random.random() < 0.3 and words_bank:
            return random.choice(words_bank)

        fill = Word("", RestType.FILL, None, remaining_beats, 0.0)
        return fill 

    viable = [w for w in candidates if abs(w.snapped_cps - C.TARGET_CPS) <= C.CPS_TOLERANCE]
    
    if viable:
        if true_pressure > 1.2: # 20% over-capacity
            sorted_viable = sorted(viable, key=lambda w: w.snapped_beats)
            top_choices = sorted_viable[:min(3, len(sorted_viable))]
            return random.choice(top_choices)
        else:
            sorted_viable = sorted(viable, key=lambda w: w.snapped_beats, reverse=True)
            top_choices = sorted_viable[:min(3, len(sorted_viable))]
            return random.choice(top_choices)
    else:
        sorted_candidates = sorted(candidates, key=lambda w: abs(w.snapped_cps - C.TARGET_CPS))
        top_choices = sorted_candidates[:min(3, len(sorted_candidates))]
        return random.choice(top_choices)

def get_base_pause(
    section_idx: int,
    intensity_profile: Optional[IntensityProfile],
    ideal_pause: float,
    min_pause: float,
    max_pause: float
) -> float:
    """
    Returns a base pause duration (in beats) for this section based on audio intensity.
    If no profile, returns ideal_pause.
    """
    if intensity_profile is None:
        return ideal_pause

    secs = intensity_profile.section_intensities
    if not secs or section_idx >= len(secs):
        return ideal_pause

    section_val = secs[section_idx]
    avg = (sum(secs) / len(secs)) if len(secs) > 0 else section_val

    ratio = section_val / (avg + 1e-6)

    if ratio < 0.8: # quiet section
        return min(max_pause, ideal_pause + 0.5)

    if ratio > 1.2: # loud section
        return max(min_pause, ideal_pause - 0.25)

    return ideal_pause

def get_beat_offset(remaining_beats: float):
    """Returns an offset that rounds the current beat to the start of the measure
    if offset < PAUSE_ROUND_THRESHOLD, else returns 0."""
    current_section_beat = C.BEATS_PER_SECTION - remaining_beats
    beat_inside_measure = current_section_beat % C.BEATS_PER_MEASURE

    if beat_inside_measure != 0: # round note to nearest beat
        if (beat_inside_measure < C.PAUSE_ROUND_THRESHOLD):
            return -beat_inside_measure
            
        elif (C.BEATS_PER_MEASURE - beat_inside_measure < C.PAUSE_ROUND_THRESHOLD):
            return beat_inside_measure
    
    return 0

# if is loudest section,add build_up
def is_loudest_section(profile: IntensityProfile, section_idx: int) -> bool:
    """Returns true if the given section is the loudest section in the song."""
    if profile is None:
        return False
    
    secs = profile.section_intensities

    if not secs or not (0 <= section_idx < len(secs)):
        return False
    return secs[section_idx] == max(secs)

def get_current_measure(section_remaining_beats: float) -> int:
    """Calculates current measure from number of remaining beats in section"""
    beat_in_section = C.BEATS_PER_SECTION - section_remaining_beats
    return int(beat_in_section // C.BEATS_PER_MEASURE)

def add_build_up_word(
    section_words: list["Word"],
    word_bank: list["Word"],
    remaining_words: list["Word"],
) -> None:
    """
    Adds a 4-letter build-up word as a single Word with snapped_beats=4.
    Priority:
      1) remaining_words 4-letter
      2) word_bank 4-letter
      3) fallback BUILD_UP_WORDS
    """
    def pick_4_letter(pool: list[Word]) -> Optional[Word]:
        candidates = [w for w in pool if (w.rest_type is None and len(w.text) == 4)]
        return random.choice(candidates) if candidates else None

    chosen = pick_4_letter(remaining_words) or pick_4_letter(word_bank)

    if chosen is None:
        chosen = Word(
            text=random.choice(C.BUILD_UP_WORDS),
            rest_type=None,
            ideal_beats=4.0,
            snapped_beats=4.0,
            snapped_cps=0.0
        )

    chosen.snapped_beats = 4.0
    section_words.append(chosen)

    if chosen in remaining_words:
        remaining_words.remove(chosen)

def assign_words(word_list: list[str], num_sections: int, beat_duration: float, intensity_profile=None) -> list[list[Word]]:
    words_bank = get_words_with_snapped_durations(word_list, beat_duration)
    remaining_words = words_bank.copy()
    sections = [[] for _ in range(num_sections)]

    before_loudest = False
    
    word_usage_count = {w.text: 0 for w in words_bank}

    for section_idx in range(num_sections):

        # music-analysis base pause
        base_pause = get_base_pause(
            section_idx,
            intensity_profile,
            C.IDEAL_PAUSE,
            C.MIN_PAUSE,
            C.MAX_PAUSE
        )

        #check if next section is loudest
        if intensity_profile and section_idx + 1 < len(intensity_profile.section_intensities):
            if is_loudest_section(intensity_profile, section_idx + 1):
                before_loudest = True

        remaining_beats = C.BEATS_PER_SECTION
        prev_measure = -1
        do_buildup = False

        while remaining_beats > C.MIN_PAUSE and remaining_words:

            current_measure = get_current_measure(remaining_beats)
            in_last_measure = (current_measure == 3)

            if in_last_measure and prev_measure != current_measure:
                do_buildup = before_loudest or (random.random() < 0.5)

            prev_measure = current_measure

            # ---- BUILD-UP INSERTION
            if do_buildup and in_last_measure and remaining_beats >= 4:
                add_build_up_word(sections[section_idx], words_bank, remaining_words)
                remaining_beats -= 4
                continue

            # ---- PRESSURE + CPS LOGIC
            raw_pressure = get_raw_pressure(
                remaining_words,
                num_sections - section_idx,
                remaining_beats
            )

            ideal_pause = get_ideal_pause_duration(
                remaining_words,
                num_sections - section_idx,
                remaining_beats
            )

            if intensity_profile:
                ideal_pause = snap_to_grid(0.5 * ideal_pause + 0.5 * base_pause)

            true_pressure = get_pressure_ratio(
                remaining_words,
                num_sections - section_idx,
                remaining_beats,
                ideal_pause
            )

            best = select_best_word(
                remaining_beats,
                words_bank,
                remaining_words,
                true_pressure,
                in_last_measure and do_buildup
            )

            sections[section_idx].append(best)

            if best.rest_type is None and best in remaining_words:
                remaining_words.remove(best)
                if best.text:
                    word_usage_count[best.text] += 1

            if len(remaining_words) < len(words_bank) * 0.3: # less than 30% left
                # USE LEAST-USED WORDS
                for word in words_bank:
                    if word.text and word_usage_count[word.text] < 2:
                        if word not in remaining_words:
                            remaining_words.append(word)

            remaining_beats -= best.snapped_beats
            if remaining_beats <= 0:
                break

            # ---- PAUSE AFTER NORMAL WORDS
            pause = min(ideal_pause, remaining_beats)
            pause = snap_to_grid(pause, C.SNAP_GRID)

            sections[section_idx].append(make_pause(pause))

            remaining_beats -= pause

        if section_idx < num_sections - 1:
            section_pause = snap_to_grid(base_pause * 1.5, C.SNAP_GRID)
            sections[section_idx].append(make_pause(section_pause))

    return sections


def vary_pause_duration(sections_words: list[list[Word]]) -> list[list[Word]]:
    """
    Varies pause durations in sections_words based on neighboring word lengths.
    Alternates increments/decrements to maintain section balance.
    
    sections_words: List of sections, each containing Words and Pauses

    CONCERN -> needs to rebalance to prevent spilling into next section by cancelling out changes
    with "should-increment". What if the word_list has two long words in a row?? 
    """
    for section in sections_words:
        pauses_with_indices = [(i, w) for i, w in enumerate(section) if w.rest_type == RestType.PAUSE]
        
        if not pauses_with_indices:
            continue

        # possibly shuffle/randomly decide start increment direction??
        should_increment = True if random.random() < 0.5 else False
        
        for idx, (pause_idx, pause_word) in enumerate(pauses_with_indices):
            if idx % 2 == 0 and random.random() < 0.2:
                # 20% chance to skip
                continue

            next_word_idx = pause_idx + 1

            if next_word_idx >= len(section): 
                continue

            next_word = section[next_word_idx]

            if next_word.rest_type is not None:
                continue
            
            base_increment = get_pause_increment(len(next_word.text))
            
            adjustment = base_increment if should_increment else -base_increment
            
            if random.random() < 0.8:  # mostly alternate, but sometimes repeat the same
                should_increment = not should_increment
            
            pause_word.snapped_beats = snap_to_grid(
                max(C.MIN_PAUSE, min(C.MAX_PAUSE, pause_word.snapped_beats + adjustment)), grid=C.SNAP_GRID)
    return sections_words
                  
def get_pause_increment(word_len: int) -> float:
    """Returns a pause duration offset based on following word length."""
    if word_len >= 8: 
        return 0.5
    elif word_len >= 5:
        return 0.25
    else:
        return -0.25 
    
def balance_section_timing(section_words: list[list[Word]], target_beats: float = C.BEATS_PER_SECTION) -> None:
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
            word.snapped_beats = max(C.MIN_PAUSE, min(C.MAX_PAUSE, word.snapped_beats))
            word.snapped_beats = snap_to_grid(word.snapped_beats, grid=0.5)

def print_beatmap(events):
    for e in events:
        print(
            f"{e.char or '·'} "
            f"[beat {e.beat_position:5.2f}] "
            f"[{e.timestamp:5.2f}s]"
        )

def get_beatmap_duration_beats(beatmap: list[CharEvent], beat_duration: float) -> float:
    if len(beatmap) < 2:
        return 0.0
    return (beatmap[-1].timestamp - beatmap[0].timestamp) / beat_duration

def remove_beatmap_event(beatmap: list[CharEvent], word: str) -> list[CharEvent] | None:
    if not beatmap:
        return None

    return [e for e in beatmap if e.word_text != word]


def get_missing_pause_indices(sections_words: list[list[Word]]) -> list[tuple[int, int]]:
    """
    Returns (section_idx, word_idx) where a pause should be inserted
    BEFORE word_idx.
    """
    indices: list[tuple[int, int]] = []

    for section_idx, section in enumerate(sections_words):
        last_word = None
        for word_idx, word in enumerate(section):
            if last_word is not None:
                if last_word.rest_type is None and word.rest_type is None:
                    indices.append((section_idx, word_idx))
            last_word = word

    return indices
                
def make_pause(pause_beats: float) -> Word:
    return Word(
        text="",
        rest_type=RestType.PAUSE,
        ideal_beats=pause_beats,
        snapped_beats=pause_beats,
        snapped_cps=0.0
    )

def add_missing_pauses(sections_words: list[list[Word]], pause_beats: float
) -> list[list[Word]]:
    """Adds pauses of 'pause_beats' in areas with no gaps between words."""
    indices = get_missing_pause_indices(sections_words)

    for section_idx, word_idx in reversed(indices):
        pause = make_pause(pause_beats)
        sections_words[section_idx].insert(word_idx, pause)

    return sections_words

def align_beatmap_to_song_duration(
    sections_words: list[list[Word]],
    song_duration_secs: float,
    beat_duration: float
) -> None:
    """Aligns a beatmap to last for roughly the given song_duration by adjusting pauses"""
    target_beats = song_duration_secs / beat_duration

    current_beats = sum(
        w.snapped_beats
        for section in sections_words
        for w in section
    )

    beat_error = target_beats - current_beats

    if abs(beat_error) < 0.25:
        return  # i think close enough

    pauses = [
        w for section in sections_words for w in section
        if w.rest_type == RestType.PAUSE
    ]

    if not pauses:
        return

    delta = beat_error / len(pauses)

    for p in pauses:
        p.snapped_beats = snap_to_grid(
            max(C.MIN_PAUSE, min(C.MAX_PAUSE, p.snapped_beats + delta)),
            C.SNAP_GRID
        )


def create_char_events(section_words : list[list[Word]], beat_duration : float) -> list[CharEvent]:
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

                for i, char in enumerate(word.text):
                    char_events.append(
                        CharEvent(
                            char=char,
                            timestamp=curr_beat * beat_duration,
                            word_text=word.text,
                            char_idx=i,
                            beat_position=curr_beat,
                        )
                    )
                    curr_beat += char_beat_duration
            else:
                curr_beat += word.snapped_beats
    
    return char_events

def get_max_words(beat_duration: float, num_sections: int, avg_word_len: float) -> int:
    """Returns the max possible words to fill num_sections based off TARGET_CPS"""
    if num_sections <= 0:
        raise ValueError("num_sections must be positive")
    
    max_chars_per_section = C.BEATS_PER_SECTION * C.TARGET_CPS * beat_duration
    max_chars_total = max_chars_per_section * num_sections

    return int(max_chars_total / avg_word_len)

def generate_beatmap(word_list : list[str], bpm : int, song_duration : int, audio_path: Optional[str] = None):
    beat_duration = 60 / bpm
    num_sections = int(song_duration / (beat_duration * C.BEATS_PER_SECTION))

    intensity_profile: Optional[IntensityProfile] = None
    if audio_path:
        intensity_profile = analyze_song_intensity(audio_path, bpm)

    avg_word_len = sum(len(w) for w in word_list) / len(word_list)
    
    total_beats = num_sections * C.BEATS_PER_SECTION
    avg_word_beats = (avg_word_len / C.TARGET_CPS) / beat_duration
    beats_per_word_with_pause = avg_word_beats + C.IDEAL_PAUSE
    
    target_words = int(total_beats / beats_per_word_with_pause)
    
    # reduce by 10-20% to ensure space
    target_words = int(target_words * 0.9)
    target_words = max(target_words, 10) # 10 IS A FILLER UNTIL LATER
    
    expanded = expand_word_list(word_list, target_words, shuffle_each_cycle=True)
    sections_words : list[list[Word]] = assign_words(expanded, 
                                  num_sections, 
                                  beat_duration, 
                                  intensity_profile=intensity_profile)

    vary_pause_duration(sections_words)
    balance_section_timing(sections_words)
    add_missing_pauses(sections_words, C.MIN_PAUSE)

    #should add cps outlier check in here + other tuning stuff 
    # if snapped_cps < MIN_CPS or snapped_cps > MAX_CPS:
    # rebalance()
    return create_char_events(sections_words, beat_duration)

    


# when incrementing up/down, could ask "is_close_to_beat" (does this make the next word on a beat)
# with a degree (if distance <= DEGREE then increment.)

#right now I USE AUDIO ANALYSIS only to vary pause_duration and check for build-up before loudest section.
# I want to also do INTENSITY SLOPE which basically varies CPS based off of audio intensity

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


