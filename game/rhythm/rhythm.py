import time
import random

# adds something that takes account of the total chars
# if too many, DO SOMETHING -> but for now just say invalid wordlist

class RhythmManager:
    GRACE = 0.08 

    def __init__(self, words : list[str], bpm : int, song_secs : int):
        self.words = words
        self.bpm = bpm
        self.song_secs = song_secs
        
        self.beat_dur = 60 / bpm
        self.start_time = time.perf_counter()

        self.current_word_index = 0
        self.current_char_index = 0

        self.current_char_dur = None
        self.char_start_time = None

        self._setup_word()
    
    # --- word set-up

    def _setup_word(self):
        if self.current_word_index >= len(self.words):
            self.current_char_dur = None
            self.char_start_time = None
            return
        
        word = self.words[self.current_word_index]
        self.num_chars = len(word)

        if self.num_chars == 0:
            self.current_char_dur = None
            self.char_start_time = None
            self._advance_word()
            return

        multiplier = get_multiplier(self.num_chars)
        # MAKE A FUNCTION get_multiplier(self.num_chars) USED HERE that manages 'multiplier' 
        # based off of how many chars there are 

        word_duration = self.beat_dur * multiplier

        self.current_char_dur = word_duration / self.num_chars
        self.current_char_index = 0
        self.char_start_time = time.perf_counter()
        
    def _advance_word(self):
        self.current_word_index += 1
        self._setup_word()

    # --- update 

    def update(self):
        if self.current_char_dur is None:
            return
        
        now = time.perf_counter()
        elapsed = now - self.char_start_time

        if elapsed >= self.current_char_dur:
            self.current_char_index += 1
            self.char_start_time = now
        
        if self.current_char_index >= self.num_chars:
            self._advance_word()
        
    def on_beat(self) -> bool:
        if self.char_start_time is None:
            return False

        now = time.perf_counter()
        offset = now - self.char_start_time

        distance = min(offset, self.current_char_dur - offset)

        return distance <= self.GRACE
    
    def current_expected_char(self) -> str | None:
        if self.current_word_index >= len(self.words):
            return None

        word = self.words[self.current_word_index]

        if self.current_char_index >= len(word):
            return None
        
        return word[self.current_char_index]
    
    def current_expected_word(self) -> str | None:
        if self.current_word_index >= len(self.words):
            return None
        
        return self.words[self.current_word_index]

def get_multiplier(num_chars: int) -> int: # why is this indented outside of rhythmmanager
    if num_chars <= 4:
        return 1
    elif num_chars <= 8:
        return 2
    else:
        return 3

# figure out char_length of word
# determine beats for that beat (beat_duration split by char_length)
# # flash beat for those beats

##### LOOK TO ADD
#while elapsed >= self.current_char_dur:
#    self.current_char_index += 1
#    self.char_start_time += self.current_char_dur