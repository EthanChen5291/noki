import time
from .beatmap_generator import CharEvent

class RhythmManager:
    GRACE = 0.5  # secs
    
    def __init__(self, beat_map: list[CharEvent], bpm: int):
        self.beat_map = beat_map
        self.char_event_idx = 0
        self.start_time = time.perf_counter()
        self.last_word = None
        self.beat_duration = 60 / bpm
    
    def update(self):
        """Advance to next character if timestamp has passed"""
        if self.char_event_idx >= len(self.beat_map):
            return
        
        elapsed = time.perf_counter() - self.start_time
        current_event = self.beat_map[self.char_event_idx]
        
        if elapsed >= current_event.timestamp:
            self.char_event_idx += 1
    
    def on_beat(self) -> bool:
        """Check if current time is within grace period of current beat"""
        if self.char_event_idx >= len(self.beat_map):
            return False
        
        elapsed = time.perf_counter() - self.start_time
        current_event = self.beat_map[self.char_event_idx]
        
        time_diff = abs(elapsed - current_event.timestamp)
        return time_diff <= self.GRACE
    
    def current_expected_char(self) -> str | None:
        """Get the character the player should type currently"""
        if self.char_event_idx >= len(self.beat_map):
            return None
        
        event = self.beat_map[self.char_event_idx]
        
        if event.char == "":
            return None
        
        return event.char
    
    def current_expected_word(self) -> str | None:
        if self.char_event_idx >= len(self.beat_map):
            return None

        event = self.beat_map[self.char_event_idx]

        if event.char == event.word_text[0]:
            self.last_word = event.word_text

        return self.last_word

    
    def is_finished(self) -> bool:
        """Check if song is complete"""
        return self.char_event_idx >= len(self.beat_map)