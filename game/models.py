from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

# --- beatmap generator

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

# --- engine 

class Song:
    def __init__(self, bpm, path, duration):
        self.bpm = bpm
        self.duration = duration
        self.file_path = path

class Level:
    def __init__(self, bg_path, cat_sprite_path, word_bank, song_path):
        self.bg_path = bg_path
        self.cat_sprite_path = cat_sprite_path
        self.word_bank = word_bank
        self.song_path = song_path