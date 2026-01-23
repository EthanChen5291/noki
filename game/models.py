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
    def __init__(self, bpm, duration, file_path):
        self.bpm = bpm
        self.duration = duration
        self.file_path = file_path

class Level:
    def __init__(self, word_bank: list[str], song_path: str):
        self.word_bank = word_bank
        self.song_path = song_path