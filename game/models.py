from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

# --- audio analysis

class SubBeatIntensity(Enum):
    """Normalized beat-level intensity relative to each beat's respective measure intensity"""
    WEAK = auto()
    MEDIUM = auto()
    STRONG = auto()

@dataclass(frozen=True)
class SubBeatInfo():
    """Beat-level timestamps, raw intensities, and normalized intensity levels"""
    time: float
    raw_intensity: float
    level: SubBeatIntensity

@dataclass(frozen=True)
class IntensityProfile:
    """Beat-level and section-level intensity curves (higher = more activity)"""
    beat_intensities: list[float]
    section_intensities: list[float]

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
    section: int
    hit: bool = False
    #section_idx: int

# --- engine 

class Song:
    def __init__(self, bpm: float, duration: float, file_path: str):
        self.bpm = bpm
        self.duration = duration
        self.file_path = file_path

class Level:
    def __init__(self, word_bank: list[str], song_path: str, bpm: Optional[int] = None):
        self.word_bank = word_bank
        self.song_path = song_path
        self.bpm = bpm