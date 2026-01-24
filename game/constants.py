import os
from typing import Optional

# --- project root

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

def asset(*parts):
    return os.path.join(BASE_DIR, "assets", *parts)

def _to_abs_path(p: Optional[str]) -> Optional[str]:
    """Convert a project-relative path like 'assets/audios/x.wav' to an absolute path."""
    if not p:
        return p
    return p if os.path.isabs(p) else os.path.join(BASE_DIR, p)

# --- audio analysis

STRONG_INTENSITY_THRESHOLD = 70 # melodies / strong beats, louder than 70%
MEDIUM_INTENSITY_THRESHOLD = 40 # louder than 40% 

# --- beatmap generator

TARGET_CPS = 3.5
MIN_CPS = 3
MAX_CPS = 4.5 # need to test cps values
CPS_TOLERANCE = 0.5

BEATS_PER_MEASURE = 4
BEATS_PER_SECTION = 16
#USABLE_SECTION_BEATS = BEATS_PER_SECTION - BEATS_PER_MEASURE 
MIN_PAUSE = 0.5 #pause between each word
IDEAL_PAUSE = 1.5
MAX_PAUSE = 2.0
# Intervals of 0.5 here to stick to eight notes (because 0.5 of a beat is a half beat) for playability 
# since triplets (0.33) are weird and sixteenth notes (0.25) are prob too fast

PAUSE_ROUND_THRESHOLD = 0.5
MIN_BEAT_GAP = 0.25  # gap between each beat. want to incorporate later as a guard/check
MAX_BEATMAP_DIFF = 3 # beats
SNAP_GRID = 0.5

BUILD_UP_WORDS = ["rush", "hope", "more", "next"]

# --- rhythm manager

GRACE = 1  # secs

# --- engine

SCROLL_SPEED = 300
HIT_X = 730
MISSED_COLOR = (255, 0, 0)
COLOR = (255, 255, 255)
UNDERLINE_LEN = 40

HIT_MARKER_Y_OFFSET = -70
HIT_MARKER_X_OFFSET = -40
HIT_MARKER_LENGTH = 200
HIT_MARKER_WIDTH = 20