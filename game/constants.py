# --- beatmap generator

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