from game.beatmap_generator import snap_to_grid, assign_words, get_beat_duration, create_char_events, generate_beatmap
import game.constants as C
import game.models as M

def test_snap_to_grid_basic():
    assert snap_to_grid(1.24) == 1.0
    assert snap_to_grid(1.26) == 1.5
    assert snap_to_grid(2.75) == 3.0

def test_get_beat_duration():
    assert get_beat_duration(120) == 0.5
    assert get_beat_duration(60) == 1
    #assert get_beat_duration(0) 

#def test_get_words_with_snapped_durations():
    #words = 
    #assert get_words_with_snapped_durations() == 

def test_sections_sum_to_16_beats():
    words = ["cat", "no", "ki", "rhythm", "engine"]
    bpm = 120
    beat_duration = 60 / bpm
    num_sections = 4

    sections = assign_words(words, num_sections, beat_duration)

    for section in sections:
        total = sum(w.snapped_beats for w in section)
        assert abs(total - C.BEATS_PER_SECTION) < 0.01

def test_char_events_monotonic():
    sections : list[list[M.Word]] = [] # migyht throw error because sections is usually list[list[word]]
    events = create_char_events(sections, beat_duration=0.5)

    timestamps = [e.timestamp for e in events]
    assert timestamps == sorted(timestamps)

# need to teste very function used in test_beatmap

# def test_basic_beatmap():
#     words = ["cat", "test", "me", "rhythm", "beat"]
#     song_path = "assets/audios/Toby Fox - THE WORLD REVOLVING"

#     beatmap = generate_beatmap(words, song_path)

#     for event in beatmap:
# #        print(
#             f"char={event.char!r}, "
#             f"beat={event.beat_position:.2f}, "
#             f"time={event.timestamp:.2f}s, "
#             f"word={event.word_text}"
#         )

#eventually do cps outlier check
# 
#test_basic_beatmap() # gets 5.25 second max durations. where pauses?
