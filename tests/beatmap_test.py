from game.beatmap_generator import snap_to_grid, assign_words, BEATS_PER_SECTION, create_char_events, generate_beatmap

def test_snap_to_grid_basic():
    assert snap_to_grid(1.24) == 1.0
    assert snap_to_grid(1.26) == 1.5
    assert snap_to_grid(2.75) == 3.0

def test_sections_sum_to_16_beats():
    words = ["cat", "no", "ki", "rhythm", "engine"]
    bpm = 120
    beat_duration = 60 / bpm
    num_sections = 4

    sections = assign_words(words, num_sections, beat_duration)

    for section in sections:
        total = sum(w.snapped_beats for w in section)
        assert abs(total - BEATS_PER_SECTION) < 0.01

def test_char_events_monotonic():
    events = create_char_events(
        [
            [
                # empty sect
            ]
        ],
        beat_duration=0.5
    )

    timestamps = [e.timestamp for e in events]
    assert timestamps == sorted(timestamps)

def test_basic_beatmap():
    words = ["cat", "no", "ki"]
    bpm = 120
    song_duration = 8.0 # seconds

    beatmap = generate_beatmap(words, bpm, song_duration)

    for event in beatmap:
        print(
            f"char={event.char!r}, "
            f"beat={event.beat_position:.2f}, "
            f"time={event.timestamp:.2f}s, "
            f"word={event.word_text}"
        )

test_basic_beatmap() # gets 5.25 second max durations. where pauses?
