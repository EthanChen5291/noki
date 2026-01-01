import time
import random

class RhythmManager(words : list[str], bpm : int):
    beat_dur = 60 / bpm
    RHYTHM_MULTIPLIERS = [list: 1, 2] # multiplies with sliced beat. Looking to expand. should I add 0?

    for word in words:
        beats = len(word) # a beat on each char
        cur_beat_duration = get_beat_duration(beats, beat_dur)
        total_duration = total_beats * cur_beat_duration
        # add some if statements to check how fast bpm is -> if past a certain checkpoint, 
        # allow some rhythm multipliers. Moreover, look into rhythm multiplier logic

        # wait until this beat has passed, then moved to next word
        # signal beat during this time
        if is_on_beat():
            beats -= 1
        
        if beats == 0:
            continue
            # move to next word
    
    def get_beat_duration(beats : int, net_beat_dur : int) -> int:
        """
        Returns a rhythm-matchable beat duration dependent on word length 'beats'. Adds variation by 
        multiplying it by a random number in 'RHYTHM_MULTIPLIERS'.
        """
        index = random.randint(1,len(RHYTHM_MULTIPLIERS) - 1)
        multiplier = RHYTHM_MULTIPLIERS.get(index) # adds variation to beats

        total_beat_dur = (beat_duration) * multiplier
        sliced_beat_dur = total_beat_dur / l # sliced beat duration
        total_beats_for_word = total_beat_dur / sliced_bear_dur 


    def is_on_beat(sliced_beat_dur, total_beats) -> Boolean:
        """
        Checks if it's on beat. ADD 80ms grace period
        """

        current_beat = time.perf_counter()
        next_beat = sliced_beat_dur
        timer = time.perf_counter()

        delta = timer - current_beat
        while !(delta == sliced_beat_dur): #edit this to accomodate grace period
            True
        else:
            time.sleep(0.001)
            False




# figure out char_length of word
# determine beats for that beat (beat_duration split by char_length)
# # flash beat for those beats