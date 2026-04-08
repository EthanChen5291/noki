import pygame
from game.engine import Game
from game.models import Level
from game.menu import MenuManager, _load_scores, _save_scores
from game.music import MusicManager

# whenever you add songs in custom, it disappears from custom and goes into canon menu after playing

# dual mode section -> no notes during first 2-3 measures for some reason (should only be grace of 1 measure)
# can't scroll down in canon level menu

# canon and custom tabs should be 120% larger and centered (center of the tab between the vertical line on the left and the right side of screen).
# they should be chrome tab style where they are curvy rectangles on the top but cut off by a line on the bottom (add a horizontal line under it)

# hold notes shouldn't disappear the moment the user presses -> only the parts of the note that have passed the hitmarker 
# and been pressed should disappear -> all others should still be there until they've been pressed 
# (or, if they past the hitmarker and arne't pressed they turn red)
# moreover some slight particles should be released around the note area whenever they're being pressed as an effect (I want a particle effect (like random particles flying from the point of impact (particles can fly anywhere from the point of impact and fading away on the hitmaker area if the user presses the key and hits the long note).)
# in dual/bounce mode the hold notes should be reversed. if the hold note is coming from the left, right now its sprite is still arriving end of hold note first

# moreover long hold notes during normal sections are fine especially if it's genuinely a very strong hold note (from audio analysis) 
# but need to ensure that it doesn't overlap with other notes -> if it does, remove the other notes (and adjust the word as appropriate)


# whenever the session slows down at the end of the level, make sure the zoom stops zooming out after a bit (so it doesnt look like the timeline is being crunched into the hitmarker)



# -- underscore for words not showing up on some words
# -- ghost letters (letters progressing without any input/circles) in ping pong mode
# -- HARD mode causes letter skips 

# animated timeline 
# animated circles + spin explode animation when pressed
# red circle animation for wrong

# cat hurt animation (red outline)
# upload file should not open separate menu -> should just grab file, and when file is grabbed, user can rename on the level screen


# HOLD NOTES -> allow some notes to be held. should be snapped to music grid, 
# not interfere with the next note, has same limitations for spacing as the other ones. user needs to hold the entire duration (with a small grace lenient at the end for when they release)
# can be at any letter of a sentence for any duration as long as it doesn't interfere, and should notably be included wherever the audio amplitudes stay basically the same (like if a singer is holding a note then we'd do hold note. don't force it though, only do it upon detection of these long notes but can also add shorter hold notes depending on how long the note is held)

# DOUBLE TIMELINE mode timeline splits into two. has 1 second animation of the timeline splitting into two vertically (where one goes up and one goes down) so their vertical center was where the (default mode timeline was). there should be no notes for one measure during entrance just like dual mode
# - first timeline supports left side of keyboard, second timeline right side
# - during double timeline mode, movement is distinct. like it's not just oh you can type normally but 
# have to look at two timelines:
# - VERS 1: double timeline where one timeline is harmony (holding letters in 1-2 letter words for long time 
# while other timeline (otherside of keyboard) types words with letters only on that side). this means that two separate word lists going on for both 
# should alternate between timeslines (like timeline 1 does holding then timeline 2 does holding with word list adjusted 
# as appropriate to only include letters on the "not holding" side for the words)
# - VERS 2: same idea as VERS 1 except no holding. full words show up on each timeline (so not split letters of the same word list)
# where each word only has letters corresponding to its respective side of the keyboard

# SPEAK MODE: 
# uses AI to say the word (girl voice) -> noki meow animation.
# # user has word_duration amount of time to type the entire word, no beats just type it in x time.
# SYMBOL: volume symbol on timeline
# this leads timeline to fade away and noki to lerp to center and camera to zoom in on noki (noki bop 2 animation).
# when it's time for word, ensure word is at least 3 letters long and noki does noki_meow animation. 
# WAIT REAL ANIMATION: they speak, and when it's typed correctly, noki meows and gets rid of red spots crawling towards him (shockwavve)
# this repeats. speak mode has a cooldown AND SHOULD ONLY WORK WITH PARTS OF THE SONG (what parts?) and should last at least 3-4 measures (until the audio appears to "change section") 

# ADD 2 MORE CATS, each with their own hitmarker skin: 
# base cat (current cat) -> red skin. THREE LIVES, normal points
# female cat (white, slender, green eyes) -> (green hitmarker). TWO LIVES, 1.25x points multiplier
# chunk cat (nods head every 2 beats). FIVE LIVES, 0.75x points multipler, slows the song down 

# cat selection mode in main title screen with button under the cat
# incorporate the two cats into beginning animation

SONG_PATH = "assets/audios/"
SONG_NAMES = [
    "Scorpion.mp3",
    "Playful Massacre.mp3", 
    "Decisive Battle.mp3", 
    "Takedown - Huntrix.mp3", 
    "Glitch in your Heart.mp3", 
    "Disturbing the Peace.mp3", 
    "Catch Catch.mp3", 
    "Toby Fox - Finale.mp3",
    "ICARIUS.mp3",
    "Fluffing A Duck.mp3",
    "Malo Kart.mp3",
    "tidalwave.mp3",
    "hustle.mp3",
    "what is love?.mp3"
]

WORD_BANK_1 = ["cat", "test", "me", "rhythm", "beat", "fish", "moon", "derp", "noki", "yeah"]
WORD_BANK_2 = ["cat", "here", "me", "chosen", "beat", "hope", "soul", "true", "love", "stay"]


def main():
    pygame.init()
    info = pygame.display.Info()
    screen = pygame.display.set_mode((info.current_w, info.current_h), pygame.RESIZABLE)
    pygame.display.set_caption("Key Dash")
    clock = pygame.time.Clock()

    music = MusicManager()

    start_state = "title"
    while True:
        menu = MenuManager(screen, clock, SONG_NAMES, start_state=start_state, music=music)
        result = menu.run()

        if result is None:
            break

        selected, difficulty, word_bank = result

        level = Level(
            word_bank=word_bank,
            song_path=SONG_PATH + SONG_NAMES[selected],
            difficulty=difficulty,
        )

        # ==========================TO DO LIST

        # for the bounce mode mode -> reverse on arrows (bounces off of arrows and reverses) almost like going back in the timeline. the measure bars should also kinda accelerae backwards and reverse too. so not like a reverse, but almost like the hitmarker bounces off (so like momentum in the opposite direction with the measure and beat markers reflecting that)
        # don't move the hitmarker. 
        # make style
        # when skipping a word (like the word is gonna get cut off in the slot, add two "ghost words", blue circles (like sans's fake hits) where the player shouldn't press but complete the word)

        #save all the beatmaps manually of al the existing levels (so they don't have to load).
        # on pause screen, make the timeline and notes freeze not disappear
        # add restart button 

        #on simpler/calmer songs, don't just end words because the melody is slow/sparse, just drag it along for longer (ex. instead of givin "cat", "blob" "yeah" and cutting off half the words, just do "blob" and "cats"). if not many slots available.
        # for complex/faster/more pacey songs/sections, just add the blue stuff

        # on finish, add some finish thing for 2-3 seconds the open 'finish' menu
        
        #make it to where, on wrong note, the next char isn't immediately deactivated.

        # some speed-ups seem to happen 

        #timeline rotate
        #EVENTUALLY
        # work on file system, saving files to cloud (so the files are there for that user every time, etc)
        #integrate font
        # doesn't understand climax sections well

        # ===== BUGS
        # doesn't wait until last char is finished in dual section before transitioning back
        # sometimes has ghost letters (skips on its own) (only bounce mode). 
        # sometimes doesn't have grace period when starting dual mode

        # Ariana Grande - Problem ft. Iggy Azalea 
        # words synced to song? is word detection from audio possible?

        music.pause_for_game()
        game = Game(level=level, screen=screen, clock=clock)
        game.run()
        music.resume_from_game()
        start_state = "level_select"

        # Persist top score for this song + difficulty
        song_key = SONG_NAMES[selected]
        scores   = _load_scores()
        prev     = scores.get(song_key, {}).get(difficulty, 0)
        if game.score > prev:
            scores.setdefault(song_key, {})[difficulty] = game.score
            _save_scores(scores)

    pygame.quit()


if __name__ == "__main__":
    main()
