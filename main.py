import os
os.environ['SDL_MOUSE_FOCUS_CLICKTHROUGH'] = '1'

import pygame
import time as _time
from game.play import PlaySession
from game.models import Level
from game.menu import MenuManager
from game.menu_utils import _load_scores, _save_scores
from game.music import MusicManager


def _show_loading_screen(screen, clock, duration=0.35):
    """Brief petal-spinner transition shown when returning to the menu after a level."""
    import random as _random
    from game.ui_components import Petal
    sw, sh = screen.get_size()
    petals = [Petal(sw, sh, randomize_y=True) for _ in range(55)]
    # Override petal colors/alpha so they're clearly visible against the black background
    _bright_colors = [
        (255, 255, 255),
        (200, 220, 255),
        (255, 200, 240),
        (180, 240, 255),
        (255, 230, 200),
    ]
    for p in petals:
        p.alpha = _random.randint(90, 190)
        p.color = _random.choice(_bright_colors)
    start = _time.perf_counter()
    while True:
        elapsed = _time.perf_counter() - start
        if elapsed >= duration:
            break
        clock.tick(60)
        screen.fill((0, 0, 0))
        for p in petals:
            p.update()
            p.draw(screen)
        # Fade out only for last 0.15s (no fade-in so petals are visible immediately)
        if elapsed > duration - 0.15:
            fade_alpha = int(255 * (1.0 - (duration - elapsed) / 0.15))
            overlay = pygame.Surface((sw, sh), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, fade_alpha))
            screen.blit(overlay, (0, 0))
        pygame.display.flip()
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return

CANON_PATH = "assets/audios/canon/"
CUSTOM_PATH = "assets/audios/custom/"
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


def main():
    pygame.init()
    info = pygame.display.Info()
    screen = pygame.display.set_mode((info.current_w, info.current_h), pygame.RESIZABLE)
    pygame.display.set_caption("Key Dash")
    # macOS clamps the window under the menu bar after creation: let that settle so
    # every screen measures the real surface
    for _ in range(3):
        pygame.event.pump()
        pygame.time.wait(30)
    screen = pygame.display.get_surface() or screen
    clock = pygame.time.Clock()

    music = MusicManager()

    menu = MenuManager(screen, clock, SONG_NAMES, start_state="title", music=music)
    while True:
        result = menu.run()

        if result is None:
            break

        selected, difficulty, word_bank, bpm, mode = (list(result) + ["words"])[:5]

        song_name = SONG_NAMES[selected]
        _canon_path = CANON_PATH + song_name
        song_path = _canon_path if os.path.exists(_canon_path) else CUSTOM_PATH + song_name
        level = Level(
            word_bank=word_bank,
            song_path=song_path,
            difficulty=difficulty,
            bpm=bpm,
            mode=mode,
        )

        # ==========================TO DO LIST


        # make the mouse a circle (like osu?)

        # prevent freezing when uploading 
        # add loading screen when finishing level

        # hearts circle around the cat's neck -> when lose a life, one of the hearts flies to the hitmarker and becomes the hitmarker
        # 2 second cooldown -> hitmarker flashes during cooldown and fly to hitmarker animation plays. 
        # hitmarker is gray and skinny and then becomes the heart's color and becomes thicker and glows.

        # upon bounce, make the markers bend animation. same for speedups and slowdowns to a certain extent.

        # for repeat notes, make the word (like the word text itself) orange the first time and get marginally larger and swap colors between the colors of the impulse particles every time that word passes

        # remove signs 

        # add ghost notes?

        # make the purple more bluish (get hex)

        # add a hitsound whenever finishing a hold note

        # can add personality to songs by repeating word patterns for very difficult seuqneces

        # hold note -> noki meows to music for duration of hold note

        # don't repeat same letter if single letter

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

        # add some gradients in the background,
        # add PRESS note animation -> it fades into mist

        #make it to where, on wrong note, the next char isn't immediately deactivated.

        # some speed-ups seem to happen 

        # zoom in mode _> everything becomes bigger (including circles and timeline) -> what does this do? double timeline?

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

        # Inner replay loop — keeps replaying the same level until player exits
        while True:
            music.pause_for_game()
            game = PlaySession(level=level, screen=screen, clock=clock, music=music)
            game_result = game.run()
            music.resume_from_game()

            # Persist top score for this song + difficulty (normalized score; the
            # accuracy and grade of the best run ride along under "<difficulty>_stats")
            song_key = SONG_NAMES[selected]
            scores   = _load_scores()
            score_key = difficulty if mode == "words" else f"{difficulty}@{mode}"
            prev     = scores.get(song_key, {}).get(score_key, 0)
            if game.score > prev:
                scores.setdefault(song_key, {})[score_key] = game.score
                scores[song_key][score_key + "_stats"] = {
                    k: game.stats.get(k) for k in ("accuracy", "grade", "stars", "max_combo", "misses")
                }
                _save_scores(scores)

            if game_result != "replay":
                break

        _show_loading_screen(screen, clock)
        menu.reset_for_return("level_select")

    pygame.quit()


if __name__ == "__main__":
    main()
