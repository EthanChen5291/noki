import pygame
import sys
import time

from game.rhythm import RhythmManager
from game.input import Input

pygame.init()

# PRIORITIES
# - match word list to accomodate song length
# - MAX CUSTOMIZABILITY -> allow users to upload their own songs -> 
# will have to decide how many chars (words per song

# MAX CUSTOMIZABILITY 2 -> allow users to upload their own words.

class Song:
     def __init__(self, bpm, duration):
          self.bpm = bpm
          self.duration = duration

class Level:
     def __init__(self, bg_path, cat_sprite_path, word_bank, song):
          self.bg_path = bg_path
          self.cat_sprite_path = cat_sprite_path
          self.word_bank = word_bank
          self.song = song
          
class Game:
    def __init__(self, level) -> None:
        self.screen = pygame.display.set_mode((1920, 1080)) # how to adjust based off screen size resolution
        self.clock = pygame.time.Clock()
        self.running = False
        self.last_char_idx = -1
        self.used_current_char = False
        self.score = 0
        self.misses = 0

        self.level = level
        self.song = level.song

        self.rhythm = RhythmManager(
             words=level.word_bank,
             bpm=level.song.bpm,
             song_secs=level.song.duration
             )

        self.input = Input()

    def run(self) -> None:
        self.running = True
        while self.running:
            dt = self.clock.tick(60) / 1000
            self.screen.fill((0, 0, 0))

            self.handle_events()
            self.update()

            pygame.display.flip()

        pygame.quit()
        sys.exit()

    def handle_events(self) -> None:
        for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

        # RENOVATE INTO HELPER FUNCTIONS
    def update(self) -> None:
        self.screen.fill((0, 0, 0))

        self.rhythm.update()

        if self.rhythm.current_expected_char() is None:
            print("Congratulations!")
            self.running = False
            return

        self.input.update()
        typed = self.input.typed_chars

        current_char_idx = self.rhythm.current_char_index

        if current_char_idx != self.last_char_idx:
            if not self.used_current_char and self.last_char_idx != -1:
                self.misses += 1
                print("Missed!")
        
            self.used_current_char = False
            self.last_char_idx = current_char_idx
             
        # NEED TO ADD LIVES VAR. ON LIVE LOST, SAY "You Lost a Life.". ON NO LIVES, END GAME 
            #NEED TO MAKE IT TIME DRIVEN AND ORGANIZE ROLES BETWEEN ENGINE AND RHYTHMMANAGER

        if typed:
             for key in typed:
                if self.used_current_char:
                     continue
                
                expected = self.rhythm.current_expected_char()
                if expected is None:
                    break

                if key == expected and self.rhythm.on_beat():
                    print("Awesome!")
                    self.score += 1
                    self.last_char_idx = current_char_idx
                else:
                     print("Yikes!")
                     self.misses += 1
                
                self.used_current_char = True
        
        print(self.rhythm.current_expected_word())