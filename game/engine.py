import pygame
import sys
import time

from .rhythm import RhythmManager
from .input import Input

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

        self.font = pygame.font.Font(None, 48)
        
        self.message = None
        self.message_duration = 0.0

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

            self.handle_events()
            self.update(dt)

            pygame.display.flip()

        pygame.quit()
        sys.exit()

# --- HANDLE EVENTS

    def handle_events(self) -> None:
        pass
        # RENOVATE INTO HELPER FUNCTIONS

# --- UPDATE

    def update(self, dt: float) -> None:
        self.screen.fill((0, 0, 0))

        self.rhythm.update()

        if self.message is not None and self.message_duration >= 0.0:
            self.draw_text(self.message, False)
            self.message_duration -= dt # need to fix, magical num

        if self.rhythm.current_expected_char() is None:
            self.show_message("Congratulations!", 5)
            # self.draw_text("Congratulations!", False)
            self.running = False
            return

# experimental

        events = pygame.event.get()

        for event in events:
                if event.type == pygame.QUIT:
                    self.running = False

        self.input.update(events=events)

        # experiment ----
        typed = self.input.typed_chars

        current_char_idx = self.rhythm.current_char_index

        if current_char_idx != self.last_char_idx:
            if not self.used_current_char and self.last_char_idx != -1:
                self.misses += 1
                self.show_message("Missed!", 2)
                #self.draw_text("Missed!", False)
        
            self.used_current_char = False
            self.last_char_idx = current_char_idx
             
        # NEED TO ADD LIVES VAR. ON LIVE LOST, SAY "You Lost a Life.". ON NO LIVES, END GAME 
            #NEED TO MAKE IT TIME DRIVEN AND ORGANIZE ROLES BETWEEN ENGINE AND RHYTHMMANAGER
        #print(
        #    f"IDX={current_char_idx}",
        #    f"offset={time.perf_counter() - self.rhythm.char_start_time:.3f}",
        #    f"dur={self.rhythm.current_char_dur:.3f}",
        #    f"on_beat={self.rhythm.on_beat()}"
        #)

        print("CHAR: " + str(current_char_idx))
        if typed:
             for key in typed:
                if self.used_current_char:
                     continue
                
                expected = self.rhythm.current_expected_char()
                if expected is None:
                    break

                if key == expected and self.rhythm.on_beat():
                    self.show_message("Awesome!", 2)
                    #self.draw_text("Awesome!", False)
                    self.score += 1
                    self.last_char_idx = current_char_idx
                else:
                     self.show_message("Yikes!", 2)
                     self.misses += 1
                
                print("HIT:", key, "expected:", expected)
                self.used_current_char = True
        
        self.draw_curr_word(self.rhythm.current_expected_word())

    # --- TEXT

    def show_message(self, txt : str, secs : int):
        self.message = txt
        self.message_duration = secs

    def draw_text(self, txt : str, left : bool): # white text
        text_surface = self.font.render(txt, True, (255, 255, 255))
        if left:
            self.screen.blit(text_surface, (100,100))
        else:
            self.screen.blit(text_surface, (400, 100))

    #def draw_text(self, txt : str, color : tuple, pos : tuple): # white text
    #    text_surface = self.font.render(txt, True, color)
    #   self.screen.blit(text_surface, pos)
    
    def draw_miss_message(self): # white text
        self.draw_text("Missed.", (255, 0, 0), (100, 100))
    
    def draw_success_message(self): # white text
        self.draw_text("Awesome!", (0, 255, 35), (200, 100))
    
    #def draw_curr_word(self, txt : str):
    #    self.draw_text(txt, (255, 255, 255), (450, 400))

    def draw_curr_word(self, txt : str):
        self.draw_text(txt, True)


    