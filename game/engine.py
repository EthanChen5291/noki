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
    SCROLL_SPEED = 300
    
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
                self.draw_text("Missed", False)
                #self.draw_text("Missed!", False)
        
            self.used_current_char = False
            self.last_char_idx = current_char_idx
             
        # NEED TO ADD LIVES VAR. ON LIVE LOST, SAY "You Lost a Life.". ON NO LIVES, END GAME 
            #NEED TO MAKE IT TIME DRIVEN AND ORGANIZE ROLES BETWEEN ENGINE AND RHYTHMMANAGER
        print(
            f"IDX={current_char_idx}",
            f"offset={time.perf_counter() - self.rhythm.char_start_time:.3f}",
            f"dur={self.rhythm.current_char_dur:.3f}",
            f"on_beat={self.rhythm.on_beat()}"
        )

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
                     self.draw_text("Yikes", False)
                     self.misses += 1
                
                #print("HIT:", key, "expected:", expected)
                self.used_current_char = True
        
        self.draw_curr_word(self.rhythm.current_expected_word())

    # --- TEXT

    def render_timeline(self):
        current_time = time.perf_counter() - self.rhythm.start_time
        
        current_word = self.rhythm.current_expected_word()
        if current_word:
            word_surface = self.font.render(current_word, True, (255, 255, 255))
            word_rect = word_surface.get_rect(center=(960, 300))
            self.screen.blit(word_surface, word_rect)
        
        pygame.draw.line(
            self.screen, (255, 255, 255),
            (0, 540), (1920, 540),
            4
        )
        
        #center hit line (idk if imma keep, prob gonna make it a bar above the line)
        pygame.draw.line(
            self.screen, (255, 255, 0),  # Yellow
            (960, 490), (960, 590),
            6
        )
        
        for event in self.rhythm.beatmap:
            time_until_hit = event.timestamp - current_time
            
            if -0.5 < time_until_hit < 3.0:
                marker_x = 960 + (time_until_hit * SCROLL_SPEED)
                
                if event.is_rest:
                    # pause marker (GONNA CHANGE TO BLUE LINE SOON)
                    pygame.draw.line(
                        self.screen, (100, 100, 255),
                        (marker_x, 540), (marker_x, 600),
                        2
                    )
                elif event.beat_position % 4 == 0:
                    # measure line
                    pygame.draw.line(
                        self.screen, (255, 255, 255),
                        (marker_x, 490), (marker_x, 590),
                        4
                    )
                elif event.beat_position % 1 == 0:
                    # beat line
                    pygame.draw.line(
                        self.screen, (100, 100, 100),
                        (marker_x, 510), (marker_x, 570),
                        2
                    )
        
        current_char_idx = self.rhythm.current_char_index
        if current_word and current_char_idx < len(current_word):
            char_width = self.font.size(current_word[0])[0]
            arrow_x = word_rect.left + (current_char_idx * char_width)
            pygame.draw.polygon(
                self.screen, (255, 0, 0),
                [(arrow_x, 350), (arrow_x-10, 370), (arrow_x+10, 370)]
            )
    
    def draw_miss_message(self): # white text
        self.draw_text("Missed.", (255, 0, 0), (100, 100))
    
    def draw_success_message(self): # white text
        self.draw_text("Awesome!", (0, 255, 35), (200, 100))
    
    #def draw_curr_word(self, txt : str):
    #    self.draw_text(txt, (255, 255, 255), (450, 400))

    def draw_curr_word(self, txt : str):
        self.draw_text(txt, True)


    