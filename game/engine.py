import pygame
import sys
import time

from .rhythm import RhythmManager
from .input import Input
from .beatmap_generator import generate_beatmap

pygame.init()

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
        self.screen = pygame.display.set_mode((1920, 1080))
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
        
        beatmap = generate_beatmap(
            word_list=level.word_bank,
            bpm=level.song.bpm,
            song_duration=level.song.duration
        )
        self.rhythm = RhythmManager(beatmap)
        
        self.input = Input()

    def run(self) -> None:
        self.running = True
        while self.running:
            dt = self.clock.tick(60) / 1000
            self.update(dt)
            pygame.display.flip()

        pygame.quit()
        sys.exit()

    def update(self, dt: float) -> None:
        self.screen.fill((0, 0, 0))
        
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                self.running = False
        
        self.input.update(events=events)
        
        self.rhythm.update()
        
        if self.rhythm.is_finished():
            self.show_message("Congratulations!", 5)
            self.running = False
            return
        
        current_char_idx = self.rhythm.char_event_idx
        
        if current_char_idx != self.last_char_idx:
            if not self.used_current_char and self.last_char_idx != -1:
                self.misses += 1
                self.show_message("Missed!", 1)
            
            self.used_current_char = False
            self.last_char_idx = current_char_idx
        
        if self.input.typed_chars:
            for key in self.input.typed_chars:
                if self.used_current_char:
                    continue
                
                expected = self.rhythm.current_expected_char()
                if expected is None:
                    break

                if key == expected and self.rhythm.on_beat():
                    self.show_message("Perfect!", 1)
                    self.score += 1
                    self.used_current_char = True
                else:
                    self.show_message("Miss!", 1)
                    self.misses += 1
                    self.used_current_char = True
        
        self.render_timeline()
        
        if self.message and self.message_duration > 0:
            self.draw_text(self.message, False)
            self.message_duration -= dt

    def render_timeline(self):
        current_time = time.perf_counter() - self.rhythm.start_time
        
        current_word = self.rhythm.current_expected_word()
        if current_word:
            word_surface = self.font.render(current_word, True, (255, 255, 255))
            word_rect = word_surface.get_rect(center=(960, 300))
            self.screen.blit(word_surface, word_rect)
        
        # --- draw timeline

        pygame.draw.line(self.screen, (255, 255, 255), (0, 540), (1920, 540), 4)
        pygame.draw.line(self.screen, (255, 255, 0), (780, 490), (780, 590), 20)
        
        for event in self.rhythm.beat_map:
            time_until_hit = event.timestamp - current_time
            
            if -0.5 < time_until_hit < 3.0:
                marker_x = 960 + (time_until_hit * self.SCROLL_SPEED)
                
                if event.char == "": #rest
                    pygame.draw.line(self.screen, (155, 155, 255), (marker_x, 540), (marker_x, 600), 2)
                elif event.beat_position % 4 == 0:
                    pygame.draw.line(self.screen, (255, 255, 255), (marker_x, 490), (marker_x, 590), 4)
                elif event.beat_position % 1 == 0:
                    pygame.draw.line(self.screen, (100, 100, 100), (marker_x, 510), (marker_x, 570), 2)
    
    def show_message(self, txt: str, secs: float):
        self.message = txt
        self.message_duration = secs
    
    def draw_text(self, txt: str, left: bool):
        text_surface = self.font.render(txt, True, (255, 255, 255))
        if left:
            self.screen.blit(text_surface, (100, 100))
        else:
            self.screen.blit(text_surface, (400, 100))
    
    def draw_curr_word(self, txt: str):
        self.draw_text(txt, True)