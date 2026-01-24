import pygame
import imageio
import numpy as np
import sys
import time
import os

from . import constants as C
from .rhythm import RhythmManager
from .input import Input
from .beatmap_generator import generate_beatmap, get_song_info

pygame.init()

class Game:
    def __init__(self, level) -> None:
        info = pygame.display.Info()
        screen_width, screen_height = info.current_w, info.current_h
        self.screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
        pygame.display.set_caption("Rhythm Typing Game")
        self.clock = pygame.time.Clock()
        self.running = False
        self.last_char_idx = -1
        self.used_current_char = False

        self.score = 0
        self.misses = 0
        self.font = pygame.font.Font(None, 48)
        
        self.message = None
        self.message_duration = 0.0

        # --- loading
        self.screen.fill((0, 0, 0))
        loading_font = pygame.font.Font(None, 72)
        loading_text = loading_font.render("Loading...", True, (255, 255, 255))
        loading_rect = loading_text.get_rect(center=(screen_width//2, screen_height//2))
        self.screen.blit(loading_text, loading_rect)
        pygame.display.flip()

        # --- load assets
        self.level = level
        abs_song_path = C._to_abs_path(level.song_path)
        if abs_song_path is None:
            raise ValueError(f"Invalid song path: {level.song_path}")
        self.song_path = abs_song_path

        assets_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'images')
        cat_frames_path = os.path.join(assets_path, "cat_frames")

        self.cat_frames = [
            pygame.image.load(os.path.join(cat_frames_path, f)).convert_alpha()
            for f in sorted(os.listdir(cat_frames_path))
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ]

        self.cat_frame_index = 0
        self.num_cat_frames = len(self.cat_frames)

        timeline_file = os.path.join(assets_path, 'noki_timeline.png')
        self.timeline_img = pygame.image.load(timeline_file).convert_alpha()
        self.timeline_img = pygame.transform.scale(self.timeline_img, (1920, 200))
        
        # --- update loading
        self.screen.fill((0, 0, 0))
        loading_text = loading_font.render("Generating beatmap...", True, (255, 255, 255))
        loading_rect = loading_text.get_rect(center=(screen_width//2, screen_height//2))
        self.screen.blit(loading_text, loading_rect)
        pygame.display.flip()

        self.song = get_song_info(self.song_path, expected_bpm=self.level.bpm, normalize=True)
        self.beat_duration = 60 / self.song.bpm

        beatmap = generate_beatmap(
            word_list=level.word_bank,
            song=self.song
        )
        self.rhythm = RhythmManager(beatmap, self.song.bpm)
        
        self.input = Input()

        # --- play music
        pygame.mixer.init()
        pygame.mixer.music.load(self.song_path)
        pygame.mixer.music.play()

    def run(self) -> None:
        self.running = True
        while self.running:
            dt = self.clock.tick(60) / 1000
            self.update(dt)
            pygame.display.flip()

        pygame.quit()
        sys.exit()

    def update_cat_animation(self):
        current_time = time.perf_counter() - self.rhythm.start_time
        
        # animation loop = 2 beats
        loop_beats = 2
        beat_phase = (current_time / self.rhythm.beat_duration) % loop_beats
        normalized = beat_phase / loop_beats
        
        self.cat_frame_index = int(normalized * self.num_cat_frames) % self.num_cat_frames
        self.cat_frame = self.cat_frames[self.cat_frame_index]

    def update(self, dt: float) -> None:
        self.screen.fill((0,0,0))
        
        # --- CAT
        self.update_cat_animation()
        if self.cat_frame:
            cat_scaled = pygame.transform.scale(self.cat_frame, (230, 250))
            self.screen.blit(cat_scaled, (150, 550))
        
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
                    event = self.rhythm.current_event()
                    event.hit = True
                    self.show_message("Perfect!", 1)
                    self.score += 1
                    self.used_current_char = True
                else:
                    self.show_message("Miss!", 1)
                    self.misses += 1
                    self.used_current_char = True
        

        self.render_timeline()
        
        # ----- SCORE / MISSES

        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        miss_text = self.font.render(f"Misses: {self.misses}", True, (255, 100, 100))
        self.screen.blit(score_text, (1300, 750))
        self.screen.blit(miss_text, (1300, 800))
        
        if self.message and self.message_duration > 0:
            self.draw_text(self.message, False)
            self.message_duration -= dt


    # --- RENDER TIMELINE 

    def render_timeline(self):
        current_time = time.perf_counter() - self.rhythm.start_time
        
        # show current word
        current_word = self.rhythm.current_expected_word()
        if current_word:
            # draw each character with underline for current character
            char_spacing = 60
            total_width = len(current_word) * char_spacing
            start_x = 800 - total_width // 2 #need something for 960
            
            for i, char in enumerate(current_word):
                char_surface = self.font.render(char, True, (255, 255, 255))
                char_x = start_x + i * char_spacing
                self.screen.blit(char_surface, (char_x, 150))
                
                # underline current character
                if self.rhythm.char_event_idx < len(self.rhythm.beat_map):
                    current_event = self.rhythm.beat_map[self.rhythm.char_event_idx]
                    if current_event.word_text == current_word and current_event.char == char and current_event.char_idx == i:
                        centering_offset = -10 # magic num for now
                        line_x = char_x + centering_offset
                        pygame.draw.line(self.screen, (255, 255, 255), 
                                       (line_x, 200), (line_x + C.UNDERLINE_LEN, 200), 3)
        
        # --- draw timeline
        timeline_y = 380
        timeline_start_x = 300
        timeline_end_x = 1500
        
        pygame.draw.line(self.screen, (255, 255, 255), 
                        (timeline_start_x, timeline_y), 
                        (timeline_end_x, timeline_y), 6)
        
        # hit marker
        hit_marker_x = C.HIT_X
        hit_marker_x -= C.HIT_MARKER_X_OFFSET
        #shift to include grace period
        
        pygame.draw.line(self.screen, (155, 255, 100), 
                        (hit_marker_x - C.HIT_MARKER_X_OFFSET, timeline_y), 
                        (hit_marker_x + C.HIT_MARKER_X_OFFSET, timeline_y), C.HIT_MARKER_WIDTH) # blip hitmarker img to it
        
        pygame.draw.line(self.screen, (0, 180, 220), 
                        (hit_marker_x, timeline_y  - C.HIT_MARKER_Y_OFFSET), 
                        (hit_marker_x, timeline_y + C.HIT_MARKER_Y_OFFSET), int(C.HIT_MARKER_WIDTH/2))
        
        # --- draw beat markers/notes
        # take account of grace
        grace = (C.GRACE * C.SCROLL_SPEED)
        hit_marker_x -= grace/6
        
        for event in self.rhythm.beat_map:
            time_until_hit = event.timestamp - current_time
            
            # --- calculate position

            if -0.75 < time_until_hit < 5.0:
                
                marker_x = hit_marker_x + (time_until_hit * C.SCROLL_SPEED)
                
                if timeline_start_x <= marker_x <= timeline_end_x:
                    if event.char != "" and not event.hit:
                        radius = 10

                        if time_until_hit < 0:
                            color = C.MISSED_COLOR
                        else:
                            color = C.COLOR

                        pygame.draw.circle(
                            self.screen,
                            color,
                            (int(marker_x), timeline_y),
                            radius
                        )
        
        # --- beat grid lines
        current_beat = current_time / self.rhythm.beat_duration
        for i in range(int(current_beat) - 8, int(current_beat) + 16):
            t = i * self.rhythm.beat_duration
            time_until = t - current_time
            x = hit_marker_x + time_until * C.SCROLL_SPEED
            
            if timeline_start_x <= x <= timeline_end_x:
                if i % 4 == 0:
                    # measure lines
                    pygame.draw.line(self.screen, (255, 255, 255), 
                                   (x, timeline_y - 50), (x, timeline_y + 50), 4)
                else:
                    # beat lines
                    pygame.draw.line(self.screen, (100, 100, 100), 
                                   (x, timeline_y - 30), (x, timeline_y + 30), 2)

    def show_message(self, txt: str, secs: float):
        self.message = txt
        self.message_duration = secs
    
    def draw_text(self, txt: str, left: bool):
        text_surface = self.font.render(txt, True, (255, 255, 255))
        if left:
            self.screen.blit(text_surface, (100, 100))
        else:
            text_rect = text_surface.get_rect(center=(1100, 250))
            self.screen.blit(text_surface, text_rect)
    
    def draw_curr_word(self, txt: str):
        self.draw_text(txt, True)