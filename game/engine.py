import pygame
import imageio
import numpy as np
import sys
import time
import os
import math

from . import constants as C
from .rhythm import RhythmManager
from .input import Input
from .beatmap_generator import generate_beatmap
from analysis.audio_analysis import (
    analyze_song_intensity, 
    get_sb_info, 
    group_info_by_section, 
    filter_sb_info,
    get_song_info
)

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

        self.message = None
        self.message_duration = 0.0

        # word transition animation
        self._last_displayed_word = None
        self._previous_word = None
        self._word_transition_start = 0.0

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

                result = self.rhythm.check_input(key)
                
                if result['hit']:
                    judgment = result['judgment']
                    combo = result['combo']
                    
                    if judgment == 'perfect':
                        self.show_message(f"PERFECT! ×{combo}", 0.8)
                    elif judgment == 'good':
                        self.show_message(f"Good ×{combo}", 0.8)
                    elif judgment == 'ok':
                        self.show_message(f"OK ×{combo}", 0.8)
                    
                    self.score = self.rhythm.get_score()
                    self.used_current_char = True
                else:
                    # HANDLE MISSES
                    judgment = result['judgment']
                    if judgment == 'wrong':
                        self.show_message("Wrong Key!", 0.8)
                    else:
                        self.show_message("Miss!", 0.8)
                    
                    self.misses = self.rhythm.miss_count
                    self.used_current_char = True

        self.render_timeline()
        
        # ----- SCORE / MISSES

        stats = self.rhythm.get_stats()

        score_text = self.font.render(f"Score: {stats['score']}", True, (255, 255, 255))
        combo_text = self.font.render(f"Combo: {stats['combo']}", True, (255, 200, 0))
        acc_text = self.font.render(f"Accuracy: {stats['accuracy']:.1f}%", True, (100, 255, 100))
        rank_text = self.font.render(f"Rank: {stats['rank']}", True, (255, 255, 255))

        self.screen.blit(score_text, (1300, 700))
        self.screen.blit(combo_text, (1300, 750))
        self.screen.blit(acc_text, (1300, 800))
        self.screen.blit(rank_text, (1300, 850))
        
    def get_next_word(self) -> str | None:
        """Get the next word that will be typed after current word completes"""
        if not self.rhythm.beat_map:
            return None
        
        # Find the next word in the beatmap
        for i in range(self.rhythm.char_event_idx, len(self.rhythm.beat_map)):
            event = self.rhythm.beat_map[i]
            if event.word_text and event.char_idx == 0 and not event.is_rest:
                # Found start of next word
                if event.word_text != self.rhythm.current_expected_word():
                    return event.word_text
        
        return None

    def draw_word_animated(
        self,
        word: str,
        position: str,  # 'left', 'center', 'right'
        transition_progress: float,
        is_current: bool,
        current_char_idx: int,
        fading_out: bool = False,
        adjacent_word_width: int = 0  # NEW parameter
    ):
        """Draw a word with 3D carousel rotation animation"""
        if not word:
            return
        
        base_char_spacing = 60

        if position == 'right':
            char_spacing = base_char_spacing * 0.7
        elif position == 'left':
            char_spacing = base_char_spacing * 0.7
        else:  #center
            char_spacing = base_char_spacing
        
        total_width = len(word) * char_spacing
        
        radius = 350  # radius of rotation circle
        center_offset = base_char_spacing / 2
        center_x = self.screen.get_width() // 2 + center_offset  # center of screen
        center_y = 150  # vertical position (like in 3d space)
        
        base_spacing = 100
        
        if position == 'center':
            target_angle = 0  # front center
            target_scale = 1.0
            target_alpha = 255
            target_color = (255, 255, 255)
            target_char_spacing = base_char_spacing
        elif position == 'right':
            # adjust angle based off combined word widths
            width_factor = (total_width + adjacent_word_width) / 2
            dynamic_spacing = base_spacing + width_factor * 0.3
            target_angle = dynamic_spacing / radius
            target_scale = 0.75
            target_alpha = 180
            target_color = (150, 150, 150)
            target_char_spacing = base_char_spacing * 0.7
        else:
            width_factor = (total_width + adjacent_word_width) / 2
            dynamic_spacing = base_spacing + width_factor * 0.3
            target_angle = -dynamic_spacing / radius
            target_scale = 0.75
            target_alpha = int(180 * (1 - transition_progress)) if fading_out else 180
            target_color = (150, 150, 150)
            target_char_spacing = base_char_spacing * 0.7
            
        if position == 'center' and transition_progress < 1.0:
            width_factor = (total_width + adjacent_word_width) / 2
            dynamic_spacing = base_spacing + width_factor * 0.3
            start_angle = dynamic_spacing / radius
            current_angle = start_angle + (target_angle - start_angle) * transition_progress
            
            current_scale = 0.75 + (target_scale - 0.75) * transition_progress
            current_alpha = int(180 + (target_alpha - 180) * transition_progress)
            
            gray_amount = int(150 + (255 - 150) * transition_progress)
            current_color = (gray_amount, gray_amount, gray_amount)

            start_char_spacing = base_char_spacing * 0.7
            current_char_spacing = start_char_spacing + (target_char_spacing - start_char_spacing) * transition_progress
        else:
            current_angle = target_angle
            current_scale = target_scale
            current_alpha = target_alpha
            current_color = target_color
            current_char_spacing = char_spacing
        
        x_offset = radius * math.sin(current_angle)
        z = radius * (1 - math.cos(current_angle))
        
        perspective_scale = 1.0 / (1.0 + z / 1000)
        final_scale = current_scale * perspective_scale
        final_alpha = int(current_alpha * perspective_scale)
        
        animated_total_width = len(word) * current_char_spacing * final_scale
    
        current_x = center_x + x_offset - animated_total_width / 2
        
        for i, char in enumerate(word):
            font_size = int(48 * final_scale)
            char_font = pygame.font.Font(None, font_size)
            
            char_surface = char_font.render(char, True, current_color)
            char_surface.set_alpha(final_alpha)
            
            char_x = current_x + i * (current_char_spacing * final_scale)
            char_y = center_y
            
            self.screen.blit(char_surface, (int(char_x), int(char_y)))
            
            if position == 'center' and is_current and self.rhythm.char_event_idx < len(self.rhythm.beat_map):
                current_event = self.rhythm.beat_map[self.rhythm.char_event_idx]
                if current_event.word_text == word and current_event.char_idx == i:
                    underline_width = int(C.UNDERLINE_LEN * final_scale)
                    line_x = char_x - 10 * final_scale
                    line_y = char_y + 50
                    
                    pygame.draw.line(
                        self.screen,
                        (255, 255, 255),
                        (int(line_x), int(line_y)),
                        (int(line_x + underline_width), int(line_y)),
                        3
                    )

    # --- RENDER TIMELINE 

    def render_timeline(self):
        current_time = time.perf_counter() - self.rhythm.start_time
        
        current_word = self.rhythm.current_expected_word()
        next_word = self.get_next_word()
        
        char_spacing = 60
        current_word_width = len(current_word) * char_spacing if current_word else 0
        next_word_width = len(next_word) * char_spacing if next_word else 0
        prev_word_width = len(self._previous_word) * char_spacing if hasattr(self, '_previous_word') and self._previous_word else 0
        
        if current_word != getattr(self, '_last_displayed_word', None):
            self._word_transition_start = current_time
            self._last_displayed_word = current_word
        
        transition_duration = 0.3  # secs
        if hasattr(self, '_word_transition_start'):
            transition_progress = min(1.0, (current_time - self._word_transition_start) / transition_duration)
        else:
            transition_progress = 1.0
        
        ease_progress = 1 - (1 - transition_progress) ** 3
        
        if current_word:
            self.draw_word_animated(
                current_word,
                position='center',
                transition_progress=ease_progress,
                is_current=True,
                current_char_idx=self.rhythm.char_event_idx,
                adjacent_word_width=next_word_width
            )
        
        if next_word:
            self.draw_word_animated(
                next_word,
                position='right',
                transition_progress=ease_progress,
                is_current=False,
                current_char_idx=-1,
                adjacent_word_width=current_word_width
            )
        
        if hasattr(self, '_previous_word') and self._previous_word is not None and transition_progress < 1.0:
            self.draw_word_animated(
                self._previous_word,
                position='left',
                transition_progress=ease_progress,
                is_current=False,
                current_char_idx=-1,
                fading_out=True,
                adjacent_word_width=current_word_width
            )
        
        if transition_progress >= 1.0:
            self._previous_word = current_word

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