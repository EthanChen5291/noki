import pygame
import imageio
import numpy as np
import sys
import time
import os
import math

from . import constants as C
from .rhythm import RhythmManager, calculate_lead_in
from .input import Input
from .beatmap_generator import generate_beatmap
from analysis.audio_analysis import (
    analyze_song_intensity,
    get_sb_info,
    group_info_by_section,
    filter_sb_info,
    get_song_info,
    detect_loudest_drop,
    classify_pace,
    calculate_energy_shifts
)
from . import models as M

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
        lead_in = calculate_lead_in(self.song.bpm)
        self.rhythm = RhythmManager(beatmap, self.song.bpm, lead_in=lead_in)

        self.input = Input()

        # --- detect drop for shockwave effect
        self.drop_event = detect_loudest_drop(self.song_path, self.song.bpm)
        self.shockwaves: list[M.Shockwave] = []
        self.drop_triggered = False
        self.drop_note_idx = self._find_drop_note_idx()

        # --- classify pace and set scroll speed
        self.pace_profile = classify_pace(self.song_path, self.song.bpm)
        self.base_scroll_speed = C.SCROLL_SPEED

        self.pace_bias = 0.85 + self.pace_profile.pace_score * 1.3

        self.scroll_speed = self.base_scroll_speed * self.pace_bias 

        # --- calculate dynamic energy shifts
        self.energy_shifts = calculate_energy_shifts(
            self.song_path,
            self.song.bpm,
            self.pace_profile.pace_score
        )

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

        # animation loop is TWO BEATSw
        loop_beats = 2
        beat_phase = (current_time / self.rhythm.beat_duration) % loop_beats
        normalized = beat_phase / loop_beats

        self.cat_frame_index = int(normalized * self.num_cat_frames) % self.num_cat_frames
        self.cat_frame = self.cat_frames[self.cat_frame_index]

    def _find_drop_note_idx(self) -> int:
        """Find the beatmap note index closest to the detected drop timestamp"""
        if not self.drop_event or not self.rhythm.beat_map:
            return -1

        drop_time = self.drop_event.timestamp + self.rhythm.lead_in
        best_idx = -1
        best_diff = float('inf')

        for i, event in enumerate(self.rhythm.beat_map):
            if event.is_rest or not event.char:
                continue

            diff = abs(event.timestamp - drop_time)
            if diff < best_diff:
                best_diff = diff
                best_idx = i

        return best_idx

    def trigger_shockwave(self):
        """Spawn multiple expanding shockwave rings"""
        center_x = self.screen.get_width() // 2
        center_y = self.screen.get_height() // 2

        num_rings = 5
        for i in range(num_rings):
            initial_radius = i * 30
            shockwave = M.Shockwave(
                center_x=center_x,
                center_y=center_y,
                radius=initial_radius,
                max_radius=800,
                alpha=150,
                thickness=4,
                speed=400 + i * 50
            )
            self.shockwaves.append(shockwave)

    def check_drop_note_hit(self, hit_note_idx: int):
        """Check if the hit note is the drop note and trigger shockwave"""
        if not self.drop_triggered and hit_note_idx == self.drop_note_idx:
            self.trigger_shockwave()
            self.drop_triggered = True

    def update_shockwaves(self, dt: float):
        """Update and render active shockwaves"""
        surviving = []
        for wave in self.shockwaves:
            if wave.update(dt):
                surviving.append(wave)
                self.render_shockwave(wave)

        self.shockwaves = surviving

    def update_dynamic_scroll_speed(self, current_time: float):
        song_time = current_time

        speed = self.base_scroll_speed * self.pace_bias

        active_shift = None
        for shift in self.energy_shifts:
            if shift.start_time <= song_time < shift.end_time:
                active_shift = shift
                break

        if active_shift:
            speed *= active_shift.scroll_modifier

        self.scroll_speed = speed



    def draw_speed_arrow(self, x: int, y: int, speed_up: bool):
        """Draw a speed change arrow (like Geometry Dash portals)"""
        arrow_color = (0, 180, 255) if speed_up else (100, 180, 255)
        arrow_height = 80
        arrow_width = 32

        if speed_up:
            # right arrow
            points = [
                (x - arrow_width // 2, y - arrow_height // 2),
                (x + arrow_width // 2, y),
                (x - arrow_width // 2, y + arrow_height // 2),
            ]
        else:
            # left arrow
            points = [
                (x + arrow_width // 2, y - arrow_height // 2),
                (x - arrow_width // 2, y),
                (x + arrow_width // 2, y + arrow_height // 2),
            ]

        pygame.draw.polygon(self.screen, arrow_color, points)
        #pygame.draw.polygon(self.screen, (255, 255, 255), points, 2) idk if i should keep outline

    def render_shockwave(self, wave: M.Shockwave):
        """Render a single shockwave ring with realistic gray coloring"""
        if wave.alpha <= 0 or wave.radius <= 0:
            return

        diameter = int(wave.radius * 2) + wave.thickness * 2
        if diameter <= 0:
            return

        surface = pygame.Surface((diameter, diameter), pygame.SRCALPHA)

        gray_value = 180
        color = (gray_value, gray_value, gray_value, wave.alpha)

        center = diameter // 2
        if wave.thickness > 0 and wave.radius > 0:
            pygame.draw.circle(
                surface,
                color,
                (center, center),
                int(wave.radius),
                wave.thickness
            )

        blit_x = int(wave.center_x - center)
        blit_y = int(wave.center_y - center)
        self.screen.blit(surface, (blit_x, blit_y))

    def update(self, dt: float) -> None:
        self.screen.fill((0, 0, 0))

        current_time = time.perf_counter() - self.rhythm.start_time

        # --- DYNAMIC SCROLL: Adjust speed based on energy shifts
        self.update_dynamic_scroll_speed(current_time)

        # --- SHOCKWAVE: Update active shockwaves
        self.update_shockwaves(dt)

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

                    self.check_drop_note_hit(current_char_idx)

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
        
        for i in range(self.rhythm.char_event_idx, len(self.rhythm.beat_map)):
            event = self.rhythm.beat_map[i]
            if event.word_text and event.char_idx == 0 and not event.is_rest:
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
        center_y = 180  # vertical position (like in 3d space)
        
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
        grace = (C.GRACE * self.scroll_speed)
        hit_marker_x -= grace/6

        for event in self.rhythm.beat_map:
            time_until_hit = event.timestamp - current_time

            # --- calculate position

            if -0.75 < time_until_hit < 5.0:

                marker_x = hit_marker_x + (time_until_hit * self.scroll_speed)
                
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
            x = hit_marker_x + time_until * self.scroll_speed
            
            if timeline_start_x <= x <= timeline_end_x:
                if i % 4 == 0:
                    # measure lines
                    pygame.draw.line(self.screen, (255, 255, 255), 
                                   (x, timeline_y - 50), (x, timeline_y + 50), 4)
                else:
                    # beat lines
                    pygame.draw.line(self.screen, (100, 100, 100),
                                   (x, timeline_y - 30), (x, timeline_y + 30), 2)

        # --- draw speed change arrows at energy shift boundaries
        for shift in self.energy_shifts:

            shift_time = shift.start_time #+ self.rhythm.lead_in
            time_until = shift_time - current_time

            if -0.5 < time_until < 5.0:
                arrow_x = hit_marker_x + time_until * self.scroll_speed

                if timeline_start_x <= arrow_x <= timeline_end_x:
                    speed_up = shift.energy_delta > 0
                    self.draw_speed_arrow(int(arrow_x), timeline_y - 70, speed_up)

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