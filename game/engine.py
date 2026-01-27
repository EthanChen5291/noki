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
    detect_drops,
    classify_pace,
    calculate_energy_shifts,
    detect_dual_side_sections
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

        # --- detect dual-side sections early (needed for beatmap generation)
        self.pace_profile = classify_pace(self.song_path, self.song.bpm)

        self.dual_side_sections = detect_dual_side_sections(
            self.song_path,
            self.song.bpm,
            self.pace_profile.pace_score,
            self.song.beat_times
        )

        beatmap = generate_beatmap(
            word_list=level.word_bank,
            song=self.song,
            dual_side_sections=self.dual_side_sections
        )
        lead_in = calculate_lead_in(self.song.beat_times)
        self.rhythm = RhythmManager(beatmap, self.song.bpm, lead_in=lead_in)

        self.input = Input()

        # --- detect multiple drops for shockwave effects
        self.drop_events = detect_drops(
            self.song.beat_times,
            self.song_path,
            self.song.bpm
        )
        self.shockwaves: list[M.Shockwave] = []
        self.drops_triggered: set[int] = set()  # Track which drops have been triggered
        self.drop_note_indices = self._find_drop_note_indices()

        # --- classify pace and set scroll speed (pace_profile already computed above)
        self.base_scroll_speed = C.SCROLL_SPEED

        self.pace_bias = 0.85 + self.pace_profile.pace_score * 1.3

        self.scroll_speed = self.base_scroll_speed * self.pace_bias

        # --- calculate dynamic energy shifts (using aligned beat times)
        self.energy_shifts = calculate_energy_shifts(
            self.song_path,
            self.song.bpm,
            self.pace_profile.pace_score,
            self.song.beat_times  # pass aligned beat times
        )

        # --- cat position for dual-side mode animation
        self.cat_base_x = 150  # normal left position
        self.cat_center_x = screen_width // 2 - 115  # center position (accounting for cat width)
        self.cat_current_x = float(self.cat_base_x)
        self.cat_velocity = 0.0  # for momentum animation
        self.dual_side_active = False
        self.dual_side_visuals_active = False

        # --- timeline animation for dual-side mode
        self.timeline_normal_start = 300
        self.timeline_normal_end = 1500
        self.timeline_dual_start = 0
        self.timeline_dual_end = screen_width
        self.timeline_current_start = float(self.timeline_normal_start)
        self.timeline_current_end = float(self.timeline_normal_end)
        self.timeline_start_velocity = 0.0
        self.timeline_end_velocity = 0.0
        # Hit marker positions
        self.hit_marker_normal_x = C.HIT_X - C.HIT_MARKER_X_OFFSET
        self.hit_marker_dual_x = screen_width // 2
        self.hit_marker_current_x = float(self.hit_marker_normal_x)
        self.hit_marker_velocity = 0.0

        # Word y-position animation for dual-side mode
        self.word_normal_y = 180  # Normal position (top area)
        self.word_dual_y = 480  # Above the cat during dual mode
        self.word_current_y = float(self.word_normal_y)
        self.word_y_velocity = 0.0

        # --- track missed notes for shockwave effect in dual mode
        self.missed_note_shockwaves: set[int] = set()  # indices of notes that triggered miss shockwave

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
        song_time = current_time - self.rhythm.lead_in 

        beat_times = self.song.beat_times
        if not beat_times or len(beat_times) < 2:
            loop_beats = 2
            beat_phase = (current_time / self.rhythm.beat_duration) % loop_beats
            normalized = beat_phase / loop_beats
        else:
            current_beat_idx = 0
            for i, bt in enumerate(beat_times):
                if bt <= song_time:
                    current_beat_idx = i
                else:
                    break

            if current_beat_idx < len(beat_times) - 1:
                beat_start = beat_times[current_beat_idx]
                beat_end = beat_times[current_beat_idx + 1]
                beat_duration = beat_end - beat_start
                phase_in_beat = (song_time - beat_start) / beat_duration if beat_duration > 0 else 0
            else:
                phase_in_beat = 0

            loop_beats = 2
            beat_in_loop = (current_beat_idx % loop_beats) + phase_in_beat
            normalized = beat_in_loop / loop_beats

        self.cat_frame_index = int(normalized * self.num_cat_frames) % self.num_cat_frames
        self.cat_frame = self.cat_frames[self.cat_frame_index]

    def _find_drop_note_indices(self) -> dict[int, int]:
        """Find beatmap note indices closest to each drop timestamp.
        Returns dict mapping drop_index -> note_index."""
        if not self.drop_events or not self.rhythm.beat_map:
            return {}

        drop_to_note: dict[int, int] = {}

        for drop_idx, drop in enumerate(self.drop_events):
            drop_time = drop.timestamp + self.rhythm.lead_in
            best_note_idx = -1
            best_diff = float('inf')

            for i, event in enumerate(self.rhythm.beat_map):
                if event.is_rest or not event.char:
                    continue

                diff = abs(event.timestamp - drop_time)
                if diff < best_diff:
                    best_diff = diff
                    best_note_idx = i

            if best_note_idx >= 0:
                drop_to_note[drop_idx] = best_note_idx

        return drop_to_note

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

    def trigger_miss_shockwave(self, x: int, y: int):
        """Spawn a small shockwave at a missed note position"""
        shockwave = M.Shockwave(
            center_x=x,
            center_y=y,
            radius=5,
            max_radius=40,
            alpha=200,
            thickness=2,
            speed=150
        )
        self.shockwaves.append(shockwave)

    def check_drop_note_hit(self, hit_note_idx: int):
        """Check if the hit note is any drop note and trigger shockwave"""
        for drop_idx, note_idx in self.drop_note_indices.items():
            if drop_idx not in self.drops_triggered and hit_note_idx == note_idx:
                self.trigger_shockwave()
                self.drops_triggered.add(drop_idx)

    def update_shockwaves(self, dt: float):
        """Update and render active shockwaves"""
        surviving = []
        for wave in self.shockwaves:
            if wave.update(dt):
                surviving.append(wave)
                self.render_shockwave(wave)

        self.shockwaves = surviving

    def update_dynamic_scroll_speed(self, current_time: float):
        """Smoothly interpolate scroll speed based on energy shifts"""
        # Convert to actual song time (subtract lead-in)
        song_time = current_time - self.rhythm.lead_in

        target_speed = self.base_scroll_speed * self.pace_bias

        active_shift = None
        for shift in self.energy_shifts:
            if shift.start_time <= song_time < shift.end_time:
                active_shift = shift
                break

        if active_shift:
            # During dual-side mode, dampen speed-up effects (only apply 30% of the modifier)
            if self.dual_side_active:
                dampened_modifier = 1.0 + (active_shift.scroll_modifier - 1.0) * 0.3
                target_speed *= dampened_modifier
            else:
                target_speed *= active_shift.scroll_modifier

        # During dual-side mode, reduce overall speed to 70%
        if self.dual_side_active:
            target_speed *= 0.7

        if target_speed > self.scroll_speed:
            lerp_factor = 0.06
        else:
            lerp_factor = 0.04

        self.scroll_speed += (target_speed - self.scroll_speed) * lerp_factor

    def update_cat_position(self, current_time: float, dt: float):
        """
        Update cat position with momentum-style animation for dual-side mode.
        Quick acceleration when entering, slow deceleration to final position.
        """
        song_time = current_time - self.rhythm.lead_in

        # Delay before visual transition back to normal (1 beat)
        visual_exit_delay = self.beat_duration * 1

        # Check if we're in a dual-side section
        # (beatmap has a built-in rest at the start of each dual section for grace period)
        self.dual_side_active = False
        self.dual_side_visuals_active = False

        for dual_sec in self.dual_side_sections:
            if dual_sec.start_time <= song_time < dual_sec.end_time:
                self.dual_side_active = True
                self.dual_side_visuals_active = True
                break
            # Keep visuals active for a short delay after section ends
            elif dual_sec.end_time <= song_time < dual_sec.end_time + visual_exit_delay:
                self.dual_side_visuals_active = True
                break

        # Determine target position (based on visuals, not note directions)
        if self.dual_side_visuals_active:
            target_x = self.cat_center_x
        else:
            target_x = self.cat_base_x

        # Momentum-style animation: use spring physics for natural movement
        # Quick acceleration (high spring constant) then slow deceleration (damping)
        distance = target_x - self.cat_current_x

        # Spring constants - different for moving to center vs returning to base
        if self.dual_side_visuals_active:
            # Moving to center: quick and snappy
            if abs(distance) > 5:
                spring_strength = 8.0
                damping = 4.0
            else:
                spring_strength = 5.0
                damping = 6.0
        else:
            # Returning to base: smoother, gentler
            if abs(distance) > 5:
                spring_strength = 5.0
                damping = 5.0
            else:
                spring_strength = 4.0
                damping = 7.0

        # Spring physics: acceleration = spring_force - damping * velocity
        acceleration = spring_strength * distance - damping * self.cat_velocity
        self.cat_velocity += acceleration * dt
        self.cat_current_x += self.cat_velocity * dt

        # Clamp to prevent overshooting too far
        if self.dual_side_visuals_active:
            self.cat_current_x = max(self.cat_base_x, min(self.cat_current_x, self.cat_center_x + 50))
        else:
            self.cat_current_x = max(self.cat_base_x - 50, min(self.cat_current_x, self.cat_center_x))

    def update_timeline_animation(self, dt: float):
        """
        Animate timeline expansion/contraction for dual-side mode.
        Uses spring physics for momentum feel.
        """
        # Determine target positions (based on visuals, which start early)
        if self.dual_side_visuals_active:
            target_start = self.timeline_dual_start
            target_end = self.timeline_dual_end
            target_hit = self.hit_marker_dual_x
            target_word_y = self.word_dual_y
        else:
            target_start = self.timeline_normal_start
            target_end = self.timeline_normal_end
            # Apply grace adjustment for normal mode
            grace = (C.GRACE * self.scroll_speed)
            target_hit = self.hit_marker_normal_x - grace/6
            target_word_y = self.word_normal_y

        # On first frame or if uninitialized, snap to target (no animation)
        if not hasattr(self, '_timeline_initialized'):
            self.timeline_current_start = target_start
            self.timeline_current_end = target_end
            self.hit_marker_current_x = target_hit
            self.word_current_y = target_word_y
            self._timeline_initialized = True
            return

        # Spring constants - different for expanding vs contracting
        if self.dual_side_visuals_active:
            # Expanding to dual mode: quick and snappy
            spring_strength = 12.0
            damping = 5.0
        else:
            # Contracting back to normal: smoother, gentler
            spring_strength = 6.0
            damping = 7.0

        # Animate timeline start
        dist_start = target_start - self.timeline_current_start
        accel_start = spring_strength * dist_start - damping * self.timeline_start_velocity
        self.timeline_start_velocity += accel_start * dt
        self.timeline_current_start += self.timeline_start_velocity * dt

        # Animate timeline end
        dist_end = target_end - self.timeline_current_end
        accel_end = spring_strength * dist_end - damping * self.timeline_end_velocity
        self.timeline_end_velocity += accel_end * dt
        self.timeline_current_end += self.timeline_end_velocity * dt

        # Animate hit marker
        dist_hit = target_hit - self.hit_marker_current_x
        accel_hit = spring_strength * dist_hit - damping * self.hit_marker_velocity
        self.hit_marker_velocity += accel_hit * dt
        self.hit_marker_current_x += self.hit_marker_velocity * dt

        # Animate word y-position
        dist_word_y = target_word_y - self.word_current_y
        accel_word_y = spring_strength * dist_word_y - damping * self.word_y_velocity
        self.word_y_velocity += accel_word_y * dt
        self.word_current_y += self.word_y_velocity * dt

    def draw_dual_side_marker(self, x: int, timeline_y: int):
        """Draw a dual-side mode activation marker (two arrows pointing inward)"""
        arrow_height = 60
        arrow_width = 12

        # Colors for dual-side indicator
        left_color = (100, 200, 255)  # Light blue
        right_color = (255, 200, 100)  # Orange

        top_y = timeline_y - arrow_height // 2
        bottom_y = timeline_y + arrow_height // 2
        center_y = timeline_y

        # Left arrow (pointing right) - offset to the left of center
        left_x = x - 15
        left_points = [
            (left_x + arrow_width, center_y),
            (left_x, top_y),
            (left_x, bottom_y),
        ]
        pygame.draw.polygon(self.screen, left_color, left_points)
        pygame.draw.polygon(self.screen, (255, 255, 255), left_points, 2)

        # Right arrow (pointing left) - offset to the right of center
        right_x = x + 15
        right_points = [
            (right_x - arrow_width, center_y),
            (right_x, top_y),
            (right_x, bottom_y),
        ]
        pygame.draw.polygon(self.screen, right_color, right_points)
        pygame.draw.polygon(self.screen, (255, 255, 255), right_points, 2)

    def draw_speed_arrow(self, x: int, timeline_y: int, timeline_height: int, speed_up: bool):
        """Draw a speed change arrow spanning the full measure line"""
        if speed_up:
            arrow_color = (0, 200, 255)
            glow_color = (0, 100, 200, 80)
        else:
            arrow_color = (200, 150, 100)
            glow_color = (50, 100, 180, 80)

        arrow_height = timeline_height  # Span full measure line
        arrow_width = 24

        top_y = timeline_y - arrow_height // 2
        bottom_y = timeline_y + arrow_height // 2
        center_y = timeline_y

        if speed_up:
            for offset in [-8, 8]:
                points = [
                    (x - arrow_width // 2 + offset, top_y),
                    (x + arrow_width // 2 + offset, center_y),
                    (x - arrow_width // 2 + offset, bottom_y),
                ]
                pygame.draw.polygon(self.screen, arrow_color, points)
                pygame.draw.polygon(self.screen, (255, 255, 255), points, 2)
        else:
            for offset in [-8, 8]:
                points = [
                    (x + arrow_width // 2 + offset, top_y),
                    (x - arrow_width // 2 + offset, center_y),
                    (x + arrow_width // 2 + offset, bottom_y),
                ]
                pygame.draw.polygon(self.screen, arrow_color, points)
                pygame.draw.polygon(self.screen, (200, 220, 255), points, 2)

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

        self.update_dynamic_scroll_speed(current_time)
        self.update_cat_position(current_time, dt)
        self.update_timeline_animation(dt)

        self.update_shockwaves(dt)

        self.update_cat_animation()
        if self.cat_frame:
            cat_scaled = pygame.transform.scale(self.cat_frame, (230, 250))
            self.screen.blit(cat_scaled, (int(self.cat_current_x), 550))
        
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
        adjacent_word_width: int = 0
    ):
        """Draw a word with 3D carousel rotation animation"""
        if not word:
            return
        
        base_char_spacing = 60

        if position == 'right':
            char_spacing = base_char_spacing * 0.7
        elif position == 'left':
            char_spacing = base_char_spacing * 0.7
        else:
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

    def draw_background_word(self, word: str):
        """Draw the current word as a large, faded background element during dual-side mode"""
        if not word:
            return

        # Large font for background word
        font_size = 180
        bg_font = pygame.font.Font(None, font_size)

        # Gray, semi-transparent color
        bg_color = (60, 60, 60)

        # Render each character with spacing
        char_spacing = 120
        total_width = len(word) * char_spacing
        start_x = (self.screen.get_width() - total_width) // 2
        center_y = self.screen.get_height() // 2 - 50  # Slightly above center

        for i, char in enumerate(word):
            char_surface = bg_font.render(char, True, bg_color)
            char_x = start_x + i * char_spacing
            char_rect = char_surface.get_rect(center=(char_x + char_spacing // 2, center_y))
            self.screen.blit(char_surface, char_rect)

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

        # During dual-side mode, show large background word instead of carousel
        if self.dual_side_visuals_active:
            # Draw large faded background word
            if current_word:
                self.draw_background_word(current_word)
        else:
            # Normal word carousel display
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

        # --- draw timeline (using animated positions)
        timeline_y = 380
        timeline_start_x = int(self.timeline_current_start)
        timeline_end_x = int(self.timeline_current_end)
        hit_marker_x = self.hit_marker_current_x

        pygame.draw.line(self.screen, (255, 255, 255),
                        (timeline_start_x, timeline_y),
                        (timeline_end_x, timeline_y), 6)

        # Draw visual hit marker at the actual hit position
        pygame.draw.line(self.screen, (155, 255, 100),
                        (int(hit_marker_x) - C.HIT_MARKER_X_OFFSET, timeline_y),
                        (int(hit_marker_x) + C.HIT_MARKER_X_OFFSET, timeline_y), C.HIT_MARKER_WIDTH)

        pygame.draw.line(self.screen, (0, 180, 220),
                        (int(hit_marker_x), timeline_y - C.HIT_MARKER_Y_OFFSET),
                        (int(hit_marker_x), timeline_y + C.HIT_MARKER_Y_OFFSET), int(C.HIT_MARKER_WIDTH/2))

        # --- draw beat markers/notes

        for note_idx, event in enumerate(self.rhythm.beat_map):
            time_until_hit = event.timestamp - current_time

            # --- calculate position

            if -0.75 < time_until_hit < 5.0:
                # Determine if note comes from left at render time
                # (avoids timing mismatch between beatmap generation and runtime)
                note_from_left = False
                if self.dual_side_active and event.char_idx >= 0:
                    # Alternate sides based on character index within word
                    note_from_left = (event.char_idx % 2 == 1)

                # Notes from the left side move right (from left edge toward hit marker)
                # Notes from the right side move left (from right edge toward hit marker)
                if note_from_left:
                    # Left-side notes: start from left edge, move toward hit marker
                    marker_x = hit_marker_x - (time_until_hit * self.scroll_speed)
                else:
                    # Right-side notes: normal behavior (start from right, move toward hit marker)
                    marker_x = hit_marker_x + (time_until_hit * self.scroll_speed)

                if timeline_start_x <= marker_x <= timeline_end_x:
                    if event.char != "" and not event.hit:
                        is_missed = time_until_hit < 0
                        radius = 10

                        # In dual mode, handle missed notes specially
                        if self.dual_side_active and is_missed:
                            # Trigger miss shockwave if not already triggered
                            if note_idx not in self.missed_note_shockwaves:
                                self.trigger_miss_shockwave(int(marker_x), timeline_y)
                                self.missed_note_shockwaves.add(note_idx)
                            # Don't render the missed note - it disappears
                            continue

                        if is_missed:
                            color = C.MISSED_COLOR
                        else:
                            color = C.COLOR

                        # Draw the main note circle
                        pygame.draw.circle(
                            self.screen,
                            color,
                            (int(marker_x), timeline_y),
                            radius
                        )

                        # During dual-side mode, draw the character above each note
                        if self.dual_side_active:
                            # Draw character above note
                            char_font = pygame.font.Font(None, 36)
                            char_surface = char_font.render(event.char, True, (255, 255, 255))
                            char_rect = char_surface.get_rect(center=(int(marker_x), timeline_y - 35))
                            self.screen.blit(char_surface, char_rect)

                            # Draw small direction indicator below char
                            indicator_y = timeline_y - 18
                            indicator_radius = 3

                            if note_from_left:
                                indicator_color = (100, 200, 255)  # Light blue for left
                            else:
                                indicator_color = (255, 200, 100)  # Orange for right

                            pygame.draw.circle(
                                self.screen,
                                indicator_color,
                                (int(marker_x), indicator_y),
                                indicator_radius
                            )
        
        # --- beat grid lines (using actual beat times from librosa)
        beat_times = self.song.beat_times
        lead_in = self.rhythm.lead_in

        for i, beat_time in enumerate(beat_times):
            # Offset by lead_in to match beatmap timestamps
            t = beat_time + lead_in
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

        # --- draw dual-side mode indicators
        if self.dual_side_visuals_active:
            # Draw arrows on both sides of the timeline indicating dual direction
            arrow_size = 15
            arrow_y = timeline_y

            # Left arrow (pointing right) near timeline start
            left_arrow_x = timeline_start_x + 30
            pygame.draw.polygon(
                self.screen,
                (100, 200, 255),  # Light blue
                [
                    (left_arrow_x, arrow_y),
                    (left_arrow_x - arrow_size, arrow_y - arrow_size // 2),
                    (left_arrow_x - arrow_size, arrow_y + arrow_size // 2),
                ]
            )

            # Right arrow (pointing left) near timeline end
            right_arrow_x = timeline_end_x - 30
            pygame.draw.polygon(
                self.screen,
                (255, 200, 100),  # Orange
                [
                    (right_arrow_x, arrow_y),
                    (right_arrow_x + arrow_size, arrow_y - arrow_size // 2),
                    (right_arrow_x + arrow_size, arrow_y + arrow_size // 2),
                ]
            )

        # --- draw dual-side section start markers (like speed arrows)
        for dual_sec in self.dual_side_sections:
            section_time = dual_sec.start_time + self.rhythm.lead_in
            time_until = section_time - current_time

            if -0.5 < time_until < 5.0:
                marker_x = hit_marker_x + time_until * self.scroll_speed

                if timeline_start_x <= marker_x <= timeline_end_x:
                    # Draw dual-side indicator (two arrows pointing at each other)
                    self.draw_dual_side_marker(int(marker_x), timeline_y)

        # --- draw speed change arrows at energy shift boundaries
        timeline_height = 100  # Matches measure line height (±50 from center)
        lead_in = self.rhythm.lead_in

        for shift in self.energy_shifts:
            # Add lead_in to shift time to align with beatmap timeline
            shift_time = shift.start_time + lead_in
            time_until = shift_time - current_time

            if -0.5 < time_until < 5.0:
                arrow_x = hit_marker_x + time_until * self.scroll_speed

                if timeline_start_x <= arrow_x <= timeline_end_x:
                    speed_up = shift.energy_delta > 0
                    self.draw_speed_arrow(int(arrow_x), timeline_y, timeline_height, speed_up)

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