import pygame
import cv2
import sys
import time
import os

from . import constants as C
from .rhythm import RhythmManager
from .input import Input
from .beatmap_generator import generate_beatmap

pygame.init()

class Song:
    def __init__(self, path, bpm, duration):
        self.bpm = bpm
        self.duration = duration
        self.file_path = path

class Level:
    def __init__(self, bg_path, cat_sprite_path, word_bank, song):
        self.bg_path = bg_path
        self.cat_sprite_path = cat_sprite_path
        self.word_bank = word_bank
        self.song = song

class Game:
    def __init__(self, level) -> None:
        self.screen = pygame.display.set_mode((1920, 1080))
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

        self.level = level
        self.song = level.song
        self.file_path = level.song.file_path
        pygame.mixer.init()
        pygame.mixer.music.load(self.file_path)
        pygame.mixer.music.play()

        # --- load assets
        assets_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'images')
        
        bg_file = os.path.join(assets_path, 'noki_bg.png')
        self.background = pygame.image.load(bg_file).convert()
        self.background = pygame.transform.scale(self.background, (1512, 982))
        
        #machine_file = os.path.join(assets_path, 'noki_machinev1.png')
        #self.machine = pygame.image.load(machine_file).convert_alpha()
        #self.machine = pygame.transform.scale(self.machine, (300, 350))
        
        timeline_file = os.path.join(assets_path, 'noki_timeline.png')
        self.timeline_img = pygame.image.load(timeline_file).convert_alpha()
        self.timeline_img = pygame.transform.scale(self.timeline_img, (1920, 200))
        
        cat_file = os.path.join(assets_path, 'noki_cat.mov')
        self.cat_video = cv2.VideoCapture(cat_file)

        self.beat_duration = 60 / self.song.bpm
        total_frames = int(self.cat_video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # loop repeats every 2 beats
        loop_duration = self.beat_duration * 2
        self.cat_fps = total_frames / loop_duration if loop_duration > 0 else 30
        self.cat_frame_time = 1.0 / self.cat_fps if self.cat_fps > 0 else 1.0 / 30
        self.cat_time_accumulator = 0.0
        self.cat_frame = None

        beatmap = generate_beatmap(
            word_list=level.word_bank,
            bpm=level.song.bpm,
            song_duration=level.song.duration
        )
        self.rhythm = RhythmManager(beatmap, level.song.bpm)
        
        self.input = Input()

    def run(self) -> None:
        self.running = True
        while self.running:
            dt = self.clock.tick(60) / 1000
            self.update(dt)
            pygame.display.flip()

        self.cat_video.release()
        pygame.quit()
        sys.exit()

    def update_cat_video(self, dt: float):
        """Update and loop cat video synced to beat"""
        self.cat_time_accumulator += dt
        
        if self.cat_time_accumulator >= self.cat_frame_time:
            self.cat_time_accumulator = 0.0
            
            ret, frame = self.cat_video.read()
            
            if not ret:
                self.cat_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cat_video.read()
            
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                frame = cv2.flip(frame, 1)
                
                self.cat_frame = pygame.surfarray.make_surface(frame)

    def update(self, dt: float) -> None:
        # bg
        self.screen.blit(self.background, (0, 0))
        
        # cat machine
        #self.screen.blit(self.machine, (80, 480))
        # CAT
        self.update_cat_video(dt)
        if self.cat_frame:
            cat_scaled = pygame.transform.scale(self.cat_frame, (230, 250))
            self.screen.blit(cat_scaled, (390, 550))
        
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