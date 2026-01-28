import pygame
import math
import time
import os


class Button:
    def __init__(self, rect, text, font, base_color=(255, 255, 255), hover_color=(200, 220, 255)):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.font = font
        self.base_color = base_color
        self.hover_color = hover_color
        self.is_hovered = False
        self._scale = 1.0
        self._target_scale = 1.0

    def check_hover(self, mouse_pos):
        self.is_hovered = self.rect.collidepoint(mouse_pos)
        self._target_scale = 1.08 if self.is_hovered else 1.0

    def check_click(self, mouse_pos, mouse_clicked):
        return mouse_clicked and self.rect.collidepoint(mouse_pos)

    def draw(self, screen, current_time):
        # smooth lerp toward target scale
        self._scale += (self._target_scale - self._scale) * 0.18
        scale = self._scale

        if self.is_hovered:
            color = self.hover_color
            glow_rect = self.rect.inflate(8, 8)
            pygame.draw.rect(screen, (80, 80, 100), glow_rect, 2, border_radius=8)
        else:
            color = self.base_color

        text_surface = self.font.render(self.text, True, color)
        scaled_w = int(text_surface.get_width() * scale)
        scaled_h = int(text_surface.get_height() * scale)
        if scaled_w > 0 and scaled_h > 0:
            scaled_surface = pygame.transform.smoothscale(text_surface, (scaled_w, scaled_h))
        else:
            scaled_surface = text_surface
        text_rect = scaled_surface.get_rect(center=self.rect.center)
        screen.blit(scaled_surface, text_rect)


class TitleScreen:
    def __init__(self, screen):
        self.screen = screen
        sw, sh = screen.get_size()
        self.title_font = pygame.font.Font(None, 120)
        self.button_font = pygame.font.Font(None, 64)
        self.play_button = Button(
            (sw // 2 - 100, sh // 2 + 40, 200, 70),
            "PLAY",
            self.button_font
        )
        self.title_y_base = sh // 2 - 80

    def update(self, mouse_pos, mouse_clicked, current_time):
        self.play_button.check_hover(mouse_pos)
        if self.play_button.check_click(mouse_pos, mouse_clicked):
            return "play"
        return None

    def draw(self, current_time):
        sw = self.screen.get_width()
        # floating title
        y_offset = 8 * math.sin(current_time * 2)
        title_surface = self.title_font.render("KEY DASH", True, (255, 255, 255))
        title_rect = title_surface.get_rect(center=(sw // 2, self.title_y_base + y_offset))
        self.screen.blit(title_surface, title_rect)

        self.play_button.draw(self.screen, current_time)


class DifficultyToggle:
    """Three small toggle buttons (J / C / M) for difficulty selection."""
    LABELS = ["J", "C", "M"]
    KEYS = ["journey", "classic", "master"]
    COLORS = {
        "J": (100, 200, 100),   # green
        "C": (200, 200, 100),   # yellow
        "M": (200, 100, 100),   # red
    }

    def __init__(self, x, y, font, size=36):
        self.buttons: list[Button] = []
        self.selected = 1  # default to Classic
        gap = 4
        for i, label in enumerate(self.LABELS):
            bx = x + i * (size + gap)
            self.buttons.append(Button(
                (bx, y, size, size),
                label,
                font,
                base_color=(120, 120, 120),
                hover_color=(255, 255, 255),
            ))

    @property
    def difficulty(self) -> str:
        return self.KEYS[self.selected]

    def check_hover(self, mouse_pos):
        for btn in self.buttons:
            btn.check_hover(mouse_pos)

    def check_click(self, mouse_pos, mouse_clicked) -> bool:
        """Returns True if any toggle was clicked."""
        for i, btn in enumerate(self.buttons):
            if btn.check_click(mouse_pos, mouse_clicked):
                self.selected = i
                return True
        return False

    def draw(self, screen, current_time):
        for i, btn in enumerate(self.buttons):
            if i == self.selected:
                # draw highlighted background
                highlight_color = self.COLORS[self.LABELS[i]]
                pygame.draw.rect(screen, highlight_color, btn.rect, border_radius=6)
            btn.draw(screen, current_time)


class LevelSelect:
    def __init__(self, screen, song_names):
        self.screen = screen
        self.song_names = song_names
        sw, sh = screen.get_size()

        self.header_font = pygame.font.Font(None, 72)
        self.button_font = pygame.font.Font(None, 48)
        self.back_font = pygame.font.Font(None, 52)
        self.diff_font = pygame.font.Font(None, 28)

        self.scroll_offset = 0
        self.button_height = 60
        self.button_spacing = 16
        self.list_top = 140

        self.level_buttons = []
        self.difficulty_toggles: list[DifficultyToggle] = []
        for i, name in enumerate(song_names):
            display_name = os.path.splitext(name)[0]
            btn_y = self.list_top + i * (self.button_height + self.button_spacing)
            btn = Button(
                (sw // 2 - 250, btn_y, 500, self.button_height),
                display_name,
                self.button_font
            )
            self.level_buttons.append(btn)
            # difficulty toggle to the right of the song button
            toggle_x = sw // 2 + 270
            toggle_y = btn_y + (self.button_height - 36) // 2
            self.difficulty_toggles.append(DifficultyToggle(toggle_x, toggle_y, self.diff_font))

        self.back_button = Button(
            (30, 30, 120, 50),
            "BACK",
            self.back_font,
            base_color=(180, 180, 180),
            hover_color=(255, 255, 255)
        )

        total_content = len(song_names) * (self.button_height + self.button_spacing)
        self.max_scroll = max(0, total_content - (sh - self.list_top - 40))

    def handle_scroll(self, event):
        if event.type == pygame.MOUSEWHEEL:
            self.scroll_offset -= event.y * 30
            self.scroll_offset = max(0, min(self.scroll_offset, self.max_scroll))

    def update(self, mouse_pos, mouse_clicked, current_time):
        self.back_button.check_hover(mouse_pos)
        if self.back_button.check_click(mouse_pos, mouse_clicked):
            return "back", -1

        adjusted_mouse = (mouse_pos[0], mouse_pos[1] + self.scroll_offset)
        for i, btn in enumerate(self.level_buttons):
            btn.check_hover(adjusted_mouse)
            self.difficulty_toggles[i].check_hover(adjusted_mouse)
            # check difficulty toggle click first (so it doesn't also trigger level select)
            self.difficulty_toggles[i].check_click(adjusted_mouse, mouse_clicked)
            if btn.check_click(adjusted_mouse, mouse_clicked):
                return "select", i

        return None, -1

    def draw(self, current_time):
        sw = self.screen.get_width()
        # header
        header = self.header_font.render("SELECT LEVEL", True, (255, 255, 255))
        header_rect = header.get_rect(center=(sw // 2, 70))
        self.screen.blit(header, header_rect)

        self.back_button.draw(self.screen, current_time)

        # clip area for scrollable list
        clip_rect = pygame.Rect(0, self.list_top - 10, sw, self.screen.get_height() - self.list_top)
        self.screen.set_clip(clip_rect)

        for i, btn in enumerate(self.level_buttons):
            shifted = Button(
                (btn.rect.x, btn.rect.y - self.scroll_offset, btn.rect.w, btn.rect.h),
                btn.text,
                btn.font,
                btn.base_color,
                btn.hover_color
            )
            shifted.is_hovered = btn.is_hovered
            shifted._scale = btn._scale
            shifted.draw(self.screen, current_time)

            # draw difficulty toggles (shifted for scroll)
            toggle = self.difficulty_toggles[i]
            for j, tbtn in enumerate(toggle.buttons):
                shifted_tbtn = Button(
                    (tbtn.rect.x, tbtn.rect.y - self.scroll_offset, tbtn.rect.w, tbtn.rect.h),
                    tbtn.text,
                    tbtn.font,
                    tbtn.base_color,
                    tbtn.hover_color,
                )
                shifted_tbtn.is_hovered = tbtn.is_hovered
                shifted_tbtn._scale = tbtn._scale
                if j == toggle.selected:
                    highlight_color = DifficultyToggle.COLORS[DifficultyToggle.LABELS[j]]
                    pygame.draw.rect(self.screen, highlight_color, shifted_tbtn.rect, border_radius=6)
                shifted_tbtn.draw(self.screen, current_time)

        self.screen.set_clip(None)


class MenuManager:
    def __init__(self, screen, clock, song_names):
        self.screen = screen
        self.clock = clock
        self.song_names = song_names
        self.state = "title"

        self.title_screen = TitleScreen(screen)
        self.level_select = LevelSelect(screen, song_names)

        # transition state
        self.transition_start = 0.0
        self.transition_duration = 0.5
        self.transition_origin = (screen.get_width() // 2, screen.get_height() // 2)
        self.transition_target_state = ""
        self.transition_selected = -1

    def run(self):
        """Main menu loop. Returns (song_index, difficulty_str) tuple or None if quit."""
        while True:
            current_time = time.time()
            mouse_pos = pygame.mouse.get_pos()
            mouse_clicked = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mouse_clicked = True
                if self.state == "level_select":
                    self.level_select.handle_scroll(event)

            self.screen.fill((0, 0, 0))

            if self.state == "title":
                action = self.title_screen.update(mouse_pos, mouse_clicked, current_time)
                self.title_screen.draw(current_time)
                if action == "play":
                    self._start_transition("level_select", self.title_screen.play_button.rect.center)

            elif self.state == "level_select":
                action, idx = self.level_select.update(mouse_pos, mouse_clicked, current_time)
                self.level_select.draw(current_time)
                if action == "back":
                    self._start_transition("title", self.level_select.back_button.rect.center)
                elif action == "select":
                    btn = self.level_select.level_buttons[idx]
                    origin = (btn.rect.centerx, btn.rect.centery - self.level_select.scroll_offset)
                    self._start_transition("launch", origin, idx)

            elif self.state == "transition":
                self._draw_transition(current_time)
                progress = (current_time - self.transition_start) / self.transition_duration
                if progress >= 1.0:
                    if self.transition_target_state == "launch":
                        difficulty = self.level_select.difficulty_toggles[self.transition_selected].difficulty
                        return (self.transition_selected, difficulty)
                    self.state = self.transition_target_state

            pygame.display.flip()
            self.clock.tick(60)

    def _draw_transition(self, current_time):
        progress = min(1.0, (current_time - self.transition_start) / self.transition_duration)
        # cubic ease-out
        ease = 1 - (1 - progress) ** 3

        # zoom toward origin
        scale = 1.0 + ease * 0.5
        alpha = int(255 * ease)

        sw, sh = self.screen.get_size()

        # Redraw current screen content
        if self.transition_target_state in ("level_select", "launch"):
            if hasattr(self, '_pre_transition_state') and self._pre_transition_state == "title":
                self.title_screen.draw(current_time)
            else:
                self.level_select.draw(current_time)
        else:
            self.level_select.draw(current_time)

        # Scale the screen toward origin
        if scale != 1.0:
            screen_copy = self.screen.copy()
            new_w = int(sw * scale)
            new_h = int(sh * scale)
            scaled = pygame.transform.smoothscale(screen_copy, (new_w, new_h))

            ox, oy = self.transition_origin
            blit_x = int(ox - ox * scale)
            blit_y = int(oy - oy * scale)

            self.screen.fill((0, 0, 0))
            self.screen.blit(scaled, (blit_x, blit_y))

        # Fade to black overlay
        overlay = pygame.Surface((sw, sh), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, alpha))
        self.screen.blit(overlay, (0, 0))

    def _start_transition(self, target_state, origin, selected=-1):
        # Store which screen we're transitioning FROM
        self._pre_transition_state = self.state
        self.state = "transition"
        self.transition_start = time.time()
        self.transition_origin = origin
        self.transition_target_state = target_state
        self.transition_selected = selected


class PauseScreen:
    """Pause overlay drawn on top of the game. Returns 'resume' or 'menu'."""

    BORDER_THICKNESS = 6
    FADE_DURATION = 0.25  # seconds for border/dim animation

    def __init__(self, screen):
        self.screen = screen
        sw, sh = screen.get_size()
        btn_font = pygame.font.Font(None, 56)
        self.resume_button = Button(
            (sw // 2 - 120, sh // 2 - 50, 240, 60),
            "RESUME",
            btn_font,
        )
        self.menu_button = Button(
            (sw // 2 - 120, sh // 2 + 30, 240, 60),
            "MAIN MENU",
            btn_font,
        )
        self.open_time = time.time()

    def update(self, mouse_pos, mouse_clicked):
        self.resume_button.check_hover(mouse_pos)
        self.menu_button.check_hover(mouse_pos)
        if self.resume_button.check_click(mouse_pos, mouse_clicked):
            return "resume"
        if self.menu_button.check_click(mouse_pos, mouse_clicked):
            return "menu"
        return None

    def draw(self, current_time):
        sw, sh = self.screen.get_size()
        t = min(1.0, (current_time - self.open_time) / self.FADE_DURATION)
        ease = 1 - (1 - t) ** 3  # cubic ease-out

        # dim overlay
        dim_alpha = int(120 * ease)
        overlay = pygame.Surface((sw, sh), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, dim_alpha))
        self.screen.blit(overlay, (0, 0))

        # white borders that lerp-accelerate inward from edges
        border_progress = ease
        thickness = int(self.BORDER_THICKNESS * border_progress)
        if thickness > 0:
            bright = int(255 * border_progress)
            c = (bright, bright, bright)
            # top
            pygame.draw.rect(self.screen, c, (0, 0, sw, thickness))
            # bottom
            pygame.draw.rect(self.screen, c, (0, sh - thickness, sw, thickness))
            # left
            pygame.draw.rect(self.screen, c, (0, 0, thickness, sh))
            # right
            pygame.draw.rect(self.screen, c, (sw - thickness, 0, thickness, sh))

        # buttons
        self.resume_button.draw(self.screen, current_time)
        self.menu_button.draw(self.screen, current_time)
