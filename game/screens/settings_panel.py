"""Settings panel — shown when the settings button is clicked on the title screen.

Rounded-rect popup (black bg, white border) with:
  - Volume slider
  - Empty space reserved for future additions
"""
from __future__ import annotations
import pygame

from ..menu_utils import _FONT
from ._constants import LEVEL_MENU_ANIM_DUR

_BORD = 2


class SettingsPanel:
    def __init__(self, screen, music, origin_rect=None):
        self.screen = screen
        self._music = music

        sw, sh = screen.get_size()
        pw = int(sw * 0.42)
        ph = int(sh * 0.46)
        px = (sw - pw) // 2
        py = (sh - ph) // 2
        self.rect = pygame.Rect(px, py, pw, ph)
        self._px, self._py, self._pw, self._ph = px, py, pw, ph

        pad = max(20, pw // 24)
        self._pad = pad

        title_sz = max(26, ph // 9)
        label_sz = max(16, ph // 18)
        self._title_font = pygame.font.Font(_FONT, title_sz)
        self._label_font = pygame.font.Font(_FONT, label_sz)

        # Volume slider geometry
        self._slider_cx     = px + pw // 2
        self._slider_y      = py + int(ph * 0.50)
        self._slider_half_w = int(pw * 0.34)
        self._slider_track_h = 8
        self._slider_knob_r  = 11

        # Current volume mirrors music master (or 0.75 fallback)
        self._volume = music.volume if music is not None else 0.75
        self._dragging = False

        # Close ×
        close_sz = 24
        self._close_rect = pygame.Rect(
            px + pw - pad - close_sz, py + pad // 2, close_sz, close_sz,
        )
        self._close_hovered = False

        self._overlay = pygame.Surface((sw, sh), pygame.SRCALPHA)

        self._origin = origin_rect.copy() if origin_rect else pygame.Rect(
            px + pw // 2, py + ph // 2, 0, 0,
        )

        self._open_elapsed:  float = 0.0
        self._close_elapsed: float = 0.0
        self._closing = False

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, dt: float, mouse_pos, mouse_clicked) -> str | None:
        """Returns 'close' or None."""
        self._update_animation(dt)
        if self._closing:
            if self._close_elapsed >= LEVEL_MENU_ANIM_DUR:
                return "close"
            return None
        return self._handle_input(mouse_pos, mouse_clicked)

    def draw(self) -> None:
        at  = self._anim_t()
        cur = self._lerp_rect(self._origin, self.rect, at)
        pad = self._pad

        self._overlay.fill((0, 0, 0, int(155 * at)))
        self.screen.blit(self._overlay, (0, 0))

        pygame.draw.rect(self.screen, (8, 8, 14),     cur, border_radius=14)
        pygame.draw.rect(self.screen, (255, 255, 255), cur, _BORD, border_radius=14)

        if at < 0.55:
            return
        content_alpha = min(255, int(255 * (at - 0.55) / 0.45))

        def _blit_a(surf, rect):
            s = surf.copy(); s.set_alpha(content_alpha)
            self.screen.blit(s, rect)

        # Title
        t_surf = self._title_font.render("SETTINGS", True, (220, 220, 220))
        _blit_a(t_surf, t_surf.get_rect(
            center=(self._px + self._pw // 2, self._py + int(self._ph * 0.15))
        ))

        # Divider
        rule_y = self._py + int(self._ph * 0.27)
        pygame.draw.line(
            self.screen,
            (int(50 * at), int(50 * at), int(60 * at)),
            (self._px + pad, rule_y), (self._px + self._pw - pad, rule_y), 1,
        )

        # Close ×
        xc = (255, 80, 80) if self._close_hovered else (120, 120, 130)
        xc = tuple(int(c * at) for c in xc)
        ccx, ccy = self._close_rect.center
        sz = 8
        pygame.draw.line(self.screen, xc, (ccx - sz, ccy - sz), (ccx + sz, ccy + sz), 2)
        pygame.draw.line(self.screen, xc, (ccx + sz, ccy - sz), (ccx - sz, ccy + sz), 2)

        # Volume label
        lbl_surf = self._label_font.render("Volume", True, (160, 160, 180))
        lbl_x = self._slider_cx - self._slider_half_w
        _blit_a(lbl_surf, lbl_surf.get_rect(midleft=(lbl_x, self._slider_y - 26)))

        # Track background
        track_rect = pygame.Rect(
            self._slider_cx - self._slider_half_w,
            self._slider_y - self._slider_track_h // 2,
            self._slider_half_w * 2,
            self._slider_track_h,
        )
        pygame.draw.rect(self.screen, (50, 50, 62), track_rect, border_radius=4)

        # Filled portion
        filled_w = max(0, int(self._slider_half_w * 2 * self._volume))
        if filled_w > 0:
            pygame.draw.rect(
                self.screen, (100, 180, 255),
                pygame.Rect(track_rect.x, track_rect.y, filled_w, track_rect.h),
                border_radius=4,
            )

        # Knob
        knob_x = int(self._slider_cx - self._slider_half_w + self._slider_half_w * 2 * self._volume)
        knob_col = (220, 220, 255) if self._dragging else (180, 180, 220)
        pygame.draw.circle(self.screen, knob_col, (knob_x, self._slider_y), self._slider_knob_r)
        pygame.draw.circle(self.screen, (255, 255, 255), (knob_x, self._slider_y), self._slider_knob_r, 2)

        # Percentage label
        pct_surf = self._label_font.render(f"{int(self._volume * 100)}%", True, (180, 180, 200))
        _blit_a(pct_surf, pct_surf.get_rect(
            midleft=(self._slider_cx + self._slider_half_w + 12, self._slider_y)
        ))

    # ── Private ───────────────────────────────────────────────────────────────

    def _handle_input(self, mouse_pos, mouse_clicked) -> str | None:
        self._close_hovered = self._close_rect.collidepoint(mouse_pos)
        mouse_held = pygame.mouse.get_pressed()[0]

        slider_left  = self._slider_cx - self._slider_half_w
        slider_right = self._slider_cx + self._slider_half_w
        knob_x = int(slider_left + (slider_right - slider_left) * self._volume)
        on_knob  = (abs(mouse_pos[0] - knob_x) <= self._slider_knob_r + 6 and
                    abs(mouse_pos[1] - self._slider_y) <= self._slider_knob_r + 6)
        on_track = (slider_left <= mouse_pos[0] <= slider_right and
                    abs(mouse_pos[1] - self._slider_y) <= 16)

        if mouse_clicked and (on_knob or on_track):
            self._dragging = True
        if not mouse_held:
            self._dragging = False

        if self._dragging:
            raw = (mouse_pos[0] - slider_left) / (slider_right - slider_left)
            self._volume = max(0.0, min(1.0, raw))
            if self._music is not None:
                self._music.volume = self._volume

        if mouse_clicked and not (on_knob or on_track):
            if self._close_hovered or not self.rect.collidepoint(mouse_pos):
                self._closing = True

        return None

    def _update_animation(self, dt: float) -> None:
        if self._closing:
            self._close_elapsed += dt
        else:
            self._open_elapsed += dt

    def _anim_t(self) -> float:
        if self._closing:
            t = min(1.0, self._close_elapsed / LEVEL_MENU_ANIM_DUR)
            return 1.0 - t * t
        else:
            t = min(1.0, self._open_elapsed / LEVEL_MENU_ANIM_DUR)
            return 1.0 - (1.0 - t) ** 3

    @staticmethod
    def _lerp_rect(r1: pygame.Rect, r2: pygame.Rect, t: float) -> pygame.Rect:
        return pygame.Rect(
            int(r1.x + (r2.x - r1.x) * t),
            int(r1.y + (r2.y - r1.y) * t),
            max(1, int(r1.w + (r2.w - r1.w) * t)),
            max(1, int(r1.h + (r2.h - r1.h) * t)),
        )
