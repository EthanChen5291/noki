import pygame

class Input:
    def __init__(self):
        self.typed_chars = []
        self.released_chars = []  # printable chars released this frame (for hold notes)
        self.enter = False
        self.backspace = False
        self._held_keys: dict[int, str] = {}  # key_code -> char for currently held keys

    def update(self, events):
        self.enter = False
        self.backspace = False
        self.typed_chars.clear()
        self.released_chars.clear()

        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:
                    self.backspace = True
                elif event.key == pygame.K_RETURN:
                    self.enter = True
                elif event.unicode and event.unicode.isprintable():
                    self.typed_chars.append(event.unicode)
                    self._held_keys[event.key] = event.unicode
            elif event.type == pygame.KEYUP:
                char = self._held_keys.pop(event.key, None)
                if char:
                    self.released_chars.append(char)

    