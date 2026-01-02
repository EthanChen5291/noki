import pygame

class Input:
    def __init__(self):
        self.typed_chars = []
        self.enter = False
        self.backspace = False

    def update(self, events):
        self.enter = False
        self.backspace = False
        self.typed_chars.clear()

        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:
                    self.backspace = True
                elif event.key == pygame.K_RETURN:
                    self.enter = True
                elif event.unicode and event.unicode.isprintable():
                    self.typed_chars.append(event.unicode)

    