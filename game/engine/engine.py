import pygame
import sys

from game.rhythm import RhythmManager
from game.input import Input

pygame.init()

# PRIORITIES
# - match word list to accomodate song length
# - MAX CUSTOMIZABILITY -> allow users to upload their own songs -> 
# will have to decide how many chars (words per song

# MAX CUSTOMIZABILITY 2 -> allow users to upload their own words.

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((1920, 1080)) # how to adjust based off screen size resolution
        self.clock = pygame.time.Clock()
        self.running = True

        self.rhythm = RhythmManager()
        self.input = Input()

    def run(self):
        while self.running:
            dt = self.clock.tick(60) / 1000
            self.screen.fill((0, 0, 0))

            self.handle_events()
            self.update()

            pygame.display.flip()

        pygame.quit()
        sys.exit()

    def handle_events(self):
        for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
    