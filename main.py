import pygame
import sys

pygame.init()

# PRIORITIES
# - match word list to accomodate song length
# - MAX CUSTOMIZABILITY -> allow users to upload their own songs -> 
# will have to decide how many chars (words per song

# MAX CUSTOMIZABILITY 2 -> allow users to upload their own words.

SCREEN_WIDTH = 900
SCREEN_HEIGHT = 600

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()

running = True

# keyboard mapping -> move to input.py 
# random generation of words?
while running:
    clock.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    key = pygame.key.get_pressed()
    if key[pygame.K_a] == True:
        player_rect.x -= speed
    elif key[pygame.K_d] == True:
        player_rect.x += speed
    elif key[pygame.K_w] == True:
        player_rect.y -= speed
    elif key[pygame.K_s] == True:
        player_rect.y += speed

    screen.fill((0, 0, 0))
    pygame.display.flip()

pygame.quit()
sys.exit()