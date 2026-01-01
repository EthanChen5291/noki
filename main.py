import pygame
import sys

pygame.init()

SCREEN_WIDTH = 900
SCREEN_HEIGHT = 600

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()

player_img = pygame.image.load("assets/images/player.png").convert_alpha()
player_rect = player_img.get_rect(center=(450, 300))

speed = 5
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
    screen.blit(player_img, player_rect)
    pygame.display.flip()

pygame.quit()
sys.exit()