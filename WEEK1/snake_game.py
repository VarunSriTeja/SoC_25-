import pygame, sys, random
from pygame.math import Vector2

class SNAKE:
    def __init__(self):
        self.body = [Vector2(5,10), Vector2(4,10),Vector2(3,10)]
        self.direction = Vector2(1, 0)
        self.new_block = False
    
    def draw_snake(self):
        for index, block in enumerate(self.body):
            x_pos = int(block.x * cell_size)
            y_pos = int(block.y * cell_size)
            block_rect = pygame.Rect(x_pos , y_pos , cell_size, cell_size)
            if index == 0:
                center = (x_pos + cell_size // 2, y_pos + cell_size // 2)
                radius = cell_size // 2
                pygame.draw.circle(screen, (100, 156, 255), center, radius)
            else:
                pygame.draw.rect(screen, (100, 156, 255), block_rect, border_radius=6)

    def move_snake(self):
        if self.new_block == True:
            body_copy = self.body[:]
            body_copy.insert(0, body_copy[0] + self.direction)
            self.body = body_copy[:]
            self.new_block = False
        else:
            body_copy = self.body[:-1]
            body_copy.insert(0, body_copy[0] + self.direction)
            self.body = body_copy[:]
    
    def add_block(self):
        self.new_block = True
    
    def reset(self):
        self.body = [Vector2(5,10), Vector2(4,10),Vector2(3,10)]
        self.direction = Vector2(1, 0)
    
    
class FRUIT:
    def __init__(self):
        self.randomize()

    def draw_fruit(self):
        x = int(self.pos.x * cell_size + cell_size // 2)
        y = int(self.pos.y * cell_size + cell_size // 2)
        pygame.draw.circle(screen, (255, 0, 0), (x, y), cell_size // 2 -1)


    def randomize(self):
        self.x = random.randint(0, cell_number - 1)
        self.y = random.randint(0, cell_number - 1)
        self.pos = Vector2(self.x, self.y)

class MAIN:
    def __init__(self):
        self.snake = SNAKE()
        self.fruit = FRUIT()
        self.is_game_over = True
        self.restart_button_rect = pygame.Rect(0, 0, 0, 0)

    def update(self):
        if not self.is_game_over:
            self.snake.move_snake()
            self.collision()
            self.check_fail()

    def draw_ele(self):
        if not self.is_game_over:
            self.draw_grass()
            self.fruit.draw_fruit()
            self.snake.draw_snake()
            self.draw_score()
        else:
            self.draw_game_over()

    def collision(self):
        if self.fruit.pos == self.snake.body[0]:
            self.fruit.randomize() 
            self.snake.add_block()
        
        for block in self.snake.body[1:]:
            if block == self.fruit.pos:
                self.fruit.randomize()
    
    def check_fail(self):
        if not 0 <= self.snake.body[0].x < cell_number or not 0 <= self.snake.body[0].y < cell_number:
            self.game_over()

        for block in self.snake.body[1:]:
            if block == self.snake.body[0]:
                self.game_over()

    def draw_grass(self):
        grass_color = (167,209,61)
        for row in range(cell_number):
            if row % 2 == 0:
                for col in range(cell_number):
                    if col % 2 == 0:
                        grass_rect = pygame.Rect(col * cell_size, row * cell_size , cell_size, cell_size)
                        pygame.draw.rect(screen, grass_color, grass_rect)
            else:
                for col in range(cell_number):
                    if col % 2 != 0:
                        grass_rect = pygame.Rect(col * cell_size, row * cell_size , cell_size, cell_size)
                        pygame.draw.rect(screen, grass_color, grass_rect)
    
    def draw_score(self):
        score_text = str(len(self.snake.body) - 3)
        score_surface = game_font.render(score_text, True, (0,0,0))
        score_x = int(cell_size * cell_number - 60)
        score_y = int(cell_size * cell_number - 40)
        score_rect = score_surface.get_rect(center = (score_x, score_y))
        apple_center = (score_rect.left - 20, score_rect.centery)
        pygame.draw.circle(screen, (255, 0, 0), apple_center, 10)

        screen.blit(score_surface, score_rect)
    
    def game_over(self):
        self.is_game_over = True

    def restart(self):
        self.snake.reset()
        self.fruit.randomize()
        self.is_game_over = False

    def draw_game_over(self):
        screen.fill((220, 255, 200))
        over_text = game_font.render("SNAKE GAME", True, (0, 0, 0))
        score_text = game_font.render(f"Score: {len(self.snake.body) - 3}", True, (0, 0, 0))
        restart_text = game_font.render("Click to Start", True, (255, 0, 0))

        over_rect = over_text.get_rect(center=(cell_size * cell_number // 2, cell_size * 6))
        score_rect = score_text.get_rect(center=(cell_size * cell_number // 2, cell_size * 8))
        restart_rect = restart_text.get_rect(center=(cell_size * cell_number // 2, cell_size * 10))

        self.restart_button_rect = restart_rect

        screen.blit(over_text, over_rect)
        screen.blit(score_text, score_rect)
        screen.blit(restart_text, restart_rect)

pygame.init()
cell_size = 40
cell_number = 20
screen = pygame.display.set_mode((cell_number * cell_size,cell_number * cell_size))
clock = pygame.time.Clock()

game_font = pygame.font.Font(None, 30)

SCREEN_UPDATE = pygame.USEREVENT
pygame.time.set_timer(SCREEN_UPDATE, 150)

main_game = MAIN()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == SCREEN_UPDATE:
            main_game.update()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT:
                if main_game.snake.direction.x != -1:
                    main_game.snake.direction = Vector2(1, 0)
            if event.key == pygame.K_LEFT:
                if main_game.snake.direction.x != 1:
                    main_game.snake.direction = Vector2(-1, 0)
            if event.key == pygame.K_UP:
                if main_game.snake.direction.y != 1:
                    main_game.snake.direction = Vector2(0, -1)
            if event.key == pygame.K_DOWN:
                if main_game.snake.direction.y != -1:
                    main_game.snake.direction = Vector2(0, 1)
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            if main_game.is_game_over:
                if main_game.restart_button_rect.collidepoint(event.pos):
                    main_game.restart()
        
    screen.fill((220, 255, 200)) 

    main_game.draw_ele()
    pygame.display.update()
    clock.tick(60)

