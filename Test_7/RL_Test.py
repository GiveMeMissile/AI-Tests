import torch
import pygame
import random
from torch import nn


WINDOW_X, WINDOW_Y = 1500, 750
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)

GLUE_DIM = 75
GLUES = 5

ACCELERATION = 1
FRICTION = 0.5
MAX_VELOCITY = 15
DAMAGE_COOLDOWN = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
window = pygame.display.set_mode((WINDOW_X, WINDOW_Y))
pygame.init()
pygame.font.init()
font = pygame.font.SysFont("New Roman", 30)


class Object:
    # Base class for all objects in the game. It has a hitbox, velocity, and color.

    def __init__(self, x, y, width, height, window, color):
        self.width = width
        self.height = height
        self.hitbox = pygame.Rect(x, y, width, height)
        self.dx = 0
        self.dy = 0
        self.color = color
        self.window = window

    def display(self):
        pygame.draw.rect(self.window, self.color, self.hitbox)

    def move(self):
        self.hitbox.x += self.dx
        self.hitbox.y += self.dy

    def location(self):
        return self.hitbox.x, self.hitbox.y
    
    def check_bounds(self):
    # Checks for collisions with the boarders of the screen and it inverts the velocity. No going of the screen, Tehe.

        if self.hitbox.x < 0:
            self.hitbox.x -= self.dx
            self.dx = -self.dx/2
        elif self.hitbox.x > WINDOW_X - self.width:
            self.hitbox.x -= self.dx
            self.dx = -self.dx/2

        if self.hitbox.y < 0:
            self.hitbox.y -= self.dy
            self.dy = -self.dy/2
        elif self.hitbox.y > WINDOW_Y - self.height:
            self.hitbox.y -= self.dy
            self.dy = -self.dy/2


class Glue(Object):
    # Simple obstacle that is an issue for both the player and AI.

    def __init__(self, x, y, width, height, window, color):
        super().__init__(x, y, width, height, window, color)
        self.timer = 0

    def check_for_collisions(self, objects, current_time):
        for obj in objects:
            # Managing collisions with da glue.
            if self.hitbox.colliderect(obj.hitbox):
                obj.dx = obj.dx/2
                obj.dy = obj.dy/2
                if not isinstance(obj, Player):
                    continue

                # Damages the player for colliding with glue.
                if current_time - self.timer >= DAMAGE_COOLDOWN:
                    obj.health -= 1
                    self.timer = current_time

                          
class Player(Object):
    def __init__(self, x, y, width, height, window, color):
        super().__init__(x, y, width, height, window, color)
        self.health = 10

    def player_move(self):
        # Player movement using WASD keys. The player can move in all directions and has a maximum velocity.

        keys = pygame.key.get_pressed()
        # Checking for key clicks and adding the proper acceleration to the velocity.
        if keys[pygame.K_w]:
            self.dy -= ACCELERATION
        if keys[pygame.K_s]:
            self.dy += ACCELERATION
        if keys[pygame.K_a]:
            self.dx -= ACCELERATION
        if keys[pygame.K_d]:
            self.dx += ACCELERATION

        # Applying fiction to the player's velocity.
        if (self.dx > 0):
            self.dx -= FRICTION
        elif (self.dx < 0):
            self.dx += FRICTION
        if (self.dy > 0):
            self.dy -= FRICTION
        elif (self.dy < 0):
            self.dy += FRICTION

        # Checking for max velocity and setting the velocity to the max velocity if it is greater than the max velocity.
        if self.dx > MAX_VELOCITY:
            self.dx = MAX_VELOCITY
        elif self.dx < -MAX_VELOCITY:
            self.dx = -MAX_VELOCITY
        if self.dy > MAX_VELOCITY:
            self.dy = MAX_VELOCITY
        elif self.dy < -MAX_VELOCITY:
            self.dy = -MAX_VELOCITY

        # And finally moving the player and checking for bounds.
        self.move()
        self.check_bounds()


class AI(Object):
    # This is the object which the neural network will control. It is the AI which will hunt the player.

    def __init__(self, x, y, width, height, window, color, model, player):
        super().__init__(x, y, width, height, window, color)
        self.model = model
        self.player = player

    def calculate_loss(self):
        pass

    def ai_move(self):
        pass

    def train_ai(self):
        pass


class SimpleNeuralNetwork(nn.Module):
    # A simple neural network which will control an object which will hunt the player.

    def __init__(self, num_layers, input_size, hidden_size, output_size):
        super(SimpleNeuralNetwork, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = []
        for _ in range(num_layers):
            layer = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU()
            )
            self.hidden_layers.append(layer)

        self.output_layer = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input tensor idea: [aix, aiy, dx, dy, player_x, player_y, glue_x1, glue_y1, glue_x2, glue_y2, ...]
        # where dx and dy are the player's velocity, player_x and player_y are the player's position,

        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        x = self.sigmoid(x)

        # Output tensor idea: [up, down, left, right]
        # where up, down, left, and right are the AI's actions to move. Each variable will be between 0 and 1. 
        return x


def draw_game(objects, glues):
    # Draws the game on the window. Quite self explanatory.
    window.fill(BLACK)

    try: 
        text = font.render(f"Health: {objects[0].health}", True, WHITE)
    except Exception:
        text = font.render("Health: 0", True, WHITE)
    window.blit(text, (10, 10))

    for obj in objects:
        obj.display()
    for glue in glues:
        glue.display()
    pygame.display.flip()


def main():
    running = True
    clock = pygame.time.Clock()
    player = Player(100, 100, 50, 50, window, WHITE)
    objects = []
    objects.append(player)
    glues = []
    for _ in range(GLUES):
        glue = Glue(random.randint(0, WINDOW_X-GLUE_DIM), random.randint(0, WINDOW_Y-GLUE_DIM), GLUE_DIM, GLUE_DIM, window, YELLOW)
        glues.append(glue)

    # Game loop, YIPPEEEEEEE
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                running = False
        try:
            objects[0].player_move()
            if objects[0].health <= 0:
                print("Player is dead")
                del objects[0]
        except Exception:
            pass
        for glue in glues:
            glue.check_for_collisions(objects, pygame.time.get_ticks())
        draw_game(objects, glues)
        clock.tick(60)


if __name__ == "__main__":
    main()
