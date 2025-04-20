import torch
import pygame
import random
from torch import nn


WINDOW_X, WINDOW_Y = 1500, 750
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)

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
                if not isinstance(obj, Player) and not isinstance(obj, AI):
                    continue
                obj.in_glue = True
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
        self.in_glue = False

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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00001)
        self.in_glue = False
        self.timer = 0

    def calculate_loss(self):
        loss_x = abs(self.player.hitbox.x - self.hitbox.x)/WINDOW_X
        loss_y = abs(self.player.hitbox.y - self.hitbox.y)/WINDOW_Y
        loss = (loss_x + loss_y)/2
        if self.in_glue:
            loss *= 2
        if self.player.in_glue:
            loss /= 2
        return loss

    def ai_move(self, glues):
        # Prepare the input tensors to be fed into the neural network.
        X = torch.tensor([
            self.hitbox.x, 
            self.hitbox.y,
            self.player.hitbox.x, 
            self.player.hitbox.y
            ], dtype=torch.float32).to(device)
        
        list_glues_location = []
        for glue in glues:
            list_glues_location.append(glue.hitbox.x)
            list_glues_location.append(glue.hitbox.y)

        glue_tensor = torch.tensor(list_glues_location, dtype=torch.float32).to(device)
        X = torch.cat((X, glue_tensor), dim=0)
        
        X = X.to(device)

        # Feed the input tensor into the neural network and get the output tensor then process the data to get the direction of the AI's acceleration.
        # output is as a vector containung 4 values that are from 0-1. With each value representing a direction: [up, down, left, right].
        original_output = self.model(X)
        output = torch.sigmoid(original_output).to("cpu")
        output = (output > 0.5)
        print(output)
        if (output[0]):
            self.dy -= ACCELERATION
        if (output[1]):
            self.dy += ACCELERATION
        if (output[2]):
            self.dx -= ACCELERATION
        if (output[3]):
            self.dx += ACCELERATION

        # Applying fiction to the AI's velocity.
        if (self.dx > 0):
            self.dx -= FRICTION
        elif (self.dx < 0):
            self.dx += FRICTION
        if (self.dy > 0):
            self.dy -= FRICTION
        elif (self.dy < 0):
            self.dy += FRICTION

        # Cheing for max velocity and setting the velocity to the max velocity if it is greater than the max velocity.
        if self.dx > MAX_VELOCITY:
            self.dx = MAX_VELOCITY
        elif self.dx < -MAX_VELOCITY:
            self.dx = -MAX_VELOCITY
        if self.dy > MAX_VELOCITY:
            self.dy = MAX_VELOCITY
        elif self.dy < -MAX_VELOCITY:
            self.dy = -MAX_VELOCITY
        
        self.move()
        self.check_bounds()
        self.train_ai(original_output)
            

    def train_ai(self, output):
        # Trains the AI using the custom loss function. The loss function is based on the distance between the AI and the player.
        # The loss function is used to update the weights of the neural network.

        self.optimizer.zero_grad()
        loss = self.model.calculate_loss(self.player, self.hitbox, self.in_glue, output)
        loss.backward()
        self.optimizer.step()

    def check_for_collisions(self, current_time):
        # check for collision between the AI and player and removes 5 health from the player if they collide.
        if self.hitbox.colliderect(self.player.hitbox) and (current_time - self.timer >= DAMAGE_COOLDOWN):
            self.player.dx = self.dx/2
            self.player.dy = self.dy/2
            self.dx = self.player.dx/2
            self.dy = self.player.dy/2
            self.timer = current_time
            self.player.health -= 5


class SimpleNeuralNetwork(nn.Module):
    # A simple neural network which will control an object which will hunt the player.

    def __init__(self, num_layers, input_size, hidden_size, output_size):
        super(SimpleNeuralNetwork, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU()
            ) for _ in range(num_layers)
        ])

        self.output_layer = nn.Linear(hidden_size, output_size)

    def calculate_loss(self, player, hitbox, in_glue, output):
        print(output)
        loss_x = abs(player.hitbox.x - hitbox.x)/WINDOW_X
        loss_y = abs(player.hitbox.y - hitbox.y)/WINDOW_Y
        loss = (loss_x + loss_y)/2
        if in_glue:
            loss *= 2
        if player.in_glue:
            loss /= 2
        loss = torch.tensor(loss, dtype=torch.float32, requires_grad=True).to(device)
        loss = torch.mean((abs((abs(output) + (loss*100))/100)**2)*2)
        print(loss)
        return loss

    def forward(self, x):
        # Input tensor idea: [aix, aiy, dx, dy, player_x, player_y, glue_x1, glue_y1, glue_x2, glue_y2, ...]
        # where dx and dy are the player's velocity, player_x and player_y are the player's position,

        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)

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
    player = Player(WINDOW_X/2-25, WINDOW_Y/2-25, 50, 50, window, WHITE)
    objects = []
    objects.append(player)
    glues = []
    for _ in range(GLUES):
        glue = Glue(random.randint(0, WINDOW_X-GLUE_DIM), random.randint(0, WINDOW_Y-GLUE_DIM), GLUE_DIM, GLUE_DIM, window, YELLOW)
        glues.append(glue)
    model = SimpleNeuralNetwork(2, 4+2*GLUES, 64, 4).to(device)
    for _ in range(6):
        ai = AI(random.randint(0, WINDOW_X-50), random.randint(0, WINDOW_Y-50), 50, 50, window, RED, model, player)
        objects.append(ai)

    # Game loop, YIPPEEEEEEE
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Reset the game if the space key is pressed.
                    main()

        for obj in objects:
            obj.in_glue = False
            if isinstance(obj, Player):
                obj.player_move()
                if obj.health <= 0:
                    main()
            if isinstance(obj, AI):
                obj.ai_move(glues)
                obj.check_for_collisions(pygame.time.get_ticks())
        for glue in glues:
            glue.check_for_collisions(objects, pygame.time.get_ticks())
        draw_game(objects, glues)
        clock.tick(60)


if __name__ == "__main__":
    main()
