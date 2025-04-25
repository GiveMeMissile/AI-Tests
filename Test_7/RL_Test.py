import torch
import pygame
import random
from torch import nn

# Game constants
WINDOW_X, WINDOW_Y = 1500, 750
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)

# Glue constants
GLUE_DIM = 75
GLUES = 10

# Other object contsants (player + AI objects)
ACCELERATION = 1
FRICTION = 0.5
MAX_VELOCITY = 15
DAMAGE_COOLDOWN = 1000
PLAYER_DIM = 50
RANDOM_MOVE = False

# AI Constants
NUM_LAYERS = 4
INPUT_SIZE = 4 + 2 * GLUES
HIDDEN_SIZE = 128
OUTPUT_SIZE = 4
SAVE_FILE = "Models/model_001.pth"
LEARNING_RATE = 0.00001

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
        self.override = not RANDOM_MOVE

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

        if self.hitbox.x < -25:
            self.hitbox.x += 100
        elif self.hitbox.x > WINDOW_X + 25:
            self.hitbox.x -= 100

        if self.hitbox.y < -25:
            self.hitbox.y += 100
        elif self.hitbox.y > WINDOW_Y + 25:
            self.hitbox.y -= 100

        if self.hitbox.y < 0:
            self.hitbox.y -= self.dy
            self.dy = -self.dy/2
        elif self.hitbox.y > WINDOW_Y - self.height:
            self.hitbox.y -= self.dy
            self.dy = -self.dy/2

    def random_move(self):
        if self.override:
            return
        # Randomly moves the object in a random direction. This is used for testing purposes.
        self.dx = random.randint(-1, 1) * ACCELERATION*2
        self.dy = random.randint(-1, 1) * ACCELERATION*2
        self.move()
        self.check_bounds()

    def apply_friction(self):
        if (self.dx > 0):
            self.dx -= FRICTION
        elif (self.dx < 0):
            self.dx += FRICTION
        if (self.dy > 0):
            self.dy -= FRICTION
        elif (self.dy < 0):
            self.dy += FRICTION
    
    def check_max_velocity(self):
        if self.dx > MAX_VELOCITY:
            self.dx = MAX_VELOCITY
        elif self.dx < -MAX_VELOCITY:
            self.dx = -MAX_VELOCITY
        if self.dy > MAX_VELOCITY:
            self.dy = MAX_VELOCITY
        elif self.dy < -MAX_VELOCITY:
            self.dy = -MAX_VELOCITY

    def get_center(self):
        # Returns the center of the object.
        return self.hitbox.x + self.width/2, self.hitbox.y + self.height/2


class Glue(Object):
    # Simple obstacle that is an issue for both the player and AI.

    def __init__(self, x, y, width, height, window, color):
        super().__init__(x, y, width, height, window, color)
        self.timer = 0
        self.glue_drag = 0.5

    def alter_collision_velocity(self, dx, dy):
        # This function is used to alter the velocity of the object when it collides with the glue.
        if dx > 0:
            dx -= self.glue_drag * dx
        elif dx < 0:
            dx -= self.glue_drag * dx
        
        if dy > 0:
            dy -= self.glue_drag * dy
        elif dy < 0:
            dy -= self.glue_drag * dy
        return dx, dy
    
    def calculate_glue_value(self, obj):
        # This function calculates the glue value which is used within the loss calculation of the AI.
        # This value increases the further the AI model is in the glue.
        obj_x, obj_y = obj.get_center()
        glue_x, glue_y = self.get_center()
        glue_value = ((self.width/2 + obj.width/2) - (abs(obj_x - glue_x))) + ((self.height + obj.height)/2 - (abs(obj_y - glue_y)))
        glue_value = glue_value/(((self.width/2 + obj.width/2) + (self.height/2 + obj.height/2))/10) + 1
        return glue_value

    def check_for_collisions(self, objects, current_time):
        for obj in objects:
            # Managing collisions with da glue.
            if self.hitbox.colliderect(obj.hitbox):

                obj.dx, obj.dy = self.alter_collision_velocity(obj.dx, obj.dy)
                
                if isinstance(obj, AI):
                    obj.glue_value = self.calculate_glue_value(obj)
                
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
            self.override = True
        if keys[pygame.K_s]:
            self.dy += ACCELERATION
            self.override = True
        if keys[pygame.K_a]:
            self.dx -= ACCELERATION
            self.override = True
        if keys[pygame.K_d]:
            self.dx += ACCELERATION
            self.override = True
        

        self.apply_friction()
        self.check_max_velocity()
        self.move()
        self.check_bounds()


class AI(Object):
    # This is the object which the neural network will control. It is the AI which will hunt the player.

    def __init__(self, x, y, width, height, window, color, model, player):
        super().__init__(x, y, width, height, window, color)
        self.model = model
        self.player = player
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.glue_value = 0
        self.timer = 0

    def ai_move(self, glues):
        # Prepare the input tensors to be fed into the neural network.
        x, y = self.player.get_center()
        x_ai, y_ai = self.get_center()
        X = torch.tensor([
            x_ai, 
            y_ai,
            x, 
            y
            ], dtype=torch.float32).to(device)
        
        list_glues_location = []
        for glue in glues:
            x_glue, y_glue = glue.get_center()
            list_glues_location.append(x_glue)
            list_glues_location.append(y_glue)

        glue_tensor = torch.tensor(list_glues_location, dtype=torch.float32).to(device)
        X = torch.cat((X, glue_tensor), dim=0)
        
        X = X.to(device)

        # Feed the input tensor into the neural network and get the output tensor then process the data to get the direction of the AI's acceleration.
        # output is as a vector containung 4 values that are from 0-1. With each value representing a direction: [up, down, left, right].
        original_output = self.model(X)
        output = torch.sigmoid(original_output).to("cpu")
        output = (output > 0.5)
        if (output[0]):
            self.dy -= ACCELERATION
        if (output[1]):
            self.dy += ACCELERATION
        if (output[2]):
            self.dx -= ACCELERATION
        if (output[3]):
            self.dx += ACCELERATION

        self.apply_friction()
        self.check_max_velocity()        
        self.move()
        self.check_bounds()
        self.train_ai(original_output)
            

    def train_ai(self, output):
        # Trains the AI using the custom loss function. The loss function is based on the distance between the AI and the player.
        # The loss function is used to update the weights of the neural network.

        ai_x, ai_y = self.get_center()
        player_x, player_y = self.player.get_center()

        proper_dx, proper_dy = False, False
        if self.dx > 0 and player_x > ai_x:
            proper_dx = True
        elif self.dx < 0 and player_x < ai_x:
            proper_dx = True
        elif self.dx == 0 and (player_x + PLAYER_DIM/2 > ai_x and player_x - PLAYER_DIM/2 < ai_x):
            proper_dx = True

        if self.dy > 0 and player_y > ai_y:
            proper_dy = True
        elif self.dy < 0 and player_y < ai_y:
            proper_dy = True
        elif self.dy == 0 and (player_y + PLAYER_DIM/2 > ai_y and player_y - PLAYER_DIM/2 < ai_y):
            proper_dy = True

        self.optimizer.zero_grad()
        loss = self.model.calculate_loss(self.player, self, self.glue_value, output, proper_dx, proper_dy)
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
            self.player.health -= 2


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

    def calculate_loss(self, player, ai, glue_value, output, proper_direction_x, proper_direction_y):
        ai_x, ai_y = ai.get_center()
        player_x, player_y = player.get_center()
        loss_x = abs(player_x - ai_x)/WINDOW_X
        loss_y = abs(player_y - ai_y)/WINDOW_Y
        loss = (loss_x + loss_y)/2
        if glue_value > 0:
            loss *= glue_value
            proper_direction_x = False
            proper_direction_y = False
        if proper_direction_x:
            loss /= 5
        if proper_direction_y:
            loss /= 5
        if proper_direction_x and proper_direction_y:
            loss = 0
        loss = torch.tensor(loss, dtype=torch.float32, requires_grad=True).to(device)
        loss = torch.mean((abs((abs(output) + (loss*100))/100)**2)*2)
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
    

    for obj in objects:
        obj.display()
    for glue in glues:
        glue.display()

    try: 
        text = font.render(f"Health: {objects[0].health}", True, WHITE)
    except Exception:
        text = font.render("Health: 0", True, WHITE)
    window.blit(text, (10, 10))
    pygame.display.flip()


def main():
    running = True
    clock = pygame.time.Clock()
    player = Player(WINDOW_X/2-PLAYER_DIM/2, WINDOW_Y/2-PLAYER_DIM/2, PLAYER_DIM, PLAYER_DIM, window, WHITE)
    objects = []
    objects.append(player)
    glues = []
    for _ in range(GLUES):
        glue = Glue(random.randint(0, WINDOW_X-GLUE_DIM), random.randint(0, WINDOW_Y-GLUE_DIM), GLUE_DIM, GLUE_DIM, window, YELLOW)
        glues.append(glue)
    model = SimpleNeuralNetwork(NUM_LAYERS, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)
    for _ in range(10):
        ai = AI(random.randint(0, WINDOW_X-PLAYER_DIM), random.randint(0, WINDOW_Y-PLAYER_DIM), PLAYER_DIM, PLAYER_DIM, window, RED, model, player)
        objects.append(ai)

    # Game loop, YIPPEEEEEEE
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Reset the game if the space key is pressed.
                    try: 
                        main()
                    except RecursionError:
                        running = False
                        print("RecursionError: Too many recursions. Program will now exit.")
                elif event.key == pygame.K_LSHIFT:
                    player.override = not player.override


        for glue in glues:
            glue.check_for_collisions(objects, pygame.time.get_ticks())

        for obj in objects:
            if isinstance(obj, Player):
                obj.player_move()
                obj.random_move()
                if obj.health <= 0:
                    try:
                        main()
                    except RecursionError:
                        running = False
                        print("RecursionError: Too many recursions. Program will now exit.")
            if isinstance(obj, AI):
                obj.ai_move(glues)
                obj.check_for_collisions(pygame.time.get_ticks())
            obj.in_glue = False
        draw_game(objects, glues)
        clock.tick(60)


    torch.save(model.state_dict(), SAVE_FILE)
    pygame.quit()


if __name__ == "__main__":
    main()
