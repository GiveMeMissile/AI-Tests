import torch
import pygame
import random
import os
import sys
from torch import nn

# Game constants
WINDOW_X, WINDOW_Y = 1500, 750
TIME_LIMIT = 60000
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)

# Glue constants
GLUE_DIM = 75
GLUES = 5
GLUE_MOVEMENT_TIME = 5000

# Other object contsants (player + AI objects)
ACCELERATION = 1
FRICTION = 0.5
MAX_VELOCITY = 15
DAMAGE_COOLDOWN = 1000
PLAYER_DIM = 50
RANDOM_MOVE = False
REMOVE_OBJ_TIME = 250

# AI Constants
NUM_LAYERS = 4
INPUT_SHAPE = 8 + 2 * GLUES
HIDDEN_SIZE = 128
OUTPUT_SIZE = 4
SAVE_FOLDER = "CL_Linear_Models"
TEXT_FILE = SAVE_FOLDER + "/" +"model_numbers.txt"
LEARNING_RATE = 0.00001
NUM_AI_OBJECTS = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
window = pygame.display.set_mode((WINDOW_X, WINDOW_Y))
previous_time = 0
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
        self.amplifier = 1

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
        # Randomly moves the object in a random direction. This is used for testing purposes.
        self.dx += random.randint(-1, 1) * ACCELERATION * self.amplifier
        self.dy += random.randint(-1, 1) * ACCELERATION * self.amplifier
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

        if self.dx < FRICTION and self.dx > -FRICTION:
            self.dx = 0
        if self.dy < FRICTION and self.dy > -FRICTION:
            self.dy = 0
    
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
        self.amplifier = 10
        self.movement_timer = 0

    def alter_velocity(self, dx, dy):
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
        glue_value = glue_value/(((self.width/2 + obj.width/2) + (self.height/2 + obj.height/2))/10) + 1.5

        if (obj.dx < 0 and obj.dy < 0) and (obj_x < glue_x and obj_y < glue_y):
            return 0
        if (obj.dx > 0 and obj.dy < 0) and (obj_x > glue_x and obj_y < glue_y):
            return 0
        if obj.dx > 0 and obj_x > glue_x:
            glue_value /= 5
        if obj.dx < 0 and obj_x < glue_x:
            glue_value /= 5
        if obj.dy > 0 and obj_y > glue_y:
            glue_value /= 5
        if obj.dy < 0 and obj_y < glue_y:
            glue_value /= 5

        return glue_value

    def check_for_collisions(self, objects, current_time):
        for obj in objects:
            # Managing collisions with da glue.
            if self.hitbox.colliderect(obj.hitbox):

                if self.dx == 0 and self.dy == 0:
                    obj.dx, obj.dy = self.alter_velocity(obj.dx, obj.dy)
                else:
                    obj.dx += (self.glue_drag * self.dx)/5
                    obj.dy += (self.glue_drag * self.dy)/5
                
                if isinstance(obj, AI):
                    obj.glue_value = self.calculate_glue_value(obj)
                
                if not isinstance(obj, Player):
                    continue
                obj.contacted_object = True
                obj.objects.append(self)
                obj.remove_objects_timer = current_time

                # Damages the player for colliding with glue.
                if current_time - self.timer >= DAMAGE_COOLDOWN:
                    obj.health -= 1
                    self.timer = current_time

                          
class Player(Object):
    def __init__(self, x, y, width, height, window, color):
        super().__init__(x, y, width, height, window, color)
        self.override = not RANDOM_MOVE
        self.health = 30
        self.contacted_object = False
        self.objects = []
        self.remove_objects_timer = 0

    def player_move(self, current_time):
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

        # If the player is controlling the square as known by self.override equaling true. 
        # Then we simply move the object and check the boundries then return ending the function.
        if self.override:
            self.move()
            self.check_bounds()
            return

        # If self.override = False and the player square has not collided with any objects.
        # Then the square moves randomly via the random_move() function
        if not self.contacted_object:
            self.random_move()
            return
        
        # If there is a collision with 1 or more objects then the player object will accelerate away from the object(s) which collided with it.
        # This allows for better automated movement of the player square when it is not being controlled by the player.
        avg_x, avg_y = self.get_objects_average()
        x, y = self.location()
        if avg_x < x:
            self.dx += ACCELERATION
        else:
            self.dx -= ACCELERATION

        if avg_y < y:
            self.dy += ACCELERATION
        else:
            self.dy -= ACCELERATION

        # This if statement below will make the player square continue to move away from the contacted objects
        # for 1/4 of a second so the player square will make some distance between the object it collided with and itself.
        if current_time - REMOVE_OBJ_TIME >= self.remove_objects_timer:
            self.contacted_object = False
            self.objects.clear()

        # Calling the move() and check_bounds() functions if random_move() is not called.
        self.move()
        self.check_bounds()


    def get_objects_average(self):
        # This function returns the average x and y coords of the center of all collided objects.
        # This will be utilized to decide the direction the player will move when it is collided with multiple objects.

        average_obj_x = 0
        average_obj_y = 0
        for obj in self.objects:
            x, y = obj.location()
            average_obj_x += x
            average_obj_y += y

        return average_obj_x/len(self.objects), average_obj_y/len(self.objects)


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
            self.dx,
            self.dy,
            x, 
            y,
            self.player.dx,
            self.player.dy
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
        self.glue_value = 0
            

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

        if self.hitbox.colliderect(self.player.hitbox):
            self.player.contacted_object = True
            self.player.objects.append(self)
            self.player.remove_objects_timer = current_time
        else:
            return
        
        # check for collision between the AI and player and removes 2 health from the player if they collide.
        if (current_time - self.timer >= DAMAGE_COOLDOWN):
            old_player_dx, old_player_dy = self.player.dx, self.player.dy
            self.player.dx = self.dx/2
            self.player.dy = self.dy/2
            self.dx = old_player_dx/2
            self.dy = old_player_dy/2
            self.timer = current_time
            self.player.health -= 2

    def moving_into_wall(self, x_axis):
        # Checks if the AI is moving into a wall. If it is, it returns True.

        if x_axis:
            if self.dx > 0 and self.hitbox.x + self.width >= WINDOW_X - 5:
                return True
            elif self.dx < 0 and self.hitbox.x <= 5:
                return True
        else:
            if self.dy > 0 and self.hitbox.y + self.height >= WINDOW_Y - 5:
                return True
            elif self.dy < 0 and self.hitbox.y <= 5:
                return True
        
        return False


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
        
        # Convert positions to tensors FIRST (maintaining gradients if needed)
        ai_x, ai_y = ai.get_center()
        player_x, player_y = player.get_center()

        ai_pos = torch.tensor([ai_x, ai_y], dtype=torch.float32, device=output.device)
        player_pos = torch.tensor([player_x, player_y], dtype=torch.float32, device=output.device)
        
        # Calculate distances using tensor operations
        distance_x = torch.abs(ai_pos[0] - player_pos[0])
        distance_y = torch.abs(ai_pos[1] - player_pos[1])

        loss_x = distance_x / WINDOW_X + torch.pow(torch.tensor(1.01, device=output.device), distance_x / 10)
        loss_y = distance_y / WINDOW_Y + torch.pow(torch.tensor(1.01, device=output.device), distance_y / 10)
        loss = (loss_x + loss_y) / 2

        # Apply Glue penalty
        if glue_value > 0:
            glue_tensor = torch.tensor(glue_value, dtype=torch.float32, device=output.device)
            loss = loss * glue_tensor
            proper_direction_x = False
            proper_direction_y = False

        # Reward the AI for moving in the proper direction towards the player.
        if proper_direction_x:
            loss = loss * 0.5
        if proper_direction_y:
            loss = loss * 0.5

        # Add penalty for the AI moving into a wall.
        if ai.moving_into_wall(True):
            loss = loss * 2
        if ai.moving_into_wall(False):
            loss = loss * 2
        
        output_penalty = torch.mean(output**2) * 0.01
        loss = loss + output_penalty
        
        # Final transformation
        loss = 2 * loss**2
        
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


def check_for_folder():
    if not os.path.isdir(SAVE_FOLDER):
        os.mkdir(SAVE_FOLDER)
    if not os.path.isfile(TEXT_FILE):
        open(TEXT_FILE, "x")


def load_model(model):
    # This function checks if one of the saved models can be loaded, and if it can the function will load the model.
    # If not the function will create a new model and save its input shape so it can be saved and used in the future.

    # Check if model exists
    with open(TEXT_FILE, "r") as f:
        lines = f.readlines()
        for line in lines:
            if (int(line) == INPUT_SHAPE):
                model.load_state_dict(torch.load(SAVE_FOLDER + "/" + "model" + "_" + str(INPUT_SHAPE) + ".pth"))
                print(f"Loaded model_{INPUT_SHAPE} successfully.")
                return model
    
    # If model does not exist save the new model to the txt file.
    with open(TEXT_FILE, "a") as f:
        f.write(f"{INPUT_SHAPE}\n")
        print(f"model_{INPUT_SHAPE} does not exist and thus cannot be found. Now creating a new model")
        return model


def save_model(model):
    # Saves the model 
    torch.save(model.state_dict(), SAVE_FOLDER + "/" + "model_" + str(INPUT_SHAPE) + ".pth")
    print(f"Model {INPUT_SHAPE} saved successfully.")


def game_end(model):
    global previous_time

    try:
        save_model(model)
        previous_time = pygame.time.get_ticks()
        main()
    except RecursionError:
        print("RecursionError: Too many recursions. Program will now exit.")


def draw_game(objects, glues, time):
    # Draws the game on the window. Quite self explanatory.
    window.fill(BLACK)

    for obj in objects:
        obj.display()
    for glue in glues:
        glue.display()

    try: 
        text = font.render(f"Health: {objects[0].health}    |   Time Remaining: {round((TIME_LIMIT - time)/1000)}", True, WHITE)
    except Exception:
        text = font.render("Health: 0", True, WHITE)
    window.blit(text, (10, 10))
    pygame.display.flip()


def main():
    sys.setrecursionlimit(100000)
    running = True
    clock = pygame.time.Clock()
    player = Player(WINDOW_X/2-PLAYER_DIM/2, WINDOW_Y/2-PLAYER_DIM/2, PLAYER_DIM, PLAYER_DIM, window, WHITE)
    objects = []
    objects.append(player)
    glues = []
    for _ in range(GLUES):
        glue = Glue(random.randint(0, WINDOW_X-GLUE_DIM), random.randint(0, WINDOW_Y-GLUE_DIM), GLUE_DIM, GLUE_DIM, window, YELLOW)
        glues.append(glue)
    model = SimpleNeuralNetwork(NUM_LAYERS, INPUT_SHAPE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)
    model = load_model(model)

    for _ in range(NUM_AI_OBJECTS):
        ai = AI(random.randint(0, WINDOW_X-PLAYER_DIM), random.randint(0, WINDOW_Y-PLAYER_DIM), PLAYER_DIM, PLAYER_DIM, window, RED, model, player)
        objects.append(ai)

    # Game loop, YIPPEEEEEEE
    while running:
        current_time = pygame.time.get_ticks()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Reset the game if the space key is pressed.
                    game_end(model)
                    running = False
                elif event.key == pygame.K_LSHIFT:
                    player.override = not player.override

        if current_time-previous_time >= TIME_LIMIT:
            game_end(model)
            running = False

        for glue in glues:
            glue.check_for_collisions(objects, current_time)
            glue.apply_friction()
            if (current_time - glue.movement_timer >= GLUE_MOVEMENT_TIME):
                glue.random_move()
                glue.movement_timer = current_time
            else:
                glue.move()
                glue.check_bounds()

        for obj in objects:
            if isinstance(obj, Player):
                obj.player_move(current_time)
                if obj.health <= 0:
                    game_end(model)
                    running = False
            if isinstance(obj, AI):
                obj.ai_move(glues)
                obj.check_for_collisions(current_time)
            obj.in_glue = False
        draw_game(objects, glues, current_time-previous_time)
        clock.tick(60)


    save_model(model)
    pygame.quit()


if __name__ == "__main__":
    check_for_folder()
    main()
