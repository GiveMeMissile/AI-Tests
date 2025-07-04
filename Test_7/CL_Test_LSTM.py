# Todo: 
# 1: Reorganize code using the new AIManager class
# 2: Add the AIs being able to know about each other and the LSTM putting a single output for all AI objects rather than how it is currently.
# 3: Change the saving method to be compatible with the changed made in #2. 
# 4: Get happy.

import torch
import sys
import os
import pygame
import random
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
GLUES = 15
GLUE_MOVEMENT_TIME = 5000

# Other object constants (player + AI objects)
ACCELERATION = 1
FRICTION = 0.5
MAX_VELOCITY = 15
DAMAGE_COOLDOWN = 1000
PLAYER_DIM = 50
RANDOM_MOVE = True
REMOVE_OBJ_TIME = 250

# AI Constants
NUM_LAYERS = 4
HIDDEN_SIZE = 128
OUTPUT_SIZE = 4
SAVE_FOLDER = "CL_LSTM_Models"
TEXT_FILE = SAVE_FOLDER + "/" +"model_numbers.txt"
LEARNING_RATE = 0.0001
AI_FORWARD_TIME = 1000/40 # The AI object will change its directional vector 15 times each second thanks to this variable, this will be used for testing later
NUM_AI_OBJECTS = 3
NUM_SAVED_FRAMES = 60
SEQUENCE_LENGTH = 8 + 2 * GLUES
INPUT_SHAPE = (NUM_SAVED_FRAMES, SEQUENCE_LENGTH)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
window = pygame.display.set_mode((WINDOW_X, WINDOW_Y))
previous_time = 0
pygame.init()
pygame.font.init()
font = pygame.font.SysFont("New Roman", 30)


class Object:
    # Base class for all objects in the game. It has a hitbox, velocity, and color.

    def __init__(self, width, height, window, color, objects, distance, x=None, y=None):
        self.width = width
        self.height = height
        self.dx = 0
        self.dy = 0
        self.color = color
        self.window = window
        self.amplifier = 1
        if x is None and y is None:
            x, y = self.find_valid_location(objects, distance)
        elif x is None:
            x, _ = self.find_valid_location(objects, distance)
        elif y is None:
            _, y = self.find_valid_location(objects, distance)
        self.hitbox = pygame.Rect(x, y, width, height)

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
            self.hitbox.x = 0
            self.dx = -self.dx/2
        elif self.hitbox.x > WINDOW_X - self.width:
            self.hitbox.x = WINDOW_X - self.width
            self.dx = -self.dx/2

        if self.hitbox.y < 0:
            self.hitbox.y = 0
            self.dy = -self.dy/2
        elif self.hitbox.y > WINDOW_Y - self.height:
            self.hitbox.y = WINDOW_Y - self.height
            self.dy = -self.dy/2

        if self.hitbox.x < -25:
            self.hitbox.x += 100
        elif self.hitbox.x > WINDOW_X + 25:
            self.hitbox.x -= 100

        if self.hitbox.y < -25:
            self.hitbox.y += 100
        elif self.hitbox.y > WINDOW_Y + 25:
            self.hitbox.y -= 100

    def random_move(self):
        # Randomly moves the object in a random direction. This is used for testing purposes.
        self.dx += random.randint(-1, 1) * ACCELERATION * self.amplifier
        self.dy += random.randint(-1, 1) * ACCELERATION * self.amplifier
        self.move()
        self.check_bounds()

    def apply_friction(self):

        if self.dx > 0:
            self.dx -= FRICTION
        elif self.dx < 0:
            self.dx += FRICTION
        if self.dy > 0:
            self.dy -= FRICTION
        elif self.dy < 0:
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
    
    def find_valid_location(self, objects, distance):
        # This function finds a random location where the object is not on top of another object.
        x, y = 0, 0
        not_valid = True
        while not_valid:
            x = random.randint(0, WINDOW_X - int(self.width/2))
            y = random.randint(0, WINDOW_Y - int(self.height/2))
            for obj in objects:
                obj_x, obj_y = obj.get_center()
                if (x > obj_x - distance) and (x < obj_x + distance):
                    break
                elif (y > obj_y - distance) and (y < obj_y + distance):
                    break
                else:
                    not_valid = False

        return x - self.width/2, y - self.height/2


class Glue(Object):
    # Simple obstacle that is an issue for both the player and AI.

    def __init__(self, width, height, window, color, objects, distance, x=None, y=None):
        super().__init__(width, height, window, color, objects, distance, x, y)
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
    def __init__(self, width, height, window, color, objects, distance, x=None, y=None):
        super().__init__(width, height, window, color, objects, distance, x, y)
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

    def __init__(self, width, height, window, color, objects, distance, x=None, y=None):
        super().__init__(width, height, window, color, objects, distance, x, y)
        self.glue_value = 0
        self.timer = 0
        self.h0 = None
        self.c0 = None
        self.change_direction_timer = 0
        self.memory = torch.zeros((NUM_SAVED_FRAMES, SEQUENCE_LENGTH), dtype=torch.float32) # AM

    def add_frame_to_memory(self, frame): # AM
        self.memory = self.memory.to("cpu")

        for i in range(NUM_SAVED_FRAMES):
            if ((self.memory[NUM_SAVED_FRAMES - (i+1)] == torch.zeros(SEQUENCE_LENGTH, dtype=torch.float32)).sum().item() == SEQUENCE_LENGTH):
                self.memory[NUM_SAVED_FRAMES - (i+1)] = frame
                return
        
        self.memory = torch.cat((self.memory[1 : NUM_SAVED_FRAMES], frame), dim=0)
        self.memory = self.memory.to(device)

    def ai_move(self, directional_vector):
        if (directional_vector[0]):
            self.dy -= ACCELERATION
        if (directional_vector[1]):
            self.dy += ACCELERATION
        if (directional_vector[2]):
            self.dx -= ACCELERATION
        if (directional_vector[3]):
            self.dx += ACCELERATION

        self.check_bounds()
        self.apply_friction()
        self.check_max_velocity()        
        self.move()

    def check_for_collisions(self, current_time, player):
        
        if self.hitbox.colliderect(player.hitbox):
            player.contacted_object = True
            player.objects.append(self)
            player.remove_objects_timer = current_time
        else:
            return
        
        # check for collision between the AI and player and removes 2 health from the player if they collide.
        if (current_time - self.timer >= DAMAGE_COOLDOWN):
            old_player_dx, old_player_dy = player.dx, player.dy
            player.dx = self.dx/2
            player.dy = self.dy/2
            self.dx = old_player_dx/2
            self.dy = old_player_dy/2
            self.timer = current_time
            player.health -= 2

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


class LSTM(nn.Module):
    # We will be using an LSTM model in this experiment. This LSTM will control the AI objects.

    def __init__(self, num_layers, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0, c0):
        # Input tensor ideas: sequence = [aix, aiy, dx, dy, player_x, player_y, glue_x1, glue_y1, glue_x2, glue_y2, ...]
        # where dx and dy are the player's velocity, player_x and player_y are the player's position,
        # Input_Tensor = [NUM_SAVED_FRAMES, sequence length]. Where there will be NUM_SAVED_FRAMES amount of sequances in the tensor.
        # The sequences include the current frame plus the last NUM_SAVED_FRAMES-1 frames.

        if (h0 is None):
            h0 = torch.zeros(self.num_layers, self.hidden_size).to(device)
        if (c0 is None):
            c0 = torch.zeros(self.num_layers, self.hidden_size).to(device)
        
        x, (h0, c0) = self.lstm(x, (h0, c0))
        x = self.output_layer(x[-1])
        
        h0 = h0.detach()
        c0 = c0.detach()

        # Output tensor idea: [up, down, left, right]
        # where up, down, left, and right are the AI's actions to move. Each variable will be between 0 and 1. 
        return x, h0, c0
    

class AIManager:
    def __init__(self, num_ais, glues, player):
        self.model = LSTM(NUM_LAYERS, SEQUENCE_LENGTH, HIDDEN_SIZE, OUTPUT_SIZE).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.glues = glues
        self.player = player
        self.ai_list = self.create_ais(num_ais)

    def create_ais(self, num_ais):
        ais = []
        for _ in range(num_ais):
            ais.append(AI(
                PLAYER_DIM, 
                PLAYER_DIM, 
                window, 
                RED,
                [self.player] + self.glues,
                (PLAYER_DIM+GLUE_DIM)/2
                ))
        return ais
    
    def create_tensor(self, ai):
        x, y = self.player.get_center()
        x_ai, y_ai = ai.get_center()
        X = torch.tensor([
            x_ai, 
            y_ai,
            ai.dx,
            ai.dy,
            x, 
            y,
            self.player.dx,
            self.player.dy
            ], dtype=torch.float32).to(device)
        
        list_glues_location = []
        for glue in self.glues:
            x_glue, y_glue = glue.get_center()
            list_glues_location.append(x_glue/WINDOW_X)
            list_glues_location.append(y_glue/WINDOW_Y)

        glue_tensor = torch.tensor(list_glues_location, dtype=torch.float32).to(device)
        X = torch.cat((X, glue_tensor), dim=0)
        frame = X.unsqueeze(0).to("cpu")
        ai.add_frame_to_memory(frame)
    
    def move_ais(self):
        for ai in self.ai_list:
            self.create_tensor(ai)
            logits, ai.h0, ai.c0 = self.model(ai.memory.to(device), ai.h0, ai.c0)
            logits = logits.squeeze(0)
            output = torch.sigmoid(logits)
            output = (output >= 0.5)
            ai.ai_move(output)
            self.train_ai(ai, logits)

    def train_ai(self, ai, output): # AM
        # Trains the AI using the custom loss function. The loss function is based on the distance between the AI and the player.
        # The loss function is used to update the weights of the neural network.

        ai_x, ai_y = ai.get_center()
        player_x, player_y = self.player.get_center()

        proper_dx, proper_dy = False, False
        if ai.dx > 0 and player_x > ai_x:
            proper_dx = True
        elif ai.dx < 0 and player_x < ai_x:
            proper_dx = True
        elif ai.dx == 0 and (player_x + PLAYER_DIM/2 > ai_x and player_x - PLAYER_DIM/2 < ai_x):
            proper_dx = True

        if ai.dy > 0 and player_y > ai_y:
            proper_dy = True
        elif ai.dy < 0 and player_y < ai_y:
            proper_dy = True
        elif ai.dy == 0 and (player_y + PLAYER_DIM/2 > ai_y and player_y - PLAYER_DIM/2 < ai_y):
            proper_dy = True

        self.optimizer.zero_grad()
        loss = self.calculate_loss(ai, ai.glue_value, output, proper_dx, proper_dy)
        loss.backward()
        self.optimizer.step()

    def calculate_loss(self, ai, glue_value, output, proper_direction_x, proper_direction_y): # AM
        
        # Convert positions to tensors FIRST (maintaining gradients if needed)
        ai_x, ai_y = ai.get_center()
        player_x, player_y = self.player.get_center()

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

    def load_model(self):
        # This function checks if one of the saved models can be loaded, and if it can the function will load the model.
        # If not the function will create a new model and save its input shape so it can be saved and used in the future.

        # Check if model exists
        with open(TEXT_FILE, "r") as f:
            lines = f.readlines()
            for line in lines:
                split = line.split()
                if (int(split[0]) == INPUT_SHAPE[0]) and (int(split[1]) == INPUT_SHAPE[1]):
                    self.model.load_state_dict(torch.load(SAVE_FOLDER + "/" + "model" + "_" + split[0] + "_" + split[1] + ".pth"))
                    print(f"Loaded model_{INPUT_SHAPE} successfully.")
                    return self.model
        
        # If model does not exist save the new model to the txt file.
        with open(TEXT_FILE, "a") as f:
            f.write(f"{INPUT_SHAPE[0]} {INPUT_SHAPE[1]}\n")
            print(f"model_{INPUT_SHAPE} does not exist and thus cannot be found. Now creating a new model")
            return self.model
        
    def save_model(self):
        # Saves the model 
        torch.save(self.model.state_dict(), SAVE_FOLDER + "/" + "model_" + str(INPUT_SHAPE[0]) + "_" + str(INPUT_SHAPE[1]) + ".pth")
        print(f"Model {INPUT_SHAPE} saved successfully.")


def check_for_folder():
    if not os.path.isdir(SAVE_FOLDER):
        os.mkdir(SAVE_FOLDER)
    if not os.path.isfile(TEXT_FILE):
        open(TEXT_FILE, "x")


def game_end(ai_manager):
    global previous_time

    try:
        ai_manager.save_model()
        previous_time = pygame.time.get_ticks()
        main()
    except RecursionError:
        print("RecursionError: Too many recursions. Program will now exit.")


def draw_game(player, glues, time, ai_manager):
    # Draws the game on the window. Quite self explanatory.
    window.fill(BLACK)

    player.display()

    for ai in ai_manager.ai_list:
        ai.display()

    for glue in glues:
        glue.display()

    try: 
        text = font.render(f"Health: {player.health}    |    Time Remaining: {round((TIME_LIMIT - time)/1000)}", True, WHITE)
    except Exception:
        text = font.render("Health: 0", True, WHITE)
    window.blit(text, (10, 10))
    pygame.display.flip()


def main():
    sys.setrecursionlimit(100000)
    running = True
    clock = pygame.time.Clock()
    player = Player(PLAYER_DIM, PLAYER_DIM, window, WHITE, objects=[], distance=None, x=WINDOW_X/2-PLAYER_DIM/2, y=WINDOW_Y/2-PLAYER_DIM/2)
    glues = []
    for _ in range(GLUES):
        glue = Glue(GLUE_DIM, GLUE_DIM, window, YELLOW, [player]+glues, distance=GLUE_DIM)
        glues.append(glue)
    ai_manager = AIManager(NUM_AI_OBJECTS, glues, player)

    # Game loop, YIPPEEEEEEE
    while running:
        current_time = pygame.time.get_ticks()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Reset the game if the space key is pressed.
                    game_end(ai_manager)
                    running = False
                elif event.key == pygame.K_LSHIFT:
                    player.override = not player.override

        if pygame.time.get_ticks()-previous_time >= TIME_LIMIT:
            game_end(ai_manager)
            running = False

        for glue in glues:
            glue.check_for_collisions(ai_manager.ai_list + [player], current_time)
            glue.check_bounds()
            glue.apply_friction()
            if (current_time - glue.movement_timer >= GLUE_MOVEMENT_TIME):
                glue.random_move()
                glue.movement_timer = current_time
            else:
                glue.move()

        ai_manager.move_ais()
        for ai in ai_manager.ai_list:
            ai.check_for_collisions(current_time, player)

        player.player_move(current_time)
        if player.health <= 0:
            game_end(ai_manager)
            running = False
        draw_game(player, glues, current_time - previous_time, ai_manager)
        clock.tick(60)


    ai_manager.save_model()
    pygame.quit()


if __name__ == "__main__":
    check_for_folder()
    main()
