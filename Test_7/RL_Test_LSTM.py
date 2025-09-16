# Note: Begin one of the most painful part of this dumb project (Hivemind time).
# Reorganize how actions are handled in order to be able to function with the hivemind structure of the AI.
# After the Hivemind has been created. Then I shall remake the model saving in order to work with the current changes (Step 6)

# Todo: 
# 1: Minor improvements: AI Input generalization , Removal of recursive loop , Optimzation of device managerment, and model saving fix. (Done)
# 2: Add better positive rewards for the AI such as when the AI closes the distance between the player and AI. (Done)
# 3: Change the summing all of the Q_values during training to training each AI separately. (Done)
# 4: Add the saving of H0 and C0 in AIManager. (Done)
# 5: Add AI collisions. 
# 6: Add model progress tracking with TensorBoard and/or Matplotlib.
# 7: Get happy... (Impossible)

import torch
import sys
import os
import pygame
import random
import json
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from collections import deque

# Game constants
MAX_EPISODES = 1000
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
HIDDEN_SIZE = 64
OUTPUT_SIZE = 9
LEARNING_RATE = 0.0001
NUM_AI_OBJECTS = 2
NUM_SAVED_FRAMES = 20
SEQUENCE_LENGTH = 4 + 4 * NUM_AI_OBJECTS + 2 * GLUES
INPUT_SHAPE = (NUM_SAVED_FRAMES, SEQUENCE_LENGTH)
DISCOUNT_FACTOR = 0.9
SYNC_MODEL = 30
BATCH_SIZE = 32
EPSILON_ACTIVE = True # Determines if epsilon is active
EPSILON_DECAY = 300 # How many episodes/games the ai plays until epsilon is 0.
AI_SAVE_DATA = {
    "Model": [],
    "Ais": [],
    "Glues": [],
    "Hidden": [],
    "Frames": [],
    "Layers": [],
    "Epsilon": []
}

# File Constants
SAVE_FOLDER = "RL_LSTM_Models"
INFO_FILE = SAVE_FOLDER + "/" +"model_info.json"
DATA_FOLDER = "RL_LSTM_Progress_Data"

# AI Reward values
GLOBAL_DIVIDE = 5
PLAYER_CONTACT = 1.5
GLUE_CONTACT = -.75
WALL_CONTACT = -.2
MOVING_TOWARDS_PLAYER = .3
PLAYER_NEARBY = .4
NO_MOVEMENT = -.5


# Other important stuff
device = "cuda" if torch.cuda.is_available() else "cpu"
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
                    obj.in_glue = True
                
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
        self.previous_y = None
        self.previous_x = None

    def player_move(self, current_time):
        # Player movement using WASD keys. The player can move in all directions and has a maximum velocity.

        keys = pygame.key.get_pressed()
        # Checking for key clicks and adding the proper acceleration to the velocity.

        self.previous_x, self.previous_y = self.get_center()

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
        self.in_glue = False
        self.touching_player = False
        self.timer = 0
        self.change_direction_timer = 0
        self.action = None
        self.previous_x = None
        self.previous_y = None

        # Debug variables
        self.action_count = 0
        self.total_actions = 0

    def ai_move(self, ai_output, epsilon):

        # Epsilon greedy (Might move to AIManager)
        if random.random() <= epsilon and EPSILON_ACTIVE:
            self.action = random.randint(0, 8)
        else:
            self.action = torch.argmax(ai_output)

        self.action_count += 1
        self.total_actions += self.action
        
        self.previous_x, self.previous_y = self.get_center()

        # Moving the AI using the action    
        directional_vector = self.get_directional_vector()
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

    def get_directional_vector(self):
        match self.action:
            case 0:
                return [False, False, False, False]
            case 1:
                return [True, False, False, False]
            case 2:
                return [False, True, False, False]
            case 3:
                return [False, False, True, False]
            case 4:
                return [False, False, False, True]
            case 5:
                return [True, False, True, False]
            case 6:
                return [True, False, False, True]
            case 7:
                return [False, True, True, False]
            case 8:
                return [False, True, False, True]

    def check_for_collisions(self, current_time, player):
        
        if self.hitbox.colliderect(player.hitbox):
            player.contacted_object = True
            player.objects.append(self)
            player.remove_objects_timer = current_time
            self.touching_player = True
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

    def moving_into_wall(self, axis='x'):
        # Checks if the AI is moving into a wall. If it is, it returns True.

        velocity = None
        location = None

        if axis == 'x':
            velocity = self.dx
            location = self.hitbox.x
        elif axis == 'y':
            velocity = self.dy
            location = self.hitbox.y
        else:
            return False


        if velocity > 0 and location + PLAYER_DIM >= WINDOW_Y - 10:
            return True
        elif velocity < 0 and location <= 10:
            return True
        elif location + PLAYER_DIM >= WINDOW_X - 1:
            return True
        elif location <= 1:
            return True
        
        return False
    
    def moving_towards_player(self, player, axis='x'):
        # checks if the AI is moving towards the player.

        ai_location = None
        player_location = None
        velocity = None

        if axis == 'x':
            ai_location, _ = self.get_center()
            player_location, _ = player.get_center()
            velocity = self.dx
        elif axis == 'y':
            _, ai_location = self.get_center()
            _, player_location = player.get_center()
            velocity = self.dy
        else:
            return False

        moving_to_player = False
        if velocity > 0 and player_location > ai_location:
            moving_to_player = True
        elif velocity < 0 and player_location < ai_location:
            moving_to_player = True
        
        return moving_to_player
    
    def nearby_player(self, player):
        # Checks if the AI is close to the player.

        ai_x, ai_y = self.get_center()
        player_x, player_y = player.get_center()
        if ((player_x + PLAYER_DIM*2 > ai_x and player_x - PLAYER_DIM*2 < ai_x) 
            and (player_y + PLAYER_DIM*2 > ai_y and player_y - PLAYER_DIM*2 < ai_y)):
            return True
        else:
            return False



class TrainingData:
    # This class will store some of the training data for future training WOHOOOOOOO!!!
    def __init__(self, max_length):
        self.data = deque([], maxlen=max_length)

    def append(self, transition):
        # A transition is a tuple which contains information used for training the AI.
        # Transition = (AI memory, action made by AI, New memory after action was made, Reward earned from the action, saved h0, saved c0)
        self.data.append(transition)

    def get_sample(self, batch_size):
        return random.sample(self.data, batch_size)
    
    def __len__(self):
        return len(self.data)


class HivemindLSTM(nn.Module):
    # We will be using an LSTM model in this experiment. This LSTM will control the AI objects.

    def __init__(self, num_layers, input_size, hidden_size, output_size, num_ais):
        super(HivemindLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_ais = num_ais
        self.num_layers = num_layers

        # LSTM used by the AI
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.shared_layer = nn.Linear(hidden_size, hidden_size)
        self.split_layers = nn.ModuleList([nn.Linear(hidden_size, output_size) for _ in range(num_ais)])

    def forward(self, state, h0=None, c0=None):
        # Input tensor ideas: sequence = [aix, aiy, dx, dy, player_x, player_y, glue_x1, glue_y1, glue_x2, glue_y2, ...]
        # where dx and dy are the player's velocity, player_x and player_y are the player's position,
        # Input_Tensor = [BATCH, NUM_SAVED_FRAMES, sequence length]. Where there will be NUM_SAVED_FRAMES amount of sequances in the tensor.
        # The sequences include the current frame plus the last NUM_SAVED_FRAMES-1 frames.

        # Create Cells if they do not exist
        if (h0 is None):
            h0 = torch.zeros(self.num_layers, state.size(0), self.hidden_size).to(device)
        if (c0 is None):
            c0 = torch.zeros(self.num_layers, state.size(0), self.hidden_size).to(device)
        
        # Put the state through the LSTM
        out, (h0, c0) = self.lstm(state, (h0, c0))

        # Input the output from the LSTM to the shared layer.
        shared_features = torch.relu(self.shared_layer(out[:, -1, :]))

        # Use the split layers
        ai_outputs = []
        for i in range(self.num_ais):
            ai_q_values = self.split_layers[i](shared_features)
            ai_outputs.append(ai_q_values)

        ai_output = torch.stack(ai_outputs, dim=1)
        
        h0 = h0.detach()
        c0 = c0.detach()

        # Output tensor: [action 0, action 1, action 2, action 3, action 4, action 5, action 6, action 7, action 8]
        # Note: the actions are structured as a binary tensor which determine which direction the thing go. [up, down, left, right]
        # 0 = False, 1 = True
        # action 0 = [0, 0, 0, 0] Going nowhere (Like my life)
        # action 1 = [1, 0, 0, 0] Going up
        # action 2 = [0, 1, 0, 0] Going down
        # action 3 = [0, 0, 1, 0] Going left
        # action 4 = [0, 0, 0, 1] Going right
        # action 5 = [1, 0, 1, 0] Going Up + Left
        # action 6 = [1, 0, 0, 1] Going Up + Right
        # action 7 = [0, 1, 1, 0] Going Down + Left
        # action 8 = [0, 1, 0, 1] Going Down + Right
        # where up, down, left, and right are the AI's actions to move. Each variable will be a binary value of 0 or 1/False or True. 
        return ai_output, h0, c0
    

class AIHivemindManager: # HIVEMIND TIME!!!
    def __init__(self, num_ais, glues, player):
        self.policy_model = HivemindLSTM(NUM_LAYERS, SEQUENCE_LENGTH, HIDDEN_SIZE, OUTPUT_SIZE, num_ais).to(device)
        self.ai_save_data = None
        self.model_number = None
        self.epsilon = 1
        self.load_model()
        self.target_model = HivemindLSTM(NUM_LAYERS, SEQUENCE_LENGTH, HIDDEN_SIZE, OUTPUT_SIZE, num_ais).to(device)
        self.target_model.load_state_dict(self.policy_model.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=LEARNING_RATE)
        self.glues = glues
        self.player = player
        self.ai_list = self.create_ais(num_ais)
        self.loss_fn = nn.SmoothL1Loss()
        self.frame_count = 0
        self.data_manager = TrainingData(max_length=3600)
        self.previous_memory = None
        self.memory = torch.zeros((NUM_SAVED_FRAMES, SEQUENCE_LENGTH), dtype=torch.float32).to(device)

        self.h0 = None
        self.c0 = None
        self.previous_h0 = None
        self.previous_c0 = None

        self.total_loss = 0
        self.total_rewards = 0

        self.loss_count = 0
        self.reward_count = 0

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
    
    def add_frame_to_memory(self, frame):
        frame = frame.to(device)

        for i in range(NUM_SAVED_FRAMES):
            if ((self.memory[NUM_SAVED_FRAMES - (i+1)] == torch.zeros(SEQUENCE_LENGTH, dtype=torch.float32).to(device)).sum().item() == SEQUENCE_LENGTH):
                self.memory[NUM_SAVED_FRAMES - (i+1)] = frame
                return
        
        self.memory = torch.cat((self.memory[1 : NUM_SAVED_FRAMES], frame), dim=0)
    
    def update_memory(self):
        x, y = self.player.get_center()
        tensor = torch.tensor([
            x/WINDOW_X, 
            y/WINDOW_Y,
            self.player.dx/MAX_VELOCITY,
            self.player.dy/MAX_VELOCITY
            ], dtype=torch.float32)
        
        list_ai_info = []
        for ai in self.ai_list:
            x, y = ai.get_center()
            list_ai_info.append(x/WINDOW_X)
            list_ai_info.append(y/WINDOW_Y)
            list_ai_info.append(ai.dx/MAX_VELOCITY)
            list_ai_info.append(ai.dy/MAX_VELOCITY)
        ai_tensor = torch.tensor(list_ai_info, dtype=torch.float32)
        
        list_glues_location = []
        for glue in self.glues:
            x_glue, y_glue = glue.get_center()
            list_glues_location.append(x_glue/WINDOW_X)
            list_glues_location.append(y_glue/WINDOW_Y)

        glue_tensor = torch.tensor(list_glues_location, dtype=torch.float32)
        tensor = torch.cat((tensor, ai_tensor ,glue_tensor), dim=0)
        tensor = tensor.unsqueeze(0)
        self.add_frame_to_memory(tensor)
    
    def move_ais(self):

        # Sync the Target model with the Policy models after 10 frames. 
        if self.frame_count >= SYNC_MODEL: 
            self.target_model.load_state_dict(self.policy_model.state_dict())
        self.previous_h0, self.previous_c0 = self.h0, self.c0
        q_values, self.h0, self.c0 = self.policy_model(self.memory.unsqueeze(0), self.h0, self.c0)
        q_values = q_values.squeeze(0)
        self.previous_memory = self.memory
        for ai_index, ai in enumerate(self.ai_list):
            individual_q_value = q_values[ai_index]
            ai.ai_move(individual_q_value, self.epsilon)

    def calculate_reward(self):

        rewards = torch.zeros(len(self.ai_list))
        global_reward = 0
        for i, ai in enumerate(self.ai_list):
            reward = 0
            if ai.in_glue:
                reward += GLUE_CONTACT
                global_reward += GLUE_CONTACT / (GLOBAL_DIVIDE*len(self.ai_list))
            if ai.touching_player:
                reward += PLAYER_CONTACT
                global_reward += PLAYER_CONTACT / (GLOBAL_DIVIDE*len(self.ai_list))
            if ai.moving_into_wall(axis='x'):
                reward += WALL_CONTACT
                global_reward += WALL_CONTACT / (GLOBAL_DIVIDE*len(self.ai_list))
            if ai.moving_into_wall(axis='y'):
                reward += WALL_CONTACT
                global_reward += WALL_CONTACT / (GLOBAL_DIVIDE*len(self.ai_list))
            if ai.moving_towards_player(self.player, axis='x') and not ai.in_glue:
                reward += MOVING_TOWARDS_PLAYER
                global_reward += MOVING_TOWARDS_PLAYER / (GLOBAL_DIVIDE*len(self.ai_list))
            if ai.moving_towards_player(self.player, axis='y') and not ai.in_glue:
                reward += MOVING_TOWARDS_PLAYER
                global_reward += MOVING_TOWARDS_PLAYER / (GLOBAL_DIVIDE*len(self.ai_list))
            if ai.nearby_player(self.player) and not ai.in_glue:
                reward += PLAYER_NEARBY
                global_reward += PLAYER_NEARBY / (GLOBAL_DIVIDE*len(self.ai_list))

            if ai.dx == 0 and ai.dy == 0:
                reward += NO_MOVEMENT
                global_reward += NO_MOVEMENT / (GLOBAL_DIVIDE*len(self.ai_list))

            if not ai.previous_x == None and not self.player.previous_x == None:
                ai_x, ai_y = ai.get_center()
                player_x, player_y = self.player.get_center()

                x_difference = (abs(ai.previous_x - self.player.previous_x) - abs(ai_x - player_x))/MAX_VELOCITY
                y_difference = (abs(ai.previous_y - self.player.previous_y) - abs(ai_y - player_y))/MAX_VELOCITY

                reward += (x_difference + y_difference)/2
                
            ai.in_glue = False
            rewards[i] = reward

        # Increases reward if multiple AIs are touching the player
        num_ais_touching_player = sum([1 for ai in self.ai_list if ai.touching_player])
        if num_ais_touching_player > 1:
            rewards += num_ais_touching_player * .5

        # Adds team success/failiure
        rewards += global_reward

        self.total_rewards = (sum(rewards)/len(rewards)).item()
        self.reward_count += 1

        return rewards

    def save_data(self):
        # Create Rewards
        if self.previous_h0 is None or self.previous_c0 is None:
            return
        if self.h0 is None or self.c0 is None:
            return
        reward = self.calculate_reward()

        # Select actions
        actions = [ai.action for ai in self.ai_list]

        self.data_manager.append((
            self.previous_memory, 
            actions, 
            self.memory, 
            reward, 
            self.previous_h0,
            self.previous_c0,
            self.h0,
            self.c0
            ))

    def train_ai(self):

        batch = self.data_manager.get_sample(BATCH_SIZE)

        # Get actions and rewrads
        actions = torch.tensor([batch[i][1] for i in range(len(batch))]).to(device)
        rewards = torch.stack([batch[i][3] for i in range(len(batch))]).to(device)

        # Get the states and new states.
        states = [batch[i][0]for i in range(len(batch))]
        states = torch.stack(states, dim=0)
        new_states = [batch[i][2] for i in range(len(batch))]
        new_states = torch.stack(new_states, dim=0)

        # Get the h0 and the c0 cells
        h0s = torch.stack([batch[i][4] for i in range(len(batch))], dim=2).to(device).squeeze(dim=1)
        c0s = torch.stack([batch[i][5] for i in range(len(batch))], dim=2).to(device).squeeze(dim=1)
        new_h0s = torch.stack([batch[i][6] for i in range(len(batch))], dim=2).to(device).squeeze(dim=1)
        new_c0s = torch.stack([batch[i][7] for i in range(len(batch))], dim=2).to(device).squeeze(dim=1)

        q_values, _, _ = self.policy_model(states, h0s, c0s)

        with torch.no_grad():
            # Get next q values from target model
            next_q, _, _ = self.target_model(new_states, new_h0s, new_c0s)
            # Get the max split q_values for each AI object
            next_q_max = next_q.max(dim=2)[0]
            # Calculate the Target using the split max q values. This will be used for training.
            targets = rewards + DISCOUNT_FACTOR * next_q_max 

        total_loss = 0

        for ai_index in range(NUM_AI_OBJECTS):
            # Get individual Q_values for each AI object
            q_values_for_ai = q_values[:, ai_index, :]
            actions_for_ai = actions[:, ai_index]
            target_for_ai = targets[:, ai_index]

            # Match Q_values to the chosen action.
            q_values_to_action = torch.stack([q_values_for_ai[i][actions_for_ai[i]] for i in range(len(batch))])

            # Calculate loss using targets and individual AI q_values
            individual_loss = self.loss_fn(q_values_to_action, target_for_ai)
            total_loss += individual_loss

        # Average the total_loss.
        total_loss /= NUM_AI_OBJECTS

        # Optimize the model
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
        self.optimizer.step()

        self.total_loss = (total_loss.detach().to("cpu")).item()
        self.loss_count += 1

    def load_model(self):
        # This function checks if one of the saved models can be loaded, and if it can the function will load the model.
        # If not the function will create a new model and save its input shape so it can be saved and used in the future.

        # Check if model exists

        with open(INFO_FILE, 'r') as f:
            self.ai_save_data = json.load(f)
        
        empty = self.match_model()

        if self.model_number == None:
            print("Model not found. Creating a new one!")
            if empty:
                self.ai_save_data["Model"].append(1)
                self.model_number = 1
            else:
                self.model_number = max(self.ai_save_data["Model"]) + 1
                self.ai_save_data["Model"].append(self.model_number)
            self.ai_save_data["Ais"].append(NUM_AI_OBJECTS)
            self.ai_save_data["Glues"].append(GLUES)
            self.ai_save_data["Hidden"].append(HIDDEN_SIZE)
            self.ai_save_data["Frames"].append(NUM_SAVED_FRAMES)
            self.ai_save_data["Layers"].append(NUM_LAYERS)
            self.ai_save_data["Epsilon"].append(1)
            return
        
        print(f"Found model {self.model_number}, Now loading model...")
        model_dir = SAVE_FOLDER + '/model_' + str(self.model_number) + ".pth"
        save_dict = torch.load(model_dir)
        self.policy_model.load_state_dict(save_dict)
        print(f"Model {self.model_number} successfully loaded!")


    def match_model(self):

        if len(self.ai_save_data["Model"]) == 0:
            return True
        
        for i in range(len(self.ai_save_data["Model"])):
            ais = self.ai_save_data["Ais"][i] == NUM_AI_OBJECTS
            glues = self.ai_save_data["Glues"][i] == GLUES
            hidden = self.ai_save_data["Hidden"][i] == HIDDEN_SIZE
            frames = self.ai_save_data["Frames"][i] == NUM_SAVED_FRAMES
            layers = self.ai_save_data["Layers"][i] == NUM_LAYERS

            if ais and glues and hidden and frames and layers:
                self.model_number = self.ai_save_data["Model"][i]
                self.epsilon = self.ai_save_data["Epsilon"][i]
                break

        return False
        
    def save_model(self):
        # Saves the model 

        with open(INFO_FILE, 'w') as f:
            json.dump(self.ai_save_data, f)

        print("Saving model")
        torch.save(self.policy_model.state_dict(), SAVE_FOLDER + "/" + "model_" + str(self.model_number) + ".pth")
        print(f"Model {self.model_number} saved successfully.\n")


class ProgressTracker:
    # Keeps track of important data and displays them at the end.

    data = {
        "Episodes": [],
        "Rewards": [],
        "Actions": [],
        "Health": [],
        "Loss": [],
        "Epsilon": [],
        "Time": []
    }

    model_number = None

    def append(self, item, location):
        try:
            self.data[location].append(item)
        except Exception:
            print("Invalid location, please input a valid location.")
    
    def __len__(self):
        return len(self.data["Episodes"])
    
    def calculate_sd(self, data, mean):
        # Calculate the Mean Absolute Diviation of the Data: sd^2 = (∑((value - mean)^2))/total

        sd = 0

        for value in data:
            sd += (value - mean)**2
        
        sd /= len(data)
        sd = math.sqrt(sd)
        return sd

    def calculate_mean(self, data):
        return sum(data)/len(data)

    def calculate_z_scores(self, data, mean, sd):
        # Calculate the z_scores of the data, equation: z_score = (value - mean_of_data)/standard deviation

        z_scores = []
        for value in data:
            try:
                z_score = (value-mean)/sd
            except ZeroDivisionError:
                z_score = 0
            z_scores.append(z_score)

        return z_scores
    
    def save_as_cvs(self):
        # Convert any potential tensors to Python numbers
        cleaned_data = {
            "Episodes": [float(x) if hasattr(x, 'item') else x for x in self.data["Episodes"]],
            "Rewards": [float(x) if hasattr(x, 'item') else x for x in self.data["Rewards"]],
            "Actions": [float(x) if hasattr(x, 'item') else x for x in self.data["Actions"]],
            "Health": [float(x) if hasattr(x, 'item') else x for x in self.data["Health"]],
            "Loss": [float(x) if hasattr(x, 'item') else x for x in self.data["Loss"]],
            "Epsilon": [float(x) if hasattr(x, 'item') else x for x in self.data["Epsilon"]],
            "Time": [float(x) if hasattr(x, 'item') else x for x in self.data["Time"]]
        }
        
        file_name = DATA_FOLDER + "/" + "model_" + str(self.model_number) + "_data.csv"
        df = pd.DataFrame(cleaned_data)

        # If no csv file exists for this save, then create a new one and return
        if not os.path.exists(file_name):
            df.to_csv(file_name, index=False)
            return
        
        # If there is already a csv file with data in it, then we merge the data
        old_df = pd.read_csv(file_name)
        # Fix the concatenation
        new_df = pd.concat([old_df, df], ignore_index=True)
        new_df.to_csv(file_name, index=False)

    def graph(self, data, names):
        # Takes in a list of data and graphs all of the data vs the episodes

        for i in range(len(data)):
            values = np.array(data[i])
            plt.plot(self.episodes, values, label = names[i])
        
        plt.legend()
        plt.show()
            

    def get_info(self, data):
        mean = self.calculate_mean(data)
        sd = self.calculate_sd(data, mean)
        z_scores = self.calculate_z_scores(data, mean, sd)
        return mean, sd, z_scores

    def graph_results(self):
        self.episodes = self.data["Episodes"]
        _, _, r_z_scores = self.get_info(self.data["Rewards"])
        _, _, h_z_scores = self.get_info(self.data["Health"])
        _, _, e_z_scores = self.get_info(self.data["Epsilon"])
        _, _, l_z_scores = self.get_info(self.data["Loss"])
        _, _, t_z_scores = self.get_info(self.data["Time"])
        z_score_list = [r_z_scores, h_z_scores, e_z_scores, l_z_scores, t_z_scores]
        name_list = ["Rewards", "Health", "Epsilon", "Loss", "Time"]
        self.graph(z_score_list, name_list)


def check_for_folder():
    # Creates AI save directory and JSON file if one is not present.

    if not os.path.isdir(SAVE_FOLDER):
        os.mkdir(SAVE_FOLDER)
    if not os.path.isfile(INFO_FILE):
        open(INFO_FILE, "x")
        with open(INFO_FILE, "w") as f:
            json.dump(AI_SAVE_DATA, f)
    if not os.path.isdir(DATA_FOLDER):
        os.mkdir(DATA_FOLDER)


def game_end(ai_manager, player, progress_tracker, time):
    global previous_time

    new_epsilon = (ai_manager.epsilon - 1/EPSILON_DECAY)
    if new_epsilon < 0:
        new_epsilon = 0

    ai_manager.ai_save_data["Epsilon"][ai_manager.model_number - 1] = new_epsilon
    ai_manager.save_model()

    previous_time = pygame.time.get_ticks()

    # Save Progress to AI
    rewards = ai_manager.total_rewards/ai_manager.reward_count
    try:
        loss = ai_manager.total_loss/ai_manager.loss_count
    except ZeroDivisionError:
        loss = 0
    actions = 0
    epsilon = ai_manager.epsilon
    health = player.health
    time = round((TIME_LIMIT - time)/1000)
    for ai in ai_manager.ai_list:
        actions += ai.total_actions/ai.action_count

    actions /= len(ai_manager.ai_list)

    progress_tracker.model_number = ai_manager.model_number
    progress_tracker.append(time, "Time")
    progress_tracker.append(rewards, "Rewards")
    progress_tracker.append(loss, "Loss")
    progress_tracker.append(epsilon, "Epsilon")
    progress_tracker.append(health, "Health")
    progress_tracker.append(actions, "Actions")


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


def main(progress_tracker):
    sys.setrecursionlimit(100000)
    end = False
    num_frames = 0
    running = True
    clock = pygame.time.Clock()
    player = Player(PLAYER_DIM, PLAYER_DIM, window, WHITE, objects=[], distance=None, x=WINDOW_X/2-PLAYER_DIM/2, y=WINDOW_Y/2-PLAYER_DIM/2)
    glues = []
    for _ in range(GLUES):
        glue = Glue(GLUE_DIM, GLUE_DIM, window, YELLOW, [player]+glues, distance=GLUE_DIM)
        glues.append(glue)
    ai_manager = AIHivemindManager(NUM_AI_OBJECTS, glues, player)

    # Game loop, YIPPEEEEEEE
    while running:
        current_time = pygame.time.get_ticks()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_end(ai_manager, player, progress_tracker, current_time - previous_time)
                running = False
                end = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Reset the game if the space key is pressed.
                    game_end(ai_manager, player, progress_tracker, current_time - previous_time)
                    running = False
                elif event.key == pygame.K_LSHIFT:
                    player.override = not player.override

        if pygame.time.get_ticks()-previous_time >= TIME_LIMIT:
            game_end(ai_manager, player, progress_tracker, current_time - previous_time)
            running = False

        # AI moves first
        ai_manager.frame_count += 1
        ai_manager.move_ais()
        for ai in ai_manager.ai_list:
            ai.check_for_collisions(current_time, player)

        for glue in glues:
            glue.check_for_collisions(ai_manager.ai_list + [player], current_time)
            glue.check_bounds()
            glue.apply_friction()
            if (current_time - glue.movement_timer >= GLUE_MOVEMENT_TIME):
                glue.random_move()
                glue.movement_timer = current_time
            else:
                glue.move()

        player.player_move(current_time)
        if player.health <= 0:
            game_end(ai_manager, player, progress_tracker, current_time - previous_time)
            running = False

        # Saved the changes made to the enviorment, Save the needed information for training, Then Train the AI.
        ai_manager.update_memory()
        ai_manager.save_data()
        if num_frames >= BATCH_SIZE:
            ai_manager.train_ai()

        draw_game(player, glues, current_time - previous_time, ai_manager)
        num_frames += 1
        clock.tick(60)
    
    return not end


if __name__ == "__main__":
    progress_tracker = ProgressTracker()

    episodes = 1
    run = True
    check_for_folder()
    while run:
        progress_tracker.append(episodes, "Episodes")
        run = main(progress_tracker)
        episodes += 1
        if episodes >= MAX_EPISODES:
            run = False
    
    pygame.quit()

    print(f"Number of Episodes: {episodes}")
    try:
        progress_tracker.save_as_cvs()
        print("Pls")
    except:
        data = progress_tracker.data
        print(f"Episodes: {data["Episodes"][0]} | Rewards: {data["Rewards"][0]} | Actions: {data["Actions"][0]}\n")
        print(f"Health: {data['Health'][0]} | Loss: {data['Loss'][0]} | Epsilon: {data['Epsilon'][0]} | Time: {data['Time'][0]}")

    progress_tracker.graph_results()
