import torch
import sys
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
GLUE_MOVEMENT_TIME = 5000

# Other object constants (player + AI objects)
ACCELERATION = 1
FRICTION = 0.5
MAX_VELOCITY = 15
DAMAGE_COOLDOWN = 1000
PLAYER_DIM = 50
RANDOM_MOVE = True

# AI Constants
NUM_LAYERS = 4
HIDDEN_SIZE = 128
OUTPUT_SIZE = 4
SAVE_FILE = "Models/LSTM_Models/"
TEXT_FILE = SAVE_FILE + "current_model.txt"
LEARNING_RATE = 0.000001
AI_FORWARD_TIME = 1000/40 # The AI object will change its directional vector 15 times each second thanks to this variable, this will be used for testing later
NUM_AI_OBJECTS = 5
NUM_SAVED_FRAMES = 60
SEQUENCE_LENGTH = 4 + 2 * GLUES


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
        if self.override:
            return
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


class Glue(Object):
    # Simple obstacle that is an issue for both the player and AI.

    def __init__(self, x, y, width, height, window, color):
        super().__init__(x, y, width, height, window, color)
        self.timer = 0
        self.glue_drag = 0.5
        self.amplifier = 10
        self.movement_timer = 0
        self.override = False

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
                # Damages the player for colliding with glue.
                if current_time - self.timer >= DAMAGE_COOLDOWN:
                    obj.health -= 1
                    self.timer = current_time

                          
class Player(Object):
    def __init__(self, x, y, width, height, window, color):
        super().__init__(x, y, width, height, window, color)
        self.health = 30

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
        
        self.check_bounds()
        self.apply_friction()
        self.check_max_velocity()
        self.move()


class AI(Object):
    # This is the object which the neural network will control. It is the AI which will hunt the player.

    def __init__(self, x, y, width, height, window, color, model, player):
        super().__init__(x, y, width, height, window, color)
        self.model = model
        self.player = player
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.glue_value = 0
        self.timer = 0
        self.h0 = None
        self.c0 = None
        self.directional_vector = None
        self.change_direction_timer = 0
        self.memory = torch.zeros((NUM_SAVED_FRAMES, SEQUENCE_LENGTH), dtype=torch.float32)

    def add_frame_to_memory(self, frame):

        for i in range(NUM_SAVED_FRAMES):
            if ((self.memory[NUM_SAVED_FRAMES - (i+1)] == torch.zeros(SEQUENCE_LENGTH, dtype=torch.float32)).sum().item() == SEQUENCE_LENGTH):
                self.memory[NUM_SAVED_FRAMES - (i+1)] = frame
                return
        
        self.memory = torch.cat((self.memory[1 : NUM_SAVED_FRAMES], frame), dim=0)

    def ai_move(self, glues, current_time):
        # Prepare the input tensors to be fed into the neural network.
        changed = False
        if self.directional_vector is None:
            changed = True
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
            X = X.unsqueeze(0).to("cpu")
            self.add_frame_to_memory(X)
            
            self.memory = self.memory.to(device)

            # Feed the input tensor into the neural network and get the output tensor then process the data to get the direction of the AI's acceleration.
            # output is as a vector containung 4 values that are from 0-1. With each value representing a direction: [up, down, left, right].
            original_output, self.h0, self.c0 = self.model(self.memory, self.h0, self.c0)
            original_output = original_output.squeeze(0)
            output = torch.sigmoid(original_output).to("cpu")
            output = (output > 0.5)
            self.directional_vector = output
            self.memory = self.memory.to("cpu")
            # print(f"Original Output: {original_output}\nOutput: {output}")

        if (self.directional_vector[0]):
            self.dy -= ACCELERATION
        if (self.directional_vector[1]):
            self.dy += ACCELERATION
        if (self.directional_vector[2]):
            self.dx -= ACCELERATION
        if (self.directional_vector[3]):
            self.dx += ACCELERATION

        self.check_bounds()
        self.apply_friction()
        self.check_max_velocity()        
        self.move()
        if changed:
            self.train_ai(original_output)
        self.glue_value = 0
        if (current_time - self.change_direction_timer >= AI_FORWARD_TIME):
            self.directional_vector = None
            self.change_direction_timer = current_time
            

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
            old_player_dx, old_player_dy = self.player.dx, self.player.dy
            self.player.dx = self.dx/2
            self.player.dy = self.dy/2
            self.dx = old_player_dx/2
            self.dy = old_player_dy/2
            self.timer = current_time
            self.player.health -= 2


class LSTM(nn.Module):
    # We will be using an LSTM model in this experiment. This LSTM will control the AI objects.

    def __init__(self, num_layers, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()

        # Equation which will used to calculate the loss of the AI. Its a linear line with a very small exponential curve.
        self.loss_equation = lambda distance_difference, window : distance_difference/window + (1.01)**(distance_difference/10)

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, output_size)
        
    def calculate_loss(self, player, ai, glue_value, output, proper_direction_x, proper_direction_y):
        # Will change this loss function later for better results, but for now this will do for now.

        ai_x, ai_y = ai.get_center()
        player_x, player_y = player.get_center()
        loss_x = self.loss_equation(abs(ai_x - player_x), WINDOW_X)
        loss_y = self.loss_equation(abs(ai_y - player_y), WINDOW_Y)
        loss = (loss_x + loss_y)/2
        loss = torch.tensor(loss, dtype=torch.float32, requires_grad=True).to(device).to(device)
        if glue_value > 0:
            loss *= glue_value
            proper_direction_x = False
            proper_direction_y = False
        '''
        if proper_direction_x:
            loss /= 2
        if proper_direction_y:
            loss /= 2
        
        if proper_direction_x and proper_direction_y:
            loss = 0
        '''
        output = torch.mean(output**2) * .01
        loss = loss + output
        loss = 2*loss**2
        return loss

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


def load_model(model, txt_file):
    with open(txt_file, "r") as f:
        lines = f.readlines()
        line = str(lines[0])
        number = lines[1]
        if line.startswith("model"):
            line = line.strip("\n")
            try:
                model.load_state_dict(torch.load(SAVE_FILE + line))
            except Exception:
                print("Model cannot be loaded. Creating a new one.")
            print("Model loaded successfully.")
            try:
                model_number = int(number)
            except ValueError:
                print("Model number not found. Starting with a new model number.")
                model_number = 0
            return model, model_number
        else:
            print("Model not found. Starting with a new model.")
            return model, 0
        
def save_model(model, model_number, text_file):
    # Saves the model to a file.
    torch.save(model.state_dict(), SAVE_FILE + "model_" +str(model_number) + ".pth")
    with open(text_file, "a") as f:
        f.write(f"model_{model_number}.pth\n" + str(model_number))
    print(f"Model {model_number} saved successfully.")


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
    model = LSTM(NUM_LAYERS, SEQUENCE_LENGTH, HIDDEN_SIZE, OUTPUT_SIZE).to(device)
    model, model_number = load_model(model, TEXT_FILE)
    
    for _ in range(NUM_AI_OBJECTS):
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
                        save_model(model, model_number, TEXT_FILE)
                        main()
                        running = False
                    except RecursionError:
                        running = False
                        print("RecursionError: Too many recursions. Program will now exit.")
                elif event.key == pygame.K_LSHIFT:
                    player.override = not player.override


        for glue in glues:
            glue.check_for_collisions(objects, pygame.time.get_ticks())
            glue.check_bounds()
            glue.apply_friction()
            if (pygame.time.get_ticks() - glue.movement_timer >= GLUE_MOVEMENT_TIME):
                glue.random_move()
                glue.movement_timer = pygame.time.get_ticks()
            else:
                glue.move()
                # glue.check_bounds()

        for obj in objects:
            if isinstance(obj, Player):
                obj.player_move()
                obj.random_move()
                if obj.health <= 0:
                    try:
                        save_model(model, model_number, TEXT_FILE)
                        main()
                        running = False
                    except RecursionError:
                        running = False
                        print("RecursionError: Too many recursions. Program will now exit.")
            if isinstance(obj, AI):
                obj.ai_move(glues, pygame.time.get_ticks())
                obj.check_for_collisions(pygame.time.get_ticks())
            obj.in_glue = False
        draw_game(objects, glues)
        clock.tick(60)


    save_model(model, model_number, TEXT_FILE)
    pygame.quit()


if __name__ == "__main__":
    main()
