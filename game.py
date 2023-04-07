'''
Hand Controlled Tetris | Python: Pygame, OpenCV and Mediapipe
Reference: https://www.youtube.com/watch?v=sFDt9upueRE

Hand-Controlled-Tetris
Reference: https://gist.github.com/kpkpkps/79357982a6044553baf3610ad39d0c90

Pygame in 90 Minutes - For Beginners
Reference: https://www.youtube.com/watch?v=jO6qQDNa2UY

'''

import pygame
import os
import random
from threading import Thread
from threading import Event
import multiprocessing
import csv 
 
pygame.font.init()
pygame.mixer.init()

#setting the whidht and height of the display window
WIDTH, HEIGHT = 1020, 760
WIN = pygame.display.set_mode((WIDTH, HEIGHT))

#Define Colours 
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BROWN = (235,245,255)

SCORE_TABLE = os.path.join('Assets','score_table.csv')

pygame.display.set_caption("Balloon Game")

FPS = 30

#Define Fonts 
SCORE_FONT = pygame.font.SysFont('comicsans', 40)
USER_FONT = pygame.font.SysFont('comicsans', 20)

BALLON_WIDTH = 200
BALLON_HEIGHT = 300

RED_BALLON_IMAGE = pygame.image.load(os.path.join('Assets','red.png'))
#line to resize the image 
RED_BALLON = pygame.transform.scale(RED_BALLON_IMAGE, (BALLON_WIDTH,BALLON_HEIGHT))
#pygame.transform.rotate(image, dagre) to rotate the image  

YELLOW_BALLON_IMAGE = pygame.image.load(os.path.join('Assets','yellow.png'))
#line to resize the image 
YELLOW_BALLON = pygame.transform.scale(YELLOW_BALLON_IMAGE, (BALLON_WIDTH + 150, BALLON_HEIGHT + 200))
#pygame.transform.rotate(image, dagre) to rotate the image  

GREEN_BALLON_IMAGE = pygame.image.load(os.path.join('Assets','green.png'))
#line to resize the image 
GREEN_BALLON = pygame.transform.scale(GREEN_BALLON_IMAGE, (BALLON_WIDTH + 150, BALLON_HEIGHT + 200))
#pygame.transform.rotate(image, dagre) to rotate the image

GREEN_BALLON_GAME = pygame.transform.scale(GREEN_BALLON_IMAGE, (BALLON_WIDTH , BALLON_HEIGHT))
#pygame.transform.rotate(image, dagre) to rotate the image

POP_IMAGE = pygame.image.load(os.path.join('Assets','pop.jpg'))
#line to resize the image  
POP =  pygame.transform.scale(POP_IMAGE, (BALLON_WIDTH ,BALLON_HEIGHT))

START_IMAGE = pygame.image.load(os.path.join('Assets','start.jpg'))
#line to resize the image 
START_RESIZE = pygame.transform.scale(START_IMAGE, (300, 100))

# velocity of the balloon
#VEL_BALLON = 2

#Creating custom event
WALL_HIT = pygame.USEREVENT + 1
Ballon_HIT = pygame.USEREVENT + 2

def draw_window(red_rect, score, wall_hits, hit):
    WIN.fill(WHITE)
    score_text = SCORE_FONT.render("Score: " + str(score), 1, BLACK)
    hit_wall_text = SCORE_FONT.render("Throws: " + str(wall_hits), 1, BLACK)
    
    WIN.blit(GREEN_BALLON_GAME, (red_rect.x, red_rect.y))
    if hit:
       WIN.blit(POP, (red_rect.x, red_rect.y))
        # if X_detect is not None or Y_detect is not None:
        #     pygame.draw.line(WIN, RED, (X_detect, Y_detect), (X_detect + 10, Y_detect + 10))
        #     pygame.draw.line(WIN, RED, (X_detect, Y_detect + 10), (X_detect+ 10, Y_detect))

    WIN.blit(score_text, (10, 10))
    WIN.blit(hit_wall_text, (WIDTH - hit_wall_text.get_width() - 10 , 10))

    #Screen update
    pygame.display.update()
    if hit:     
        pygame.time.delay(250)
    
def draw_menu(play_rect, play_text):
    pygame.display.set_caption("Menu")
    hello_text = SCORE_FONT.render("POP IT!", 1, BLACK)
    instrection_text = SCORE_FONT.render("You have 6 throws, pop the balloon to start.", 1, BLACK)
    
    WIN.fill(WHITE)
    WIN.blit(GREEN_BALLON, (play_rect.x, play_rect.y ))
    WIN.blit(play_text, (play_rect.x + 60 , play_rect.y + 150))
    WIN.blit(hello_text,((WIDTH-hello_text.get_width())//2 , 10))
    WIN.blit(instrection_text,((WIDTH - instrection_text.get_width())//2, 20 + hello_text.get_height() ))
    #Screen update 
    pygame.display.update()

def draw_game_over(score, wall_hits, user_text):
    WIN.fill(BLACK)
    score_text = SCORE_FONT.render("You got " + str(score) +" out of "+ str(wall_hits) + " trys", 1, WHITE)
    WIN.blit(score_text, ((WIDTH - score_text.get_width())//2 , HEIGHT//2))   
    text_surface = USER_FONT.render(user_text, True, WHITE) 
    WIN.blit(text_surface, ((WIDTH - score_text.get_width())//2 , HEIGHT//2 + score_text.get_height()))
    pygame.display.update()
   

def restart_rect(red_rect):
    red_rect.x = random.randint(BALLON_WIDTH, WIDTH - 2*BALLON_WIDTH)
    red_rect.y = HEIGHT - BALLON_HEIGHT
    return red_rect.x, red_rect.y

def add_competitor(score,name):
    score = int((score /6) *100)
    comp_table = {} 
    comp_table[name] = score   
    with open(SCORE_TABLE, 'a', newline='') as file: 
        writer = csv.writer(file)
        writer.writerows(comp_table.items())
   
    with open(SCORE_TABLE, 'r', newline='') as file:
        reader = csv.reader(file)
        dicts_list = []
        for row in reader:
            my_dict = {}
            my_dict[row[0]]= row[1]
            dicts_list.append(my_dict)
    n = {}
    for e in dicts_list:
        n.update(e)
    topTen = dict(sorted(n.items(), key=lambda x: int(x[1]),reverse=True))
    if len(topTen)> 10:
        topTen = dict(list(topTen.items())[:10])
    return topTen


def Competitors_table(dicts_list):
    pygame.display.set_caption("Score Table")
    events = pygame.event.get()
    WIN.fill((0, 0, 0))
    font = pygame.font.SysFont("Arial", 28)
    x_pos = 300
    y_pos = 100
    competitor_text = font.render("Competitors", True, (255, 255, 255))
    WIN.blit(competitor_text, (x_pos, y_pos))
    score_text = font.render("Score (%)", True, (255, 255, 255))
    WIN.blit(score_text, (x_pos + 200, y_pos))
    pygame.draw.line(WIN, (255, 255, 255), (x_pos, y_pos + 30), (x_pos + 300, y_pos + 30))
    pygame.draw.line(WIN, (255, 255, 255), (x_pos + 200, y_pos), (x_pos + 200, y_pos + 200))
    y_pos += 50
    for name, score in dicts_list.items():            
        name_text = font.render(name, True, (255, 255, 255))
        WIN.blit(name_text, (x_pos, y_pos))
        score_text = font.render(str(score), True, (255, 255, 255))
        WIN.blit(score_text, (x_pos + 200, y_pos))
        y_pos += 30
        pygame.draw.line(WIN, (255, 255, 255), (x_pos, y_pos), (x_pos + 300, y_pos))
        # pygame.draw.line(WIN, (255, 255, 255), (x_pos + 200, y_pos), (x_pos + 200, y_pos))
    pygame.display.update()
    run_end = True
    while run_end:
        for event in pygame.event.get():    
            if event.type == pygame.QUIT:
                run_end = False 

def gameOver(score, wall_hits, winSound, lossSound):
    wall_hits = 6 - wall_hits
    pygame.display.set_caption("Game Over")
    run_GO = True
    user_text = 'Enter Name: '
    text_len = len(user_text)
    if score > 3:
        winSound.play()
    else:
        lossSound.play()
    while run_GO:
        for event in pygame.event.get():    
            if event.type == pygame.QUIT:
                run_GO = False 
            elif event.type == pygame.KEYDOWN:

                if event.key == pygame.K_BACKSPACE:
                    user_text = user_text[:-1]
                elif event.key == pygame.K_RETURN:  
                    run_GO = False 
                else:
                    user_text += event.unicode
        draw_game_over(score, wall_hits, user_text)
    comp_table = add_competitor(score,user_text[text_len:])    
    Competitors_table(comp_table)




def game_main(event_hit, game_detection):
    pygame.display.set_caption("Pop it!")
    hitSound = pygame.mixer.Sound(os.path.join('Assets', "pop_balloon.mp3"))
    winSound = pygame.mixer.Sound(os.path.join('Assets', "winning.mp3"))
    lossSound = pygame.mixer.Sound(os.path.join('Assets', "lossing.mp3"))
    score = 0 
    wall_hits = 6

    #the argomants for this X, Y, width, height.
    X_ballon = random.randint(BALLON_WIDTH, WIDTH - 2*BALLON_WIDTH)
    red_rect = pygame.Rect(X_ballon,HEIGHT - BALLON_HEIGHT, BALLON_WIDTH,BALLON_HEIGHT)
    
    clock = pygame.time.Clock()
    
    run = True 
    #hit ballon falg
    hit = False
    #hit wall falg
    hit_wall = False

    velocity = 2

    while run:

        clock.tick(FPS)
        mouse_pos_menu = pygame.mouse.get_pos()

        #Conditions for the custom event
        if game_detection.is_set():
            pygame.event.post(pygame.event.Event(WALL_HIT))    
            game_detection.clear()
            

        #Game events loops         
        for event in pygame.event.get():
            #if user press on the x in the game window 
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                wall_hits -= 1
                if mouse_pos_menu[0] in range(red_rect.x, red_rect.x + red_rect.width) and mouse_pos_menu[1] in range(red_rect.y, red_rect.y + red_rect.height):                   
                    hit = True
                    print("Button Press!")
            elif event.type == WALL_HIT: 
                wall_hits -= 1
                #pygame.time.delay(800)
                event_hit.wait(timeout= 0.7)
        
        red_rect.y -= velocity   

        if red_rect.y < 0 or hit or event_hit.is_set():
            if hit or event_hit.is_set():
                hit = True
                event_hit.clear()  
                hitSound.play()
                score += 1            
                velocity += 1
            draw_window(red_rect, score, wall_hits, hit)
            red_rect.x,  red_rect.y = restart_rect(red_rect)
        hit = False
        
        if wall_hits <= 0:
            pygame.time.delay(1000)
            run = False
        draw_window(red_rect, score, wall_hits, hit)
    
    gameOver(score, wall_hits, winSound, lossSound)
    #pygame.quit() 

#Creating the manu for the game
def menu(event_hit, game_detection):

    WIN.fill(WHITE) 
    x_play = WIDTH//2 - BALLON_WIDTH
    #X_explain =  WIDTH//2 + BALLON_WIDTH
    y_menu = HEIGHT//2 - 200

    play_rect = pygame.Rect(x_play , y_menu , BALLON_WIDTH + 100, BALLON_HEIGHT + 100)

    play_text = SCORE_FONT.render("Pop to Start", 1, WHITE)

    run = True

    while run:

        mouse_pos_menu = pygame.mouse.get_pos()
        clock1 = pygame.time.Clock()
        clock1.tick(FPS)

        draw_menu(play_rect, play_text)

        for event in pygame.event.get():
            #if user press on the x in the game window 
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if mouse_pos_menu[0] in range(play_rect.x, play_rect.x + play_rect.width) and mouse_pos_menu[1] in range(play_rect.y, play_rect.y + play_rect.height):                   
                     game_main(event_hit, game_detection)
                     run = False
                     
    pygame.quit() 
            
if __name__ == "__main__":

    event_hit = multiprocessing.Event()
    event_wall_hit = multiprocessing.Event()


    game_proc = multiprocessing.Process(target=menu, args=(event_hit, event_wall_hit, X_detect, Y_detect ))

    game_proc.start()
    game_proc.join()