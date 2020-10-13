import pygame 
import sys
from utils import load_model , predict , translate
import numpy as np
import cv2
import time
import random

rock_surface = pygame.image.load('data/rock.png')
paper_surface = pygame.image.load('data/paper.png')
scissors_surface = pygame.image.load('data/scissors.png')

question_mark = cv2.imread('data/question.png')
question_mark = cv2.resize(question_mark,(300,200))
question_mark_surface = pygame.surfarray.make_surface(question_mark)

data = {
	'rock' : rock_surface,
	'paper': paper_surface,
	'scissors': scissors_surface
}


def game_loop():
	pygame.init()
	screen = pygame.display.set_mode((400,400))
	pygame.display.set_caption('Rock Paper Scissors AI')
	font = pygame.font.Font('freesansbold.ttf', 18)

	vc = cv2.VideoCapture(1) # give it 0 for built in camera
	vc.set(cv2.CAP_PROP_FRAME_WIDTH, 300)
	vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 200)

	if vc.isOpened(): # try to get the first frame
	    rval, frame = vc.read()
	else:
	    rval = False

	model = load_model((120,120,3))
	model.load_weights('output/weights.h5')

	computer_move = random.choice(['rock' , 'paper' , 'scissors'])
	play_flag = False
	done = False

	while not done:

		rval, frame = vc.read()
		conf , pred = predict(model , frame)

		if conf:
			player_move = translate(pred)
		else:
			player_move = 'NOT CLASS'

		shown_frame = cv2.resize(frame , (300,200) )
		surf = pygame.surfarray.make_surface(shown_frame)
		screen.blit(surf , (200,0))


		if not play_flag:
			screen.blit(question_mark_surface , (0,0))
			computer_move = random.choice(['rock' , 'paper' , 'scissors'])
			screen.fill(pygame.Color("black") , (0,300,400,400))
			text = font.render('Put your hand then press Space bar' , True ,(255,255,255))
			textRect = text.get_rect()
			textRect.center = (400 // 2, 350)  

		else:
			computer_move_surface = data[computer_move]
			screen.blit(computer_move_surface , (0,0))
			screen.fill(pygame.Color("black") , (0,300,400,400))
			if player_move == 'NOT CLASS':
				print('Please put your hand')
			elif player_move == computer_move:
				text = font.render('Tie' , True ,(255,255,255))
			if player_move == 'rock' and computer_move == 'paper':
				text = font.render('Computer Win' , True ,(255,255,255))				
			elif player_move == 'rock' and computer_move == 'scissors':	
				text = font.render('Player Win' , True ,(255,255,255))				
			elif player_move == 'paper' and computer_move == 'rock':					
				text = font.render('Player Win' , True ,(255,255,255))
			elif player_move == 'paper' and computer_move == 'scissors':					
				text = font.render('Computer Win' , True ,(255,255,255))
			elif player_move == 'scissors' and computer_move == 'paper':					
				text = font.render('Player Win' , True ,(255,255,255))
			elif player_move == 'scissors' and computer_move == 'rock':				
				text = font.render('Computer Win' , True ,(255,255,255))

			textRect = text.get_rect()
			textRect.center = (400 // 2, 350)			


		shown_frame = cv2.resize(frame , (300,200) )
		surf = pygame.surfarray.make_surface(shown_frame)
		screen.blit(surf , (200,0))

		screen.blit(text , textRect)

		for event in pygame.event.get():
			if event.type is pygame.QUIT:
				pygame.quit()
				quit()
			elif event.type is pygame.KEYDOWN:
				if event.key is pygame.K_SPACE:
					if play_flag == False: play_flag = True
					elif play_flag == True: play_flag = False
				elif event.key is pygame.K_ESCAPE:
					done = True

		pygame.display.update()					





if __name__ == '__main__':
	game_loop()