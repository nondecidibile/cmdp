import numpy as np
from gym.envs.toy_text import gridworld_cont
from tkinter import *
import time

DIM = 300
window = Tk()
canvas = Canvas(window, width=DIM, height=DIM, borderwidth=0, highlightthickness=0, bg="black")
canvas.pack()

def draw_circle(x, y, r, canvas, **kwargs):
	x0 = x - r
	y0 = y - r
	x1 = x + r
	y1 = y + r
	return canvas.create_oval(x0, y0, x1, y1, **kwargs)

mdp = gridworld_cont.GridworldContEnv()
mdp.horizon = 50

state = mdp.reset()
reward = 0
done = False

x,y,xg,yg = state
xx = x
yy = y

def lerp(a,b,alpha):
	return b*alpha + (1-alpha)*a

while not done:

	state,reward,done = mdp.step(np.array(np.random.normal(0,1,2)))
	x,y,xg,yg = state

	for i in range(30):

		tx = lerp(xx,x,i/30)
		ty = lerp(yy,y,i/30)

		canvas.delete('all')
		draw_circle(DIM/2+tx*(DIM/mdp.DIM)/2,DIM/2+ty*(DIM/mdp.DIM)/2,10,canvas,fill="blue")
		draw_circle(DIM/2+xg*(DIM/mdp.DIM)/2,DIM/2+yg*(DIM/mdp.DIM)/2,DIM/2*mdp.END_DISTANCE/mdp.DIM,canvas,outline="red")
		window.update()

		time.sleep(0.004)
	
	xx = x
	yy = y