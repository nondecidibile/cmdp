import numpy as np
from tkinter import *
import time

def lerp(a, b, alpha):
	return b*alpha+a*(1-alpha)

class Car:

	def __init__(self):

		self.pos = np.zeros(shape=(2),dtype=np.float32)
		self.angle = np.float32(0)
		self.speed = np.float32(0)
		self.collision = False

		self.maxSpeed = 6
		self.maxAccel = 0.5
		self.maxAngularVel = np.pi/64

		self.dim = 25

		self.sensorDist = 150
		self.sensors = np.array([
			[self.sensorDist,np.pi/4],
			[self.sensorDist,np.pi/6],
			[self.sensorDist,np.pi/12],
			[self.sensorDist,0],
			[self.sensorDist,-np.pi/12],
			[self.sensorDist,-np.pi/6],
			[self.sensorDist,-np.pi/4]
		],dtype=np.float32)
		self.sensorValues = np.ones(shape=self.sensors.shape[0],dtype=np.float32)
	
	def reset(self):
		self.pos = np.array([0,self.dim],dtype=np.float32)
		self.angle = np.float32(0)
		self.speed = np.float32(0)

	def step(self, action): # force [-1,1], steering [-1,1]
		acceleration = np.clip(action[0],-1,1)*self.maxAccel
		steering = np.clip(action[1],-1,1)

		self.speed += acceleration
		self.speed = np.clip(self.speed,-self.maxSpeed,self.maxSpeed)

		dtheta_coeff = np.float32(0)
		if(abs(self.speed/self.maxSpeed)<0.1):
			dtheta_coeff = 10.0*abs(self.speed/self.maxSpeed)
		else:
			dtheta_coeff = 1.0-(abs(self.speed/self.maxSpeed)-0.1)*0.3
		dtheta = steering*dtheta_coeff*self.maxAngularVel
		self.angle += dtheta

		self.pos[0] += -self.speed*np.sin(self.angle)
		self.pos[1] += self.speed*np.cos(self.angle)
	
	def printInfo(self):
		print("CAR: Pos =",np.around(self.pos,decimals=2),". Speed =",np.around(self.speed,decimals=2),". Angle =",np.around(self.angle/(np.pi),decimals=2),"π.")

class Road:
	def __init__(self):
		self.nControlPoints = 4
		self.nPoints = 2*self.nControlPoints-1
		self.segmentSize = 100
		self.roadWidth = 150

		self.points = None

		self.mean = 0
		self.stddev = 50
	
	def reset(self):
		p = np.random.normal(self.mean,self.stddev,(self.nControlPoints-2))
		self.points = np.zeros(shape=(self.nPoints,2),dtype=np.float32)
		for i in range(self.nPoints):
			if i==0 or i==1:
				# start with angle=0
				self.points[i] = np.array([0,i*self.segmentSize])
			else:
				if i%2==0: # CONTROL POINTS
					# if last point, end with angle=0, otherwise apply gaussian shift
					self.points[i] = self.points[i-1] + (p[np.int32(i/2-1)] if i!=self.nPoints-1 else 0)
					self.points[i,1] = i*self.segmentSize
				else: # SYMMETRIC POINTS
					self.points[i] = self.points[i-1] + (self.points[i-1]-self.points[i-2])
					self.points[i,1] = i*self.segmentSize

class DrivingEnv:
	def __init__(self, renderFlag=False):
		self.car = Car()
		self.road = Road()
		self.renderFlag = renderFlag
		self.sensorPoints = np.zeros(shape=(self.car.sensors.shape[0],2), dtype=np.float32)

		if self.render:
			self.window = Tk()
			self.window_X = 500
			self.window_Y = 500
			self.canvas = Canvas(self.window, width=self.window_X, height=self.window_Y, borderwidth=0, highlightthickness=0, bg="black")
			self.canvas.pack()
			self.renderPoints = None
			self.renderQuality = 25 # points per curve
	
	def _computeSensorsValues(self):
		self.sensorPoints[:,0] = -self.car.sensors[:,0]*np.sin(self.car.angle+self.car.sensors[:,1])
		self.sensorPoints[:,1] = self.car.sensors[:,0]*np.cos(self.car.angle+self.car.sensors[:,1])
		self.car.sensorValues = np.ones(shape=self.car.sensors.shape[0],dtype=np.float32)

		for sensor_i in range(self.sensorPoints.shape[0]):
			#min_x = np.min([self.car.pos[0],self.car.pos[0]+self.sensorPoints[sensor_i,0]])
			#max_x = np.max([self.car.pos[0],self.car.pos[0]+self.sensorPoints[sensor_i,0]])
			min_y = np.min([self.car.pos[1],self.car.pos[1]+self.sensorPoints[sensor_i,1]])
			max_y = np.max([self.car.pos[1],self.car.pos[1]+self.sensorPoints[sensor_i,1]])
			alpha = self.car.pos
			beta = self.car.pos+self.sensorPoints[sensor_i]
			G = alpha[0]
			H = beta[0]-alpha[0]
			I = alpha[1]
			J = beta[1]-alpha[1]
			for segment_j in range(np.int32(min_y/self.road.segmentSize/2)*2,np.int32(max_y/self.road.segmentSize)+1):
				#print("SENSOR =",sensor_i,"- SEGMENT =",segment_j)
				if segment_j >= 0 and segment_j + 2 < self.road.nPoints and segment_j%2==0:
					for side in range(2):
						p0 = np.copy(self.road.points[segment_j])
						p0[0] = p0[0] + (0.5-1*side)*self.road.roadWidth
						p1 = np.copy(self.road.points[segment_j+1])
						p1[0] = p1[0] + (0.5-1*side)*self.road.roadWidth
						p2 = np.copy(self.road.points[segment_j+2])
						p2[0] = p2[0] + (0.5-1*side)*self.road.roadWidth
						A = p0[0]-2*p1[0]+p2[0]
						B = 2*p1[0]-2*p0[0]
						C = p0[0]
						D = p0[1]-2*p1[1]+p2[1]
						E = 2*p1[1]-2*p0[1]
						F = p0[1]
						L = A-H*D/J
						M = B-H*E/J
						N = C-G-H*(F-I)/J
						Delta = (M**2) - (4*L*N)
						if(Delta>0):
							t_curve = np.array([(-M-np.sqrt(Delta))/(2*L),(-M+np.sqrt(Delta))/(2*L)])
							t_sensr = D/J*(t_curve**2) + E/J*t_curve + (F-I)/J
							if(t_curve[0]>0 and t_curve[0]<1 and t_sensr[0]>0 and t_sensr[0]<1):
								if t_sensr[0]<self.car.sensorValues[sensor_i]:
									self.car.sensorValues[sensor_i] = t_sensr[0]
							if(t_curve[1]>0 and t_curve[1]<1 and t_sensr[1]>0 and t_sensr[1]<1):
								if t_sensr[1]<self.car.sensorValues[sensor_i]:
									self.car.sensorValues[sensor_i] = t_sensr[1]
	
	def _computeCollisions(self):
		x1 = -self.car.dim*np.sin(self.car.angle+np.pi/6)+self.car.pos[0]
		y1 = -self.car.dim*np.cos(self.car.angle+np.pi/6)+self.car.pos[1]
		x2 = -self.car.dim*np.sin(self.car.angle-np.pi/6)+self.car.pos[0]
		y2 = -self.car.dim*np.cos(self.car.angle-np.pi/6)+self.car.pos[1]
		x3 = -self.car.dim*np.sin(self.car.angle+np.pi+np.pi/6)+self.car.pos[0]
		y3 = -self.car.dim*np.cos(self.car.angle+np.pi+np.pi/6)+self.car.pos[1]
		x4 = -self.car.dim*np.sin(self.car.angle-np.pi-np.pi/6)+self.car.pos[0]
		y4 = -self.car.dim*np.cos(self.car.angle-np.pi-np.pi/6)+self.car.pos[1]
		min_y = np.min([y1,y2,y3,y4])
		max_y = np.max([y1,y2,y3,y4])
		lines = [
			[[x1,y1],[x2,y2]],
			[[x2,y2],[x3,y3]],
			[[x3,y3],[x4,y4]],
			[[x4,y4],[x1,y1]]
		]
		self.car.collision = False
		for points in lines:
			G = points[0][0]
			H = points[1][0]-points[0][0]
			I = points[0][1]
			J = points[1][1]-points[0][1]+0.0000001
			for segment_j in range(np.int32(min_y/self.road.segmentSize/2)*2,np.int32(max_y/self.road.segmentSize)+1):
				if segment_j >= 0 and segment_j + 2 < self.road.nPoints and segment_j%2==0:
					for side in range(2):
						p0 = np.copy(self.road.points[segment_j])
						p0[0] = p0[0] + (0.5-1*side)*self.road.roadWidth
						p1 = np.copy(self.road.points[segment_j+1])
						p1[0] = p1[0] + (0.5-1*side)*self.road.roadWidth
						p2 = np.copy(self.road.points[segment_j+2])
						p2[0] = p2[0] + (0.5-1*side)*self.road.roadWidth
						A = p0[0]-2*p1[0]+p2[0]
						B = 2*p1[0]-2*p0[0]
						C = p0[0]
						D = p0[1]-2*p1[1]+p2[1]
						E = 2*p1[1]-2*p0[1]
						F = p0[1]
						L = A-H*D/J
						M = B-H*E/J
						N = C-G-H*(F-I)/J
						Delta = (M**2) - (4*L*N)
						if(Delta>0):
							t_curve = np.array([(-M-np.sqrt(Delta))/(2*L),(-M+np.sqrt(Delta))/(2*L)])
							t_side = D/J*(t_curve**2) + E/J*t_curve + (F-I)/J
							if(t_curve[0]>0 and t_curve[0]<1 and t_side[0]>0 and t_side[0]<1):
								self.car.collision = True
							if(t_curve[1]>0 and t_curve[1]<1 and t_side[1]>0 and t_side[1]<1):
								self.car.collision = True
						if self.car.collision:
							break
				if self.car.collision:
					break
			if self.car.collision:
				break

	def _getStateFeatures(self):
		roadlen = self.road.segmentSize*(self.road.nPoints-1)
		features = [
			self.car.speed/self.car.maxSpeed,
			np.sin(self.car.angle),
			np.cos(self.car.angle),
			self.car.pos[1]/roadlen
		]
		state = np.append(self.car.sensorValues,features)
		return state

	def step(self,action):
		self.car.step(action)
		self._computeSensorsValues()
		self._computeCollisions()

		roadlen = self.road.segmentSize*(self.road.nPoints-1)
		done = (self.car.collision or (self.car.pos[1] > roadlen) or (self.car.pos[1] < 0))
		return [self._getStateFeatures(),(self.car.speed/self.car.maxSpeed),(self.car.collision or (self.car.pos[1] < 0)),done]
		
	def reset(self):
		self.car.reset()
		self.road.reset()
		self._computeSensorsValues()
		self._computeCollisions()

		if self.renderFlag:
			self.renderPoints = []
			for i in range(0,self.road.nPoints-2,2):
				p1 = self.road.points[i]
				p2 = self.road.points[i+1]
				p3 = self.road.points[i+2]
				for j in range(self.renderQuality):
					alpha = j/self.renderQuality
					a = np.array([lerp(p1[0],p2[0],alpha),lerp(p1[1],p2[1],alpha)])
					b = np.array([lerp(p2[0],p3[0],alpha),lerp(p2[1],p3[1],alpha)])
					P = np.array([lerp(a[0],b[0],alpha),lerp(a[1],b[1],alpha)])
					self.renderPoints.append(P)
			p1 = self.road.points[self.road.nPoints-3]
			p2 = self.road.points[self.road.nPoints-2]
			p3 = self.road.points[self.road.nPoints-1]
			a = np.array([lerp(p1[0],p2[0],1),lerp(p1[1],p2[1],1)])
			b = np.array([lerp(p2[0],p3[0],1),lerp(p2[1],p3[1],1)])
			P = np.array([lerp(a[0],b[0],1),lerp(a[1],b[1],1)])
			self.renderPoints.append(P)
			self.renderPoints = np.array(self.renderPoints)
		
		return self._getStateFeatures()

	def render(self):
		self.canvas.delete('all')
		x1 = -self.car.dim*np.sin(self.car.angle+np.pi/6)+self.window_X/2
		y1 = -self.car.dim*np.cos(self.car.angle+np.pi/6)+self.window_Y*3/4
		x2 = -self.car.dim*np.sin(self.car.angle-np.pi/6)+self.window_X/2
		y2 = -self.car.dim*np.cos(self.car.angle-np.pi/6)+self.window_Y*3/4
		x3 = -self.car.dim*np.sin(self.car.angle+np.pi+np.pi/6)+self.window_X/2
		y3 = -self.car.dim*np.cos(self.car.angle+np.pi+np.pi/6)+self.window_Y*3/4
		x4 = -self.car.dim*np.sin(self.car.angle-np.pi-np.pi/6)+self.window_X/2
		y4 = -self.car.dim*np.cos(self.car.angle-np.pi-np.pi/6)+self.window_Y*3/4
		for i in range(-5,6):
			self.canvas.create_line(self.road.segmentSize*i+self.window_X/2-self.car.pos[0], 0, self.road.segmentSize*i+self.window_X/2-self.car.pos[0], self.window_Y, fill='grey', width=1)
		for i in range(0,self.road.nPoints):
			self.canvas.create_line(0, self.window_Y*3/4-self.road.segmentSize*i+self.car.pos[1], self.window_X, self.window_Y*3/4-self.road.segmentSize*i+self.car.pos[1], fill='grey', width=1)
		self.canvas.create_line(self.road.points[0,0]-self.road.roadWidth/2-self.car.pos[0]+self.window_X/2, self.window_Y*3/4-(self.road.points[0,1]-self.car.pos[1]), self.road.points[0,0]+self.road.roadWidth/2-self.car.pos[0]+self.window_X/2, self.window_Y*3/4-(self.road.points[0,1]-self.car.pos[1]), fill='blue', width=2)
		self.canvas.create_line(self.road.points[self.road.nPoints-1,0]-self.road.roadWidth/2-self.car.pos[0]+self.window_X/2, self.window_Y*3/4-(self.road.points[self.road.nPoints-1,1]-self.car.pos[1]), self.road.points[self.road.nPoints-1,0]+self.road.roadWidth/2-self.car.pos[0]+self.window_X/2, self.window_Y*3/4-(self.road.points[self.road.nPoints-1,1]-self.car.pos[1]), fill='blue', width=2)
		
		for i in range(self.renderPoints.shape[0]-1):
			P = self.renderPoints[i]
			Q = self.renderPoints[i+1]
			self.canvas.create_line(
					P[0]-self.car.pos[0]+self.window_X/2-self.road.roadWidth/2,
					self.window_Y*3/4-(P[1]-self.car.pos[1]),
					Q[0]-self.car.pos[0]+self.window_X/2-self.road.roadWidth/2,
					self.window_Y*3/4-(Q[1]-self.car.pos[1]),
					fill='blue', width=2)
			self.canvas.create_line(
					P[0]-self.car.pos[0]+self.window_X/2+self.road.roadWidth/2,
					self.window_Y*3/4-(P[1]-self.car.pos[1]),
					Q[0]-self.car.pos[0]+self.window_X/2+self.road.roadWidth/2,
					self.window_Y*3/4-(Q[1]-self.car.pos[1]),
					fill='blue', width=2)
				
		self.canvas.create_polygon([x1,y1,x2,y2,x3,y3,x4,y4], outline='red', fill='black', width=2)

		for i in range(self.sensorPoints.shape[0]):
			Q = self.sensorPoints[i]
			R = self.car.sensorValues[i]*Q
			self.canvas.create_line(
					self.window_X/2,
					self.window_Y*3/4,
					R[0]+self.window_X/2,
					self.window_Y*3/4-R[1],
					fill='yellow', width=2)
			self.canvas.create_line(
					Q[0]+self.window_X/2,
					self.window_Y*3/4-Q[1],
					R[0]+self.window_X/2,
					self.window_Y*3/4-R[1],
					fill='red', width=2)
		
		self.window.update()
		time.sleep(0.02)

# Usage:
'''
env = DrivingEnv(renderFlag=True)
env.reset()
for i in range(250):
	_,_,_,done = env.step([1,np.pi/32])
	env.render()
	if done:
		break
'''