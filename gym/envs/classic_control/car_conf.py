import numpy as np
from scipy.stats import multivariate_normal
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
			#[self.sensorDist,np.pi/12],
			#[self.sensorDist,0],
			#[self.sensorDist,-np.pi/12],
			[self.sensorDist,-np.pi/6],
			[self.sensorDist,-np.pi/4]
		],dtype=np.float32)
		self.sensorValues = np.ones(shape=self.sensors.shape[0],dtype=np.float32)
	
	def reset(self):
		self.pos = np.array([0,self.dim],dtype=np.float32)
		self.angle = np.float32(0)
		self.speed = np.float32(0)

	def _step(self,pos,angle,speed,action):
		acceleration = np.clip(action[0],-1,1)*self.maxAccel
		steering = np.clip(action[1],-1,1)
		newspeed = np.clip(speed + acceleration,-self.maxSpeed,self.maxSpeed)

		dtheta_coeff = np.float32(0)
		if(abs(newspeed/self.maxSpeed)<0.1):
			dtheta_coeff = 10.0*abs(newspeed/self.maxSpeed)
		else:
			dtheta_coeff = 1.0-(abs(newspeed/self.maxSpeed)-0.1)*0.3
		dtheta = steering*dtheta_coeff*self.maxAngularVel
		newangle = angle + dtheta

		newpos = np.array([pos[0]-newspeed*np.sin(newangle),pos[1]+newspeed*np.cos(newangle)],dtype=np.float32)
		return [newpos,newangle,newspeed]

	def step(self, action): # force [-1,1], steering [-1,1]
		newpos,newangle,newspeed = self._step(self.pos,self.angle,self.speed,action)
		self.pos[0] = newpos[0]
		self.pos[1] = newpos[1]
		self.angle = newangle
		self.speed = newspeed
	
	def printInfo(self):
		print("CAR: Pos =",np.around(self.pos,decimals=2),". Speed =",np.around(self.speed,decimals=2),". Angle =",np.around(self.angle/(np.pi),decimals=2),"π.")

class Road:
	def __init__(self,model_w):
		self.segmentSize = 500
		self.roadWidth = 100

		self.points = np.zeros(shape=(3,2),dtype=np.float32)
		self.points[0] = np.array([0,0])
		self.points[1] = np.array([0,self.segmentSize/2],dtype=np.float32)
		self.points[2] = np.array([0,self.segmentSize])

		#self.mean = 0
		#self.stddev = 50
		self.model_w = model_w
	
	def reset(self):
		#self.points[1][0] = np.random.normal(self.mean,self.stddev)
		self.points[1][0] = self.model_w

class ConfDrivingEnv:
	def __init__(self, model_w, renderFlag=False):
		self.car = Car()
		self.road = Road(model_w)
		self.renderFlag = renderFlag
		self.sensorPoints = np.zeros(shape=(self.car.sensors.shape[0],2), dtype=np.float32)

		self.sensorNoiseVariance = 0.1
		self.model_w = model_w

		if self.renderFlag:
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
			alpha = self.car.pos
			beta = self.car.pos+self.sensorPoints[sensor_i]
			G = alpha[0]
			H = beta[0]-alpha[0]
			I = alpha[1]
			J = beta[1]-alpha[1]
			for side in range(2):
				p0 = np.copy(self.road.points[0])
				p0[0] = p0[0] + (0.5-1*side)*self.road.roadWidth
				p1 = np.copy(self.road.points[1])
				p1[0] = p1[0] + (0.5-1*side)*self.road.roadWidth
				p2 = np.copy(self.road.points[2])
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
			self.car.sensorValues[sensor_i] += np.random.normal(loc=0.0,scale=self.sensorNoiseVariance)
			self.car.sensorValues[sensor_i] = np.clip(self.car.sensorValues[sensor_i],0,1)
	
	def _getSensorValues(self,pos,angle,model_w,get_dMean_dw=False):
		sensorPoints = np.zeros(shape=(self.car.sensors.shape[0],2), dtype=np.float32)
		sensorPoints[:,0] = -self.car.sensors[:,0]*np.sin(angle+self.car.sensors[:,1])
		sensorPoints[:,1] = self.car.sensors[:,0]*np.cos(angle+self.car.sensors[:,1])
		sensorValues = np.ones(shape=self.car.sensors.shape[0],dtype=np.float32)
		if get_dMean_dw:
			dMean_dw = np.zeros(shape=self.car.sensors.shape[0],dtype=np.float32)

		for sensor_i in range(sensorPoints.shape[0]):
			alpha = pos
			beta = pos+self.sensorPoints[sensor_i]
			G = alpha[0]
			H = beta[0]-alpha[0]
			I = alpha[1]
			J = beta[1]-alpha[1]
			for side in range(2):
				p0 = np.copy(self.road.points[0])
				p0[0] = p0[0] + (0.5-1*side)*self.road.roadWidth
				p1 = np.copy(self.road.points[1])
				p1[0] = model_w + (0.5-1*side)*self.road.roadWidth
				p2 = np.copy(self.road.points[2])
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
						if t_sensr[0]<sensorValues[sensor_i]:
							sensorValues[sensor_i] = t_sensr[0]
							if get_dMean_dw:
								dMean_dw[sensor_i] = -2*E/J * (N/L/np.sqrt(Delta) + (np.sqrt(Delta)+M)/(2*L**2))
					if(t_curve[1]>0 and t_curve[1]<1 and t_sensr[1]>0 and t_sensr[1]<1):
						if t_sensr[1]<sensorValues[sensor_i]:
							sensorValues[sensor_i] = t_sensr[1]
							if get_dMean_dw:
								dMean_dw[sensor_i] = -2*E/J * (-N/L/np.sqrt(Delta) - (np.sqrt(Delta)-M)/(2*L**2))
			sensorValues[sensor_i] = np.clip(sensorValues[sensor_i],0,1)
		return (sensorValues if not get_dMean_dw else [sensorValues,dMean_dw])

	def _computeCollisions(self):
		x1 = -self.car.dim*np.sin(self.car.angle+np.pi/6)+self.car.pos[0]
		y1 = -self.car.dim*np.cos(self.car.angle+np.pi/6)+self.car.pos[1]
		x2 = -self.car.dim*np.sin(self.car.angle-np.pi/6)+self.car.pos[0]
		y2 = -self.car.dim*np.cos(self.car.angle-np.pi/6)+self.car.pos[1]
		x3 = -self.car.dim*np.sin(self.car.angle+np.pi+np.pi/6)+self.car.pos[0]
		y3 = -self.car.dim*np.cos(self.car.angle+np.pi+np.pi/6)+self.car.pos[1]
		x4 = -self.car.dim*np.sin(self.car.angle-np.pi-np.pi/6)+self.car.pos[0]
		y4 = -self.car.dim*np.cos(self.car.angle-np.pi-np.pi/6)+self.car.pos[1]
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
			for side in range(2):
				p0 = np.copy(self.road.points[0])
				p0[0] = p0[0] + (0.5-1*side)*self.road.roadWidth
				p1 = np.copy(self.road.points[1])
				p1[0] = p1[0] + (0.5-1*side)*self.road.roadWidth
				p2 = np.copy(self.road.points[2])
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

	def _getStateFeatures(self):
		features = [
			self.car.speed/self.car.maxSpeed
			#np.sin(self.car.angle),
			#np.cos(self.car.angle),
			#self.car.pos[0]/self.road.roadWidth,
			#self.car.pos[1]/self.road.segmentSize
		]
		state = np.append(self.car.sensorValues,features)
		return state

	def _getRealState(self,s):
		pos = np.array([s[-2]*self.road.roadWidth,s[-1]*self.road.segmentSize],dtype=np.float32)
		angle = np.arctan2(s[-4],s[-3])
		speed = s[-5]*self.car.maxSpeed
		return [pos,angle,speed]
	
	def p_model(self,newSensorValues,s,a,model_w):
		pos,angle,speed = self._getRealState(s)
		newpos,newangle,_ = self.car._step(pos,angle,speed,a)
		newSensorsValuesMean = self._getSensorValues(newpos,newangle,model_w)
		k = newSensorsValuesMean.size # num sensors
		return multivariate_normal.pdf(newSensorValues,mean=newSensorsValuesMean,cov=self.sensorNoiseVariance*np.eye(k))

	def grad_log_p_model(self,newSensorValues,s,a,model_w):
		pos,angle,speed = self._getRealState(s)
		newpos,newangle,_ = self.car._step(pos,angle,speed,a)
		newSensorsValuesMean,dmean_dw = self._getSensorValues(newpos,newangle,model_w,get_dMean_dw=True)
		k = newSensorsValuesMean.size # num sensors
		df_dmean = np.dot(0.1*np.eye(k),newSensorValues-newSensorsValuesMean)
		grad = np.dot(df_dmean,dmean_dw)
		return grad

	def step(self,action):
		self.car.step(action)
		self._computeSensorsValues()
		self._computeCollisions()

		roadlen = self.road.segmentSize
		done = (self.car.collision or (self.car.pos[1] > roadlen) or (self.car.pos[1] < 0))
		return [self._getStateFeatures(),(self.car.speed/self.car.maxSpeed),(self.car.collision or (self.car.pos[1] < 0)),done]
		
	def reset(self):
		self.car.reset()
		self.road.reset()
		self._computeSensorsValues()
		self._computeCollisions()

		if self.renderFlag:
			self.renderPoints = []
			p0 = self.road.points[0]
			p1 = self.road.points[1]
			p2 = self.road.points[2]
			for j in range(self.renderQuality):
				alpha = j/(self.renderQuality-1)
				a = np.array([lerp(p0[0],p1[0],alpha),lerp(p0[1],p1[1],alpha)])
				b = np.array([lerp(p1[0],p2[0],alpha),lerp(p1[1],p2[1],alpha)])
				P = np.array([lerp(a[0],b[0],alpha),lerp(a[1],b[1],alpha)])
				self.renderPoints.append(P)
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
		for i in range(0,3):
			self.canvas.create_line(0, self.window_Y*3/4-self.road.segmentSize*i+self.car.pos[1], self.window_X, self.window_Y*3/4-self.road.segmentSize*i+self.car.pos[1], fill='grey', width=1)
		self.canvas.create_line(self.road.points[0,0]-self.road.roadWidth/2-self.car.pos[0]+self.window_X/2, self.window_Y*3/4-(self.road.points[0,1]-self.car.pos[1]), self.road.points[0,0]+self.road.roadWidth/2-self.car.pos[0]+self.window_X/2, self.window_Y*3/4-(self.road.points[0,1]-self.car.pos[1]), fill='blue', width=2)
		self.canvas.create_line(self.road.points[2,0]-self.road.roadWidth/2-self.car.pos[0]+self.window_X/2, self.window_Y*3/4-(self.road.points[2,1]-self.car.pos[1]), self.road.points[2,0]+self.road.roadWidth/2-self.car.pos[0]+self.window_X/2, self.window_Y*3/4-(self.road.points[2,1]-self.car.pos[1]), fill='blue', width=2)
		
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