import sys
from six import StringIO
from gym import utils
from gym.envs.toy_text import discrete
import numpy as np

MAP = [
	"╔═════════╗",
	"║ · · · · ║",
	"║ · · · · ║",
	"║ · · · · ║",
	"║ · · · · ║",
	"║ · · · · ║",
	"╚═════════╝",
]


class GridworldEnv(discrete.DiscreteEnv):
	"""
	Gridworld
	
	Actions: 
	There are 4 discrete deterministic actions:
	- 0: move UP
	- 1: move DOWN
	- 2: move LEFT
	- 3: move RIGHT 
	
	Rewards: 
	There is a reward of -1 for each action.
	
	Rendering:
	- yellow: agent
	- blue 'G': destination
	"""
	metadata = {'render.modes': ['human', 'ansi']}

	def __init__(self, model=None):
		self.desc = np.asarray(MAP,dtype='U')

		nS = 625
		nR = 5
		nC = 5
		maxR = nR-1
		maxC = nC-1
		isd = np.zeros(nS)
		nA = 4
		P = {s : {a : [] for a in range(nA)} for s in range(nS)}

		self.w_pr = np.zeros(5,dtype=np.float32) if model is None else model[0]
		self.w_pc = np.zeros(5,dtype=np.float32) if model is None else model[1]
		self.w_gr = np.zeros(5,dtype=np.float32) if model is None else model[2]
		self.w_gc = np.zeros(5,dtype=np.float32) if model is None else model[3]

		for row in range(5):
			for col in range(5):
				for row_g in range(5):
					for col_g in range(5):
						state = self.encode(row, col, row_g, col_g)
						#if (row, col) != (row_g, col_g):
						prob = self.initialProb(row,col,row_g,col_g)
						isd[state] += prob
						for a in range(nA):
							# defaults
							newrow, newcol = row, col
							reward = -1
							done = False

							if a==0:
								newrow = max(row-1, 0)
							elif a==1:
								newrow = min(row+1, maxR)
							elif a==2:
								newcol = max(col-1, 0)
							elif a==3:
								newcol = min(col+1, maxC)
							
							if (newrow, newcol) == (row_g, col_g):
								done = True
							
							newstate = self.encode(newrow, newcol, row_g, col_g)
							P[state][a].append((1.0, newstate, reward, done))

		discrete.DiscreteEnv.__init__(self, nS, nA, P, isd)
	
	def initialProb(self, agentrow, agentcol, goalrow, goalcol):
		p_r = np.exp(self.w_pr[agentrow])/np.sum(np.exp(self.w_pr))
		p_c = np.exp(self.w_pc[agentcol])/np.sum(np.exp(self.w_pc))
		p_gr = np.exp(self.w_gr[goalrow])/np.sum(np.exp(self.w_gr))
		p_gc = np.exp(self.w_gc[goalcol])/np.sum(np.exp(self.w_gc))
		return p_r * p_c * p_gr * p_gc
	
	def dInitialProb_dw(self, agentrow, agentcol, goalrow, goalcol):
		dw_pr = -np.exp(self.w_pr)/np.sum(np.exp(self.w_pr))
		dw_pr[agentrow] += 1
		dw_pc = -np.exp(self.w_pc)/np.sum(np.exp(self.w_pc))
		dw_pc[agentcol] += 1
		dw_gr = -np.exp(self.w_gr)/np.sum(np.exp(self.w_gr))
		dw_gr[goalrow] += 1
		dw_gc = -np.exp(self.w_gc)/np.sum(np.exp(self.w_gc))
		dw_gc[goalcol] += 1
		return np.ravel(np.array([dw_pr,dw_pc,dw_gr,dw_gc]))

	def getInitialProbs(self):
		p_r = np.exp(self.w_pr)/np.sum(np.exp(self.w_pr))
		p_c = np.exp(self.w_pc)/np.sum(np.exp(self.w_pc))
		p_gr = np.exp(self.w_gr)/np.sum(np.exp(self.w_gr))
		p_gc = np.exp(self.w_gc)/np.sum(np.exp(self.w_gc))
		return [p_r,p_c,p_gr,p_gc]

	def encode(self, agentrow, agentcol, goalrow, goalcol):
		# (5) 5, 5, 5
		i = agentrow
		i *= 5
		i += agentcol
		i *= 5
		i += goalrow
		i *= 5
		i += goalcol
		return i

	def decode(self, i):
		out = []
		out.append(i % 5)
		i = i // 5
		out.append(i % 5)
		i = i // 5
		out.append(i % 5)
		i = i // 5
		out.append(i)
		assert 0 <= i < 5
		return reversed(out)

	def render(self, mode='human'):
		outfile = StringIO() if mode == 'ansi' else sys.stdout

		out = self.desc.copy().tolist()
		out = [[c for c in line] for line in out]
		agentrow, agentcol, goalrow, goalcol = self.decode(self.s)
		def ul(x): return "_" if x == " " else x
		out[1+agentrow][2*agentcol+1] = utils.colorize(' ', 'yellow', bold=True, highlight=True)
		out[1+goalrow][2*goalcol+1] = utils.colorize('G', 'blue', bold=True, highlight=True)

		if (agentrow, agentcol) == (goalrow, goalcol):
			out[1+agentrow][2*agentcol+1] = utils.colorize('G', 'yellow', bold=True, highlight=True)

		outfile.write("\n".join(["".join(row) for row in out])+"\n")
		if self.lastaction is not None:
			outfile.write("  ({})\n".format(["UP", "DOWN", "LEFT", "RIGHT"][self.lastaction]))
		else: outfile.write("\n")

		# No need to return anything for human
		if mode != 'human':
			return outfile
