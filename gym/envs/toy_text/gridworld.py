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

    def __init__(self, changeProb=0.0, targetRow=-1, targetCol=-1):
        self.desc = np.asarray(MAP,dtype='U')

        nS = 625
        nR = 5
        nC = 5
        maxR = nR-1
        maxC = nC-1
        isd = np.zeros(nS)
        nA = 4
        P = {s : {a : [] for a in range(nA)} for s in range(nS)}

        if changeProb<0:
            changeProb = 0
        if changeProb>1:
            changeProb = 1
        
        if changeProb>0:
            if targetRow==-1 and targetCol==-1:
                print("Error in Gridworld::__init__, no target row or column specified.")
                changeProb = 0
            if targetRow!=-1 and targetCol!=-1:
                print("Error in Gridworld::__init__, you can only specify one row or one column.")
                changeProb = 0
        
        if changeProb>0:
            if targetCol!=-1:
                targetCol = np.int32(np.clip(targetCol,0,maxC))
            if targetRow!=-1:
                targetRow = np.int32(np.clip(targetRow,0,maxR))

        for row in range(5):
            for col in range(5):
                for row_g in range(5):
                    for col_g in range(5):
                        state = self.encode(row, col, row_g, col_g)
                        if (row, col) != (row_g, col_g):
                            if changeProb==0:
                                prob = 1/25
                            else:
                                if targetCol!=-1:
                                    if col_g==targetCol:
                                        prob = changeProb/5
                                    else:
                                        prob = (1-changeProb)/20
                                elif targetRow!=-1:
                                    if row_g==targetRow:
                                        prob = changeProb/5
                                    else:
                                        prob = (1-changeProb)/20
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
