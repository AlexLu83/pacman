# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        # print('SG: ', successorGameState, 'newpos:', newPos,'newfood:',newFood ,'ghost:',newGhostStates,'scaredtime:', newScaredTimes) 
        # print('successorGameState.getScore:', successorGameState.getScore())
        food_list = newFood.asList()
        ghost_list = successorGameState.getGhostPositions()
        f_dist = []
        g_dist = []
        val = 0
        found = False
        for x, y in food_list:
            f_dist.append(manhattanDistance((x,y), newPos))
        for x, y in ghost_list:
            g_dist.append(manhattanDistance((x,y), newPos))

        if len(currentGameState.getFood().asList()) >len(successorGameState.getFood().asList()):
            val += 800
        else:
            val += min(g_dist) - 10 * min(f_dist)     
        if min(g_dist) == 0:
            val -= 10000
        if currentGameState.getPacmanPosition() == newPos:
            val -= 100
        return  val

        
        "*** YOUR CODE HERE ***"

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        max_depth = self.depth
        eval_func = self.evaluationFunction
        max_val = float('-inf')
        result = None
        for action in gameState.getLegalActions(0):
            s = gameState.generateSuccessor(0, action)
            cur = value(s, 1, 0, max_depth, eval_func)
            if cur > max_val:
                max_val = cur
                result = action
        return result        

        # max_index = 0
        # legal_act = gameState.getLegalActions(0)
        # result = value(gameState, 0, 0, max_depth, eval_func)#edited
        # for index in range(len(legal_act)):
        #     if legal_act[index] == result:
        #         max_index = index

def value(state, agent_index, depth, max_depth, eval_func):
    if depth == max_depth or state.isWin() or state.isLose():
        return eval_func(state)
    # if agent_index == state.getNumAgents() - 1:
    #     depth = depth + 1
    if agent_index == 0:
        return max_value(state, agent_index, depth, max_depth, eval_func)
    else:
        return min_value(state, agent_index, depth, max_depth, eval_func)  

def max_value(state, agent_index, depth, max_depth, eval_func):
    max_val = float('-inf')
    for action in state.getLegalActions(agent_index):
        s = state.generateSuccessor(agent_index, action)
        max_val = max(max_val, value(s, 1, depth, max_depth, eval_func))
    return max_val        
        
def min_value(state, agent_index, depth, max_depth, eval_func):
    min_val = float('inf')
    for action in state.getLegalActions(agent_index):
        s = state.generateSuccessor(agent_index, action)
        if agent_index == state.getNumAgents() - 1:
            min_val = min(min_val, value(s, 0, depth + 1, max_depth, eval_func))
        else:
            min_val = min(min_val, value(s, agent_index + 1, depth, max_depth, eval_func))    

        # min_val = min(min_val, value(s, (agent_index + 1) % state.getNumAgents(), depth + 1, max_depth, eval_func))
        # min_val = min(min_val, value(s, agent_index, depth, max_depth, eval_func))
    return min_val


# def min_value(state, depth, agent_index, max_depth, eval_func):
#     min_val = float('inf')
#     if depth == max_depth or state.isWin() or state.isLose():
#         return eval_func(state)
#         # min_val = float('inf')
#     for action in state.getLegalActions(agent_index):
#             s = state.generateSuccessor(agent_index, action)#
#             min_val = min(min_val, max_value(s, depth + 1, max_depth, eval_func))
#     return min_val


# def max_value(state, depth, max_depth, eval_func):
#     max_val = float('-inf')
#     ghosts_num = state.getNumAgents()
#     if depth == max_depth or state.isWin() or state.isLose():
#         return eval_func(state)
#     for action in state.getLegalActions(0):
#         s = state.generateSuccessor(0, action)
#         for ghost in range(1, ghosts_num):
#             max_val = max(max_val, min_value(s, depth + 1, ghost, max_depth, eval_func))
#     return max_val                    


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        max_val = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        result = None
        for action in gameState.getLegalActions(0):
            s = gameState.generateSuccessor(0, action)
            cur = self.value(s, 1, 0, alpha, beta)
            if cur > max_val:
                max_val = cur
                result = action
            alpha = max(alpha, max_val)   
        return result        
        

    def value(self, state, agent_index, depth, alpha, beta):
        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        if agent_index == 0:
            return self.max_value(state, agent_index, depth, alpha, beta)
        else:
            return self.min_value(state, agent_index, depth, alpha, beta)

    def max_value(self, state, agent_index, depth, alpha, beta):
        max_val = float('-inf')
        for action in state.getLegalActions(agent_index):
            s = state.generateSuccessor(agent_index, action)
            max_val = max(max_val, self.value(s, 1, depth, alpha, beta))
            if max_val > beta:
                return max_val
            alpha = max(alpha, max_val)                       
        return max_val

    def min_value(self, state, agent_index, depth, alpha, beta):
        min_val = float('inf')
        for action in state.getLegalActions(agent_index):
            s = state.generateSuccessor(agent_index, action)
            if agent_index == state.getNumAgents() - 1:
                min_val = min(min_val, self.value(s, 0, depth + 1, alpha, beta))
            else:
                min_val = min(min_val, self.value(s, agent_index + 1, depth, alpha, beta))
            if min_val < alpha:
                return min_val             
            beta = min(beta, min_val)

        return min_val    
        
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        max_val = float('-inf')
        result = None
        for action in gameState.getLegalActions(0):
            s = gameState.generateSuccessor(0, action)
            cur = self.value(s, 1, 0)
            if cur > max_val:
                max_val = cur
                result = action
           
        return result

    def value(self, state, agent_index, depth):
        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        if agent_index == 0:
            return self.max_value(state, agent_index, depth)
        else:
            return self.exp_value(state, agent_index, depth)

    def max_value(self, state, agent_index, depth):
        max_val = float('-inf')
        for action in state.getLegalActions(agent_index):
            s = state.generateSuccessor(agent_index, action)
            max_val = max(max_val, self.value(s, 1, depth))                     
        return max_val

    def exp_value(self, state, agent_index, depth):
        exp_val = 0
        actions = state.getLegalActions(agent_index)
        # p = 1/len(actions)
        for action in actions:
            s = state.generateSuccessor(agent_index, action)
            if agent_index == state.getNumAgents() - 1:
                exp_val += self.value(s, 0, depth + 1) 
            else:
                exp_val += self.value(s, agent_index + 1, depth)

        return exp_val                    
        
        
                

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    if(currentGameState.isLose()):
        return float('-inf')
    if(currentGameState.isWin()):
        return float('inf')    
    pos = currentGameState.getPacmanPosition()
    food_list = currentGameState.getFood().asList()
    ghost_list = currentGameState.getGhostPositions()
    capsule_list = currentGameState.getCapsules()
    f_dist = []
    g_dist = []
    c_dist = []
    val = 0
    for x, y in ghost_list:
        g_dist.append(manhattanDistance((x,y), pos))
    for x, y in food_list:
        f_dist.append(manhattanDistance((x,y), pos))
    for x, y in capsule_list:
        c_dist.append(manhattanDistance((x,y), pos))

    if pos in food_list:
        val += 100
    if not len(f_dist) == 0:
        val -= 2 * min(f_dist) - max(f_dist)
    if pos in ghost_list:
        val -= 3000
    elif not len(ghost_list) == 0:
        val += 1.5*min(g_dist) + max(g_dist)
    if pos in capsule_list:
        val += 200
    elif not len(capsule_list) == 0:
        val = 20/min(c_dist)        

    return val + currentGameState.getScore()      

    
    # successorGameState = currentGameState.generatePacmanSuccessor(action)
    # newPos = successorGameState.getPacmanPosition()
    # newFood = successorGameState.getFood()
    # newGhostStates = successorGameState.getGhostStates()
    # newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    #     # print('SG: ', successorGameState, 'newpos:', newPos,'newfood:',newFood ,'ghost:',newGhostStates,'scaredtime:', newScaredTimes) 
    #     # print('successorGameState.getScore:', successorGameState.getScore())
    # food_list = newFood.asList()
    # ghost_list = successorGameState.getGhostPositions()
    # f_dist = []
    # g_dist = []
    # val = 0
    # found = False
    # for x, y in food_list:
    #     f_dist.append(manhattanDistance((x,y), newPos))
    # for x, y in ghost_list:
    #     g_dist.append(manhattanDistance((x,y), newPos))

    # if len(currentGameState.getFood().asList()) >len(successorGameState.getFood().asList()):
    #     val += 800
    # else:
    #     val += min(g_dist) - 10 * min(f_dist)     
    # if min(g_dist) == 0:
    #     val -= 10000
    # if currentGameState.getPacmanPosition() == newPos:
    #     val -= 100
    # return  val
    

# Abbreviation
better = betterEvaluationFunction
