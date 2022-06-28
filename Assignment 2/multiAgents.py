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


from cmath import inf
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

        eval_score = float(0) #return value : evaluated value of each state
        FoodList = currentGameState.getFood().asList() #current (not new!) coordinates of foods in the maze
        ghostNum = len(newGhostStates) #number of ghost agents

        #1. finding food
        if (newPos in FoodList) == True: eval_score += 1 #If the successor state has food, it will be good

        minFoodDist = float("inf") #distance from the pacman to the closest food
        for foodPos in FoodList:
            minFoodDist = min(minFoodDist, manhattanDistance(newPos, foodPos)) 

        if minFoodDist != 0: eval_score += (float(1) / minFoodDist) 
        #If successor state has food => minFoodDist will be 0 (It will cause DivisionByZero error)
        #smaller minFoodDist => higher probability to eat it => better! 
        #So we add the inverse of minFoodDist to the evaluation score

        for ghostIdx in range(ghostNum): #for the each ghost

            ghostPos = newGhostStates[ghostIdx].getPosition() #position of ith ghost
            ghostDist = manhattanDistance(newPos, ghostPos) #distance from the pacman to ith ghost (using pre-defined function in util.py)

            if ghostDist <= newScaredTimes[ghostIdx]: eval_score += ghostDist #if the time that ghost is scared is bigger than distance, we can eat the ghost! 
            if ghostDist <= 2: eval_score -= ghostDist #we have to run away from the ghost (if the manhattan distance is smaller than 2, in the next situation pacman could be dead)

        return eval_score

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
        legalMoves = gameState.getLegalActions(0) #legal actions that pacman can do

        targetIdx = 0 #index for return
        maxValue = -float("inf") #best value among values of possible actions

        for actionIdx in range(len(legalMoves)):
            successor = gameState.generateSuccessor(0, legalMoves[actionIdx]) #pacman's successor for (actionIdx)th action in legalMoves
            value = self.value(successor, 1, 0) #next agent is MIN (ghost) => agentIndex is 1 / initial depth is 0
            if value > maxValue: #better value for action
                targetIdx = actionIdx
                maxValue = value

        return legalMoves[targetIdx] #return the action that has the best value for the pacman agent

    def max_value(self, gameState, agentIndex, depth):
        legalMoves = gameState.getLegalActions(agentIndex)

        value = -float("inf") 
        for action in legalMoves:
            successor = gameState.generateSuccessor(agentIndex, action) #successor for each action
            value = max(value, self.value(successor, 1, depth)) #maximum value among successors : next agent is MIN(ghost) => agentIndex is 1

        return value

    def min_value(self, gameState, agentIndex, depth):
        legalMoves = gameState.getLegalActions(agentIndex)

        value = float("inf")
        for action in legalMoves:
            successor = gameState.generateSuccessor(agentIndex, action) #successor for each action
            #we can have multiple MIN agents(ghosts) : their index - 1 ~ gameState.getNumAgents() - 1
            if agentIndex == gameState.getNumAgents() - 1: #if we checked the last ghost
                value = min(value, self.value(successor, 0, depth+1)) #next agent is MAX(pacman) => agentIndex is 0 / depth should be increased
            else:
                value = min(value, self.value(successor, agentIndex+1, depth)) #next agent is MIN (other ghost) : (agentIndex)th ghost => we have to check them

        return value

    def value(self, gameState, agentIndex, depth):
        if self.depth == depth or gameState.isWin() or gameState.isLose(): #If we reached the pre-defined depth limit or the game is ended (win or lose)
            return self.evaluationFunction(gameState) #we don't have to do deeper search : just return the evaluation function value
        if agentIndex == 0: #MAX (pacman)
            return self.max_value(gameState, agentIndex, depth)
        else: #MIN (ghost)
            return self.min_value(gameState, agentIndex, depth)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        #Similar to MiniMaxAgent, but we have alpha and beta 
        #Based on pseudocode in instruction pdf

        legalMoves = gameState.getLegalActions(0) #legal actions that pacman can do

        targetIdx = 0
        a = -float("inf") #alpha : initial value is -infinity (we keep increasing it)
        b = float("inf") #beta : initial value is infinity (we keep decreasing it)
        maxValue = -float("inf")

        for actionIdx in range(len(legalMoves)):
            successor = gameState.generateSuccessor(0, legalMoves[actionIdx])
            value = self.value(successor, 1, 0, a, b)

            if value > maxValue: #better value for action
                targetIdx = actionIdx
                maxValue = value
                a = value #setting alpha : same as a = max(a, value) 
                
        return legalMoves[targetIdx] #best action for pacman

    def max_value(self, gameState, agentIndex, depth, a, b):
        legalMoves = gameState.getLegalActions(agentIndex)

        value = -float("inf")
        for action in legalMoves:
            successor = gameState.generateSuccessor(agentIndex, action)

            value = max(value, self.value(successor, 1, depth, a, b))
            if value > b: return value #beta pruning : if the value is bigger than beta, we don't have to traverse other successors
            a = max(a, value) #setting alpha to the larger one

        return value

    def min_value(self, gameState, agentIndex, depth, a, b):
        legalMoves = gameState.getLegalActions(agentIndex)

        value = float("inf")
        for action in legalMoves:
            successor = gameState.generateSuccessor(agentIndex, action)
            if agentIndex == gameState.getNumAgents() - 1:
                value = min(value, self.value(successor, 0, depth+1, a, b))
            else:
                value = min(value, self.value(successor, agentIndex+1, depth, a, b)) 
            if value < a: return value #alpha pruning : if the value is smaller than alpha, we don't have to traverse other successors
            b = min(b, value) #setting beta to the smaller one

        return value

    def value(self, gameState, agentIndex, depth, a, b):
        if self.depth == depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        if agentIndex == 0: 
            return self.max_value(gameState, agentIndex, depth, a, b)
        else: 
            return self.min_value(gameState, agentIndex, depth, a, b)

    

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
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
