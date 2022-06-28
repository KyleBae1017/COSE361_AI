# myTeam.py
# ---------
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


from captureAgents import CaptureAgent
import random, util

################# 
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffenseMiniMaxAgent', second = 'DefenseMiniMaxAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class CaptureMiniMaxAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):

    CaptureAgent.registerInitialState(self, gameState) 
    self.start = gameState.getAgentPosition(self.index) #start position of agent
    self.depthLimit = 2 #depth limit for search tree
    self.agentList = [self.index] + self.getOpponents(gameState) #agent itself + 2 opponent agents

  def chooseAction(self, gameState):
      
    actionList = gameState.getLegalActions(self.index)
    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actionList:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start, pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    alpha = -float("inf")
    beta = float("inf")

    successorList = [gameState.generateSuccessor(self.index, action) for action in actionList]
    valueList = [self.value(successor, 0, 0, alpha, beta) for successor in successorList] 

    #print(valueList, self.index)

    maxValue = max(valueList)
    bestActionList = [a for a, v in zip(actionList, valueList) if v == maxValue]
    #print(bestActionList, self.index)

    return random.choice(bestActionList)

  def value(self, gameState, agentIndex, currDepth, alpha, beta): #calculating value of nodes using alpha-beta pruning
    if currDepth == self.depthLimit or gameState.isOver(): #reached to limit of the depth
      return self.evaluate(gameState)
    if self.agentList[agentIndex] == self.index: #player's agent
      return self.max_value(gameState, agentIndex, currDepth, alpha, beta)
    else: #opponent's agent
      return self.min_value(gameState, agentIndex, currDepth, alpha, beta)

  def max_value(self, gameState, agentIndex, depth, alpha, beta):

        currAgentIdx = self.agentList[agentIndex]
        #print("max", currAgentIdx)
        legalMoves = gameState.getLegalActions(currAgentIdx)

        value = -float("inf")
        for action in legalMoves:
            successor = gameState.generateSuccessor(currAgentIdx, action)
            value = max(value, self.value(successor, 1, depth, alpha, beta))
            if value > beta: return value  #beta pruning
            alpha = max(alpha, value)

        return value

  def min_value(self, gameState, agentIndex, depth, alpha, beta):

      currAgentIdx = self.agentList[agentIndex]
      #print("min", currAgentIdx)
      legalMoves = gameState.getLegalActions(currAgentIdx)

      value = float("inf")
      for action in legalMoves:
          successor = gameState.generateSuccessor(currAgentIdx, action)
          if agentIndex == len(self.agentList) - 1: #last agent
            value = min(value, self.value(successor, 0, depth+1, alpha, beta)) #go to the next depth
          else:
            value = min(value, self.value(successor, agentIndex+1, depth, alpha, beta)) #go to the next agent
          if value < alpha: return value #alpha pruning
          beta = min(beta, value) 

      return value

  def getSuccessor(self, gameState, action):

    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != util.nearestPoint(pos):
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState):
    features = self.getFeatures(gameState)
    weights = self.getWeights(gameState)
    return features * weights

  def getFeatures(self, gameState):
      features = util.Counter()

      features['currentScore'] = gameState.getScore()

      return features

  def getWeights(self, gameState):
      return {'currentScore': 1.0}

class OffenseMiniMaxAgent(CaptureMiniMaxAgent):

  def getFeatures(self, gameState):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    foodList = self.getFood(gameState).asList()
    features['foodNum'] = -len(foodList)

    capsuleList = [capsulePos for capsulePos in gameState.getCapsules() if capsulePos[0] > self.getFood(gameState).width]
    features['capsuleNum'] = -len(capsuleList)
    
    currentState = gameState.getAgentState(self.index)
    currentPos = currentState.getPosition()

    features['score'] = self.getScore(gameState)

    if len(foodList) > 0: 
      distanceToFood = min([self.getMazeDistance(currentPos, foodPos) for foodPos in foodList])
      features['1/distanceToFood'] = 1/distanceToFood

    enemyList = [gameState.getAgentState(idx) for idx in self.getOpponents(gameState)]
    ghostEnemyList = [enemy for enemy in enemyList if not enemy.isPacman and enemy.getPosition() != None]

    nonScaredGhostList = [ghost for ghost in ghostEnemyList if ghost.scaredTimer == 0]
    scaredGhostList = [ghost for ghost in ghostEnemyList if ghost.scaredTimer != 0]
    features['scaredGhostNum'] = -len(scaredGhostList)

    features['distanceTononScaredGhost'] = 80
    if len(nonScaredGhostList) > 0:
      distList = [self.getMazeDistance(currentPos, ghost.getPosition()) for ghost in nonScaredGhostList]
      features['distanceTononScaredGhost'] = min(distList)

    features['1/distanceToScaredGhost'] = 0
    if len(scaredGhostList) > 0:
      distList = [self.getMazeDistance(currentPos, ghost.getPosition()) for ghost in scaredGhostList]
      features['1/distanceToScaredGhost'] = 1/min(distList)
      
    features['1/distanceToHome'] = 0
    if currentState.numCarrying > 0: #number of food that agent eats
        distanceToHome = self.getMazeDistance(currentPos, self.start)
        if distanceToHome > 0: features['1/distanceToHome'] = 1/distanceToHome * currentState.numCarrying

    features['1/distanceToCapsule'] = 0
    if len(capsuleList) > 0:
        distList = [self.getMazeDistance(currentPos, capsulePos) for capsulePos in capsuleList]
        if min(distList) == 0:
            features['1/distanceToCapsule'] = 5000
        else: features['1/distanceToCapsule'] = 1/min(distList)

    actionList = gameState.getLegalActions(self.index)
    if len(actionList) <= 2 and features['ghostDistance'] < 5: features['isTrapped'] = -1
    
    return features

  def getWeights(self, gameState):

    return {'foodNum': 1000, 'capsuleNum': 700, 'score': 10000, '1/distanceToFood': 600, 'scaredGhostNum': 200, 'distanceTononScaredGhost': 0.1, '1/distanceToScaredGhost': 150,  '1/distanceToHome': 70000, '1/distanceToCapsule': 300, 'isTrapped' : 1000000}

class DefenseMiniMaxAgent(CaptureMiniMaxAgent):

  def getFeatures(self, gameState):

    features = util.Counter()
    foodList = self.getFoodYouAreDefending(gameState).asList()
    capsuleList = [capsulePos for capsulePos in gameState.getCapsules() if capsulePos[0] <= self.getFood(gameState).width]

    currentState = gameState.getAgentState(self.index)
    currentPos = currentState.getPosition()

    features['score'] = self.getScore(gameState)

    features['foodNum'] = len(foodList)

    enemyList = [gameState.getAgentState(idx) for idx in self.getOpponents(gameState)]
    pacmanEnemyList = [enemy for enemy in enemyList if enemy.isPacman and enemy.getPosition() != None]

    features['1/distanceToInvader'] = 150
    if len(pacmanEnemyList) > 0:
      distList = [self.getMazeDistance(currentPos, invader.getPosition()) for invader in pacmanEnemyList]
      features['1/distanceToInvader'] = 1/min(distList)
      features['invaderNum'] = -len(pacmanEnemyList)

    features['1/distanceToCapsule'] = 0
    if len(capsuleList) > 0:
        distList = [self.getMazeDistance(currentPos, capsulePos) for capsulePos in capsuleList]
        if min(distList) == 0:
            features['1/distanceToCapsule'] = 1000
        else: features['1/distanceToCapsule'] = 1/min(distList)

    return features
    
  def getWeights(self, gameState):
      return {'score': 20, 'foodNum': 3000, '1/distanceToInvader': 10000, 'invaderNum': 100000, '1/distanceToCapsule': 30}







