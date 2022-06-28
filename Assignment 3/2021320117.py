# myAgents.py
# ---------------
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

from argparse import Action
from game import Agent
from searchProblems import PositionSearchProblem

from util import Queue, PriorityQueue

"""
IMPORTANT
`agent` defines which agent you will use. By default, it is set to ClosestDotAgent,
but when you're ready to test your own agent, replace it with MyAgent
"""

def createAgents(num_pacmen, agent='MyAgent'):
    return [eval(agent)(index=i) for i in range(num_pacmen)]

def BFS(problem, currentGoalList):

    visited = {} #visited[prev] = curr : used for tracking the information for child and parent node
    direction = {} #direction[position] = direction : used for saving the information for actions betwenn nodes
    actions = [] #final return : total actions for moving to target position

    fringe = Queue() #BFS fringe : FIFO queue
    numCompute = 0 #number of computation
    goalPos = 0 #target goal position

    fringe.push((problem.getStartState(), 'None')) #(position, direction)

    while not fringe.isEmpty():

        numCompute += 1
        currPos, currDir = fringe.pop()
        direction[currPos] = currDir 
        
        if problem.isGoalState(currPos): 
            if currPos not in currentGoalList: #We have to choose goal position not already in the current goal lists(other pacman will visit there)
                goalPos = currPos
                break

        for pos, move, cost in problem.getSuccessors(currPos):
            if pos not in visited.keys() and pos not in direction.keys():
                visited[pos] = currPos
                fringe.push((pos, move))

        if numCompute > 601: return [['Stop']] #computation limit

    if goalPos == 0: return [['Stop']] #failure to find goal position

    curr = goalPos

    while(curr in visited.keys()): #backtracking the actions for current position to goal position
        prev = visited[curr]
        actions.append(direction[curr])
        curr = prev

    return [actions, goalPos] #[[list of actions], (goal_x, goal_y)]

def reducedBFS(problem): #exactly same as BFS

    visited = {}
    direction = {}
    actions = []

    fringe = Queue()
    numCompute = 0
    goalPos = 0

    fringe.push((problem.getStartState(), 'None'))

    while not fringe.isEmpty():

        numCompute += 1
        currPos, currDir = fringe.pop()
        direction[currPos] = currDir

        if problem.isGoalState(currPos): #but in this function we don't have to compare with current goal list
            goalPos = currPos
            break

        for pos, move, cost in problem.getSuccessors(currPos):
            if pos not in visited.keys() and pos not in direction.keys():
                visited[pos] = currPos
                fringe.push((pos, move))

        if numCompute > 18: return [['Stop']] 

    if goalPos == 0: return [['Stop']]

    curr = goalPos

    while(curr in visited.keys()):
        prev = visited[curr]
        actions.append(direction[curr])
        curr = prev

    return [actions, goalPos]


class MyAgent(Agent):
    """
    Implementation of your agent.
    """
    GoalList = [set()] #list for future goals : pacman will visting this positions

    def getAction(self, state):
        """
        Returns the next action the agent will take
        """

        idx = self.index #agent index
        actionList = self.actions #list for actions

        if idx not in actionList.keys(): #first assignment for dictionary

            actionList[idx] = self.findPathToClosestDot(state)    
            #actionList[idx] will return [[direction1, direction2, ...], goalPosition]
            if actionList[idx][0] == ['Stop']: return 'Stop' #Directions.STOP
            self.GoalList[0].add(actionList[idx][1])
            
        if actionList[idx][0] == ['Stop']: return 'Stop'  

        if len(actionList[idx][0]) == 0: #no more remaining moves

            self.GoalList[0].discard(actionList[idx][1]) #discard from future goal position list : we will now visit here!
            actionList[idx] = self.findPathToClosestDot(state)

            if actionList[idx][0] == ['Stop']: return 'Stop'
            self.GoalList[0].add(actionList[idx][1])

        if actionList[idx][0] == ['Stop']: return 'Stop'

        return actionList[idx][0].pop() #first action for list of actions
        
    def initialize(self):
        """
        Intialize anything you want to here. This function is called
        when the agent is first created. If you don't need to use it, then
        leave it blank
        """
        self.actions = {}
        self.GoalList[0] = set()
        
    def findPathToClosestDot(self, gameState):

        if gameState.getNumFood() < len(self.GoalList[0]): 
            return reducedBFS(AnyFoodSearchProblem(gameState, self.index))

        return BFS(AnyFoodSearchProblem(gameState, self.index), self.GoalList[0])

"""
Put any other SearchProblems or search methods below. You may also import classes/methods in
search.py and searchProblems.py. (ClosestDotAgent as an example below)
"""

class ClosestDotAgent(Agent):

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from
        gameState.
        """
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition(self.index)
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState, self.index)


        "*** YOUR CODE HERE ***"

        pacmanCurrent = [problem.getStartState(), [], 0]
        visitedPosition = set()
        # visitedPosition.add(problem.getStartState())
        fringe = PriorityQueue()
        fringe.push(pacmanCurrent, pacmanCurrent[2])
        while not fringe.isEmpty():
            pacmanCurrent = fringe.pop()
            if pacmanCurrent[0] in visitedPosition:
                continue
            else:
                visitedPosition.add(pacmanCurrent[0])
            if problem.isGoalState(pacmanCurrent[0]):
                return pacmanCurrent[1]
            else:
                pacmanSuccessors = problem.getSuccessors(pacmanCurrent[0])
            Successor = []
            for item in pacmanSuccessors:  # item: [(x,y), 'direction', cost]
                if item[0] not in visitedPosition:
                    pacmanRoute = pacmanCurrent[1].copy()
                    pacmanRoute.append(item[1])
                    sumCost = pacmanCurrent[2]
                    Successor.append([item[0], pacmanRoute, sumCost + item[2]])
            for item in Successor:
                fringe.push(item, item[2])
        return pacmanCurrent[1]

    def getAction(self, state):
        return self.findPathToClosestDot(state)[0]

class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem, but has a
    different goal test, which you need to fill in below.  The state space and
    successor function do not need to be changed.

    The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
    inherits the methods of the PositionSearchProblem.

    You can use this search problem to help you fill in the findPathToClosestDot
    method.
    """

    def __init__(self, gameState, agentIndex):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition(agentIndex)
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def isGoalState(self, state):
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """
        return self.food[state[0]][state[1]]
