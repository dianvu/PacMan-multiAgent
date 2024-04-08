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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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

        "*** YOUR CODE HERE ***"
        """Check if the game is won or lost"""
        if successorGameState.isWin():
            return float('inf')
        if successorGameState.isLose():
            return -float('inf')
        
        """Evaluate the distance from pacman to the food"""
        # Get new food (x, y) positions
        newFoodPos = newFood.asList()
        # Initialize the minimum distance to food                          
        min_food_distance = float('inf')
        # for each food position, calculate the distance from pacman using manhattan distance
        for food in newFoodPos:
            # assign minimim distance 
            min_food_distance = min(min_food_distance, manhattanDistance(newPos, food))

        """Evaluate the distances from pacman to the ghosts"""
        # Initialize the distances to ghosts equal to 1 to avoid ghost collide
        distances_to_ghosts = 1
        # Initialize the penetration distance to 0
        penetration_distance = 0
        # for each ghost state, calculate the distance from pacman using manhattan distance
        for ghost_state in newGhostStates:
            if ghost_state.scaredTimer == 0:
                distance = manhattanDistance(newPos, ghost_state.getPosition())
                # add calculated distance to the total distance
                distances_to_ghosts += distance
                # if the distance is less than 2, add 1 to the penetration distance
                if distance < 2:
                    penetration_distance += 1
                    
        """Evaluation metrics: 1/(min_food_distance) (maximize score), 1/(distances_to_ghosts) (minimize score), penetration_distance (minimize score)"""
        return successorGameState.getScore() + (1 / (min_food_distance)) - (1 / (distances_to_ghosts)) - penetration_distance
        

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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
        """Returns value as pair (score, action)"""
        score, action = self.minimax_search(gameState, 0, 0) # start at agentIndex 0 and depth 0
        # Return the action from the result
        return action
        
    def minimax_search(self, gameState, agentIndex, depth):
        """ function MINIMAX-SEARCH(game, state) returns an action
            player <- game.TO-MOVE(state)
            value, move <- MAX-VALUE(game, state)
            return move """
        # if no legal moves or the depth is reached
        if (len(gameState.getLegalActions(agentIndex)) == 0) or depth == self.depth:
            return gameState.getScore(), ""
        # when agentIndex is 0, return max-agent
        if agentIndex == 0: 
            return self.max_agent(gameState, agentIndex, depth)
        # else, return min-agent
        else:
            return self.min_value(gameState, agentIndex, depth)
            
    def max_agent (self, gameState, agentIndex, depth):
        """ function MAX-VALUE(game, state) returns a (score, action) pair
            if game.IS-TERMINAL(state) then return game.UTILITY(state, player), null
                v <- -infinity
                a <- null
            for each a in game.ACTIONS(state) do
                v2, a2 <- MIN-VALUE(game, game.RESULT(state, a))
                if v2 > v then
                    v, action <- v2, a
            return v, action"""
        maxScore = -float('inf')
        maxAction = ""
        # for each action in legal moves
        for action in gameState.getLegalActions(agentIndex):
            # generate the successor state upon the action
            successor = gameState.generateSuccessor(agentIndex, action)
            # increment agenIndex by 1 for each iteration loop
            successorIndex = agentIndex + 1
            successorDepth = depth
            # if the successorIndex is equal to the number of agents, reset the agentIndex to 0 and increment the depth
            if successorIndex == gameState.getNumAgents():
                successorIndex = 0
                successorDepth += 1
            # recursively call the minimax_search function to get the value of the successor
            currScore = self.minimax_search(successor, successorIndex, successorDepth)[0]
            # find the maxAction that leads to maxScore if currScore is greater than maxScore
            maxScore, maxAction = (currScore, action) if currScore > maxScore else (maxScore, maxAction)  
        return maxScore, maxAction
        
    def min_value(self, gameState, agentIndex, depth):
        """ function MIN-VALUE(game, state) returns a (score, action) pair
            if game.IS-TERMINAL(state) then return game.UTILITY(state, player), null
                v <- +infinity
            for each a in game.ACTIONS(state) do
                v2, a2 <- MAX-VALUE(game, game.RESULT(state, a))
                if v2 < v then
                    v, action <- v2, a
            return v, action"""
        minScore = float('inf')
        minAction = ""
        # for each action in legal moves
        for action in gameState.getLegalActions(agentIndex):
            # generate the successor state upon the action
            successor = gameState.generateSuccessor(agentIndex, action)
            # increment agenIndex by 1 for each iteration loop
            successorIndex = agentIndex + 1
            successorDepth = depth
            # if the successorIndex is equal to the number of agents, reset the agentIndex to 0 and increment the depth
            if successorIndex == gameState.getNumAgents():
                successorIndex = 0
                successorDepth += 1
            # recursively call the minimax_search function to get the value of the successor
            currScore = self.minimax_search(successor, successorIndex, successorDepth)[0]
            # find the minAction that leads to minScore if currScore is less than minScore
            minScore, minAction = (currScore, action) if currScore < minScore else (minScore, minAction)  
        return minScore, minAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        action, score = self.alphabeta_search(gameState, 0, 0, -float('inf'), float('inf'))
        return action
        
    def alphabeta_search(self, gameState, agentIndex, depth, alpha, beta):
        """Similar with minimax_search but with alpha-beta pruning
        alpha = -infinity
        beta = infinity"""
        if (len(gameState.getLegalActions(agentIndex)) == 0) or depth == self.depth:
            return "", gameState.getScore()
        if agentIndex == 0:
            return self.max_value(gameState, agentIndex, depth, alpha, beta)
        else:
            return self.min_value(gameState, agentIndex, depth, alpha, beta)

    def max_value(self, gameState, agentIndex, depth, alpha, beta):
        """ function MAX-VALUE(game, state) returns a (score, action) pair
            if game.IS-TERMINAL(state) then return game.UTILITY(state, player), null
                v <- -infinity
                a <- null
            for each a in game.ACTIONS(state) do
                v2, a2 <- MIN-VALUE(game, game.RESULT(state, a), alpha, beta)
                if v2 > v then
                    v, action <- v2, a
                    alpha <- max(alpha, v)
                if v > beta then return v, action"""
        maxScore = -float('inf')
        maxAction = ""
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            successorIndex = agentIndex + 1
            successorDepth = depth
            if successorIndex == gameState.getNumAgents():
                successorIndex = 0
                successorDepth += 1
            currScore = self.alphabeta_search(successor, successorIndex, successorDepth, alpha, beta)[1]
            maxScore, maxAction = (currScore, action) if currScore > maxScore else (maxScore, maxAction)
            # update alpha
            alpha = max(alpha, maxScore)
            # prune the tree if maxScore is greater than beta
            if maxScore > beta:
                return maxAction, maxScore
        return maxAction, maxScore
    
    def min_value(self, gameState, agentIndex, depth, alpha, beta):
        """ function MIN-VALUE(game, state) returns a (score, action) pair
            if game.IS-TERMINAL(state) then return game.UTILITY(state, player), null
                v <- +infinity
            for each a in game.ACTIONS(state) do
                v2, a2 <- MAX-VALUE(game, game.RESULT(state, a))
                if v2 < v then
                    v, action <- v2, a
                    beta <- min(beta, v)
                if v < alpha then return v, action"""
        minScore = float('inf')
        minAction = ""
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            successorIndex = agentIndex + 1
            successorDepth = depth
            if successorIndex == gameState.getNumAgents():
                successorIndex = 0
                successorDepth += 1
            currScore = self.alphabeta_search(successor, successorIndex, successorDepth, alpha, beta)[1]
            minScore, minAction = (currScore, action) if currScore < minScore else (minScore, minAction)
            # update beta
            beta = min(beta, minScore)
            # prune the tree if minScore is less than alpha
            if minScore < alpha:
                return minAction, minScore
        return minAction, minScore
    
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        action, score = self.expectimax_search(gameState, 0, 0)
        return action
        
    def expectimax_search(self, gameState, agentIndex, depth):
        if (len(gameState.getLegalActions(agentIndex)) == 0) or depth == self.depth:
            return "", gameState.getScore()
        if agentIndex == 0:
            return self.max_value(gameState, agentIndex, depth)
        else:
            return self.expected_value(gameState, agentIndex, depth)
    
    def max_value(self, gameState, agentIndex, depth):
        """Similar with max_value function in MinimaxAgent"""
        maxScore = -float("inf")
        maxAction = ""
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            successorIndex = agentIndex + 1
            successorDepth = depth
            if successorIndex == gameState.getNumAgents():
                successorIndex = 0
                successorDepth += 1
            currAction, currScore = self.expectimax_search(successor, successorIndex, successorDepth)
            maxScore, maxAction = (currScore, action) if currScore > maxScore else (maxScore, maxAction)
        return maxAction, maxScore
    
    def expected_value(self, gameState, agentIndex, depth):
        """ function EXPECTIMAX-VALUE(game, state) returns a (score, action) pair
            if game.IS-TERMINAL(state) then return game.UTILITY(state, player), null
                v <- 0
                probability <- 1 / |game.ACTIONS(state)|
            for each a in game.ACTIONS(state) do
                v2, a2 <- MAX-VALUE(game, game.RESULT(state, a))
                v, action <- v + probability * v2, a
            return v, action"""
        expectedScore = 0
        expectedAction = ""
        successor_probability = 1 / len(gameState.getLegalActions(agentIndex))
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            successorIndex = agentIndex + 1
            successorDepth = depth
            if successorIndex == gameState.getNumAgents():
                successorIndex = 0
                successorDepth += 1
            # recursively call the expectimax_search function to get the value of the successor
            currAction, currScore = self.expectimax_search(successor, successorIndex, successorDepth)
            # calculate the expected score
            expectedScore += successor_probability * currScore
        return expectedAction, expectedScore
    
def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Evaluation function to guide Pacman agent to make decision that maximize the score
    1. Calculate the distance from pacman to the nearest food 
    2. Calculate the distance from pacman to the nearest ghost
    3. Return the score plus the inverse of the distance to the nearest food minus the inverse of the distance to the nearest ghost
    In summary,  the goal of this evaluation is to encourage agent to eat food and discourage the agent to collide with the ghost
    """
    "*** YOUR CODE HERE ***"
    
    # Useful information you can extract from a GameState
    foods = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    pacmanPos = currentGameState.getPacmanPosition()
    
    def distanceToNearestFood():
        distance = float('inf')
        for food in foods:
            distance = min(distance, manhattanDistance(currentGameState.getPacmanPosition(), food))
        return distance
    
    def distanceToNearestGhost():
        distance = 1
        for ghostState in ghostStates:
            distance += manhattanDistance(pacmanPos, ghostState)
        return distance
    
    return currentGameState.getScore() + 1 / distanceToNearestFood() - 1 / distanceToNearestGhost()
    
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
