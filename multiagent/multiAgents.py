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
import random, util, time

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
        newGhostStates = successorGameState.getGhostStates()
        #newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
		
        "*** YOUR CODE HERE ***"
        score = float(0)
        currentFood = currentGameState.getFood().asList()	#모든 음식의 거리 리스트
        x, y = newPos
        
        for i in range(len(newGhostStates)):
            a, b = newGhostStates[i].getPosition() #유령의 위치 구함
            movesAway = abs(x-a) + abs(y-b) # 유령과 팩맨의 맨해튼 거리

            if movesAway < 2:   #유령이 너무 가까우면 점수에 -20점 뺌.
                score += -20

        FoodDistance = []   #음식-팩맨 맨해튼 거리 리스트
        for c, d in currentFood:
            foodfar = abs(x-c) + abs(y-d)   #음식-팩맨 맨해튼 거리
            FoodDistance.append(foodfar)    #저장
        score += -1 * min(FoodDistance)    #모든 음식에 대해 음식거리리스트 중 제일 작은값으로 역수를 취해 값을 줄임
        
        return score	#score 반환

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
        maxValue = -100000000		#가장 큰 이득이 되는 값
        maxAction = Directions.STOP	#최종 팩맨의 이동
        for action in gameState.getLegalActions(self.index):	#팩맨이 이동할 수 있는 방향 수만큼 반복
            nextState = gameState.generateSuccessor(0, action)	#팩맨이 이동 가능한 상태 저장
            FinalValue = self.Value(nextState, 0, 1)			#minimax 수행, 이 때, 최종 값은 유령의 min 값이 나옴.
            if FinalValue > maxValue:	#유령의 min값이 최대의 이득값이면
                maxValue = FinalValue	#변경
                maxAction = action		#움직임도 변경
        return maxAction	#액션 반환

        util.raiseNotDefined()

    def Value(self, gameState, currentDepth, agentIndex):	#팩맨과 유령, 터미널을 구분하는 함수
        if currentDepth == self.depth or gameState.isWin() or gameState.isLose():	#터미널이거나, 이기거나 지면
            return self.evaluationFunction(gameState)	#터미널에 있는 유틸리티 값 가지고옴.
        elif agentIndex == 0:	#팩맨이면 maxValue 수행
            return self.maxValue(gameState,currentDepth)
        else:	#유령이면 minValue 수행
            return self.minValue(gameState,currentDepth,agentIndex)

    def maxValue(self, gameState, currentDepth):	#팩맨이 가장 이득이 되는 값을 가지고 오는 구간
        maxValue = -100000000	#이득값 저장 변수
        for action in gameState.getLegalActions(self.index):	#팩맨이 움직일 수 있는 액션만큼 반복
            nextValue = self.Value(gameState.generateSuccessor(0, action), currentDepth, 1)	#아래 노드의 유령들의 min 값 가지고옴
            maxValue = max(maxValue,nextValue)	#가지고온 min과 현재 max 값 중 큰값으로 변환
        return maxValue	#maxValue 반환

    def minValue(self, gameState, currentDepth, agentIndex):	#유령이 가장 손해가 되는 값을 가지고 오는 구간
        minValue = 100000000	#손해값 저장 변수
        for action in gameState.getLegalActions(agentIndex):	#유령이 움직일 수 있는 액션만큼 반복
            if agentIndex == gameState.getNumAgents()-1:	#해당 유령이 마지막 유령이면
                nextValue = self.Value(gameState.generateSuccessor(agentIndex, action), currentDepth+1, 0)	#아래 노드의 팩맨의 이득값을 가지고옴.
                minValue = min(minValue, nextValue)	#가지고온 min과 현재 max 값 중 적은값으로 변환
            else:	#마지막 유령이 아니면
                nextValue = self.Value(gameState.generateSuccessor(agentIndex, action), currentDepth, agentIndex+1) #형제 노드의 유령의 다른 손해값을 가지고 옴.
                minValue = min(minValue, nextValue)	#가지고온 min과 현재 min 값 중 적은값으로 변환
        return minValue	#minValue 반환

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        maxValue = -100000000			#가장 큰 이득이 되는 값
        maxAction = Directions.STOP		#최종 팩맨의 이동
        alpha = -100000000         		#알파값
        beta = 100000000				#베타값
        for action in gameState.getLegalActions(self.index):              #팩맨이 이동할 수 있는 방향 수만큼 반복
            nextState = gameState.generateSuccessor(0, action)            #팩맨이 이동 가능한 상태 저장
            FinalValue = self.Value(nextState, 0, 1, alpha, beta)         #AlphaBetaPurning 수행, 이 때, 최종 값은 유령의 min 값이 나옴.
            if FinalValue > maxValue:	#유령의 min값이 최대의 이득값이면
                maxValue = FinalValue	#변경
                maxAction = action   	#움직임도 변경
                alpha = FinalValue		#알파값 또한 변경
        return maxAction	#액션 반환
        util.raiseNotDefined()

    def Value(self, gameState, currentDepth, agentIndex, alpha, beta):		#팩맨인지 유령인지 터미널인지 비교하는 함수
        if currentDepth == self.depth or gameState.isWin() or gameState.isLose():	#터미널이거나, 이기거나 지면
            return self.evaluationFunction(gameState)	#터미널에 있는 유틸리티 값 가지고옴.
        elif agentIndex == 0:	#팩맨이면 maxValue 수행
            return self.maxValue(gameState,currentDepth, alpha, beta)
        else:	#유령이면 minValue 수행
            return self.minValue(gameState,currentDepth,agentIndex, alpha, beta)

    def maxValue(self, gameState, currentDepth, alpha, beta):	#팩맨이 가장 이득이 되는 값을 가지고 오는 구간
        maxValue = -100000000
        for action in gameState.getLegalActions(self.index):	#팩맨이 움직일 수 있는 액션만큼 반복
            maxiValue = self.Value(gameState.generateSuccessor(0, action), currentDepth, 1, alpha, beta)	#아래 노드의 유령들의 손해값을 가지고 옴
            maxValue = max(maxValue,maxiValue)	#유령들 손해값과, 현재 이득값과 큰 값을 비교하여 적용
            if maxValue > beta:	#이득값이 베타값보다 크면
                return maxValue	#알파 변환 없이 바로 반환
            alpha = max(alpha, maxValue)	#알파값을 maxValue와 비교하여 큰값을 알파에 다시 넣음.
        return maxValue	#maxValue 반환

    def minValue(self, gameState, currentDepth, agentIndex, alpha, beta):	#유령이 가장 손해가 되는 값을 가지고 오는 구간
        minValue = 100000000												#minValue 선언
        for action in gameState.getLegalActions(agentIndex):				#유령이 움직일 수 있는 액션만큼 반복
            if agentIndex == gameState.getNumAgents()-1:					#해당 유령이 마지막 유령이면
                miniValue = self.Value(gameState.generateSuccessor(agentIndex, action), currentDepth+1, 0, alpha, beta)	#아래 노드의 팩맨의 이득값을 가지고옴.
                minValue = min(minValue, miniValue)	#가지고온 min과 현재 max 값 중 적은값으로 변환
                if minValue < alpha:	#minValue 값이 알파값보다 적으면 바로 minValue 반환
                    return minValue
                beta = min(beta, minValue)#그렇지않으면 베타값을 바꿈.
            else:
                miniValue = self.Value(gameState.generateSuccessor(agentIndex, action), currentDepth, agentIndex+1, alpha, beta) #형제 노드의 유령의 다른 손해값을 가지고 옴.
                minValue = min(minValue, miniValue)	#가지고온 min과 현재 min 값 중 적은값으로 변환
                if minValue < alpha:	#minValue 값이 알파값보다 적으면 바로 minValue 반환
                    return minValue
                beta = min(beta, minValue)	#그렇지않으면 베타값을 바꿈.
        return minValue	#minValue 반환

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
        maxValue = -100000000         #가장 큰 이득이 되는 값
        maxAction = Directions.STOP   #최종 팩맨의 이동
        for action in gameState.getLegalActions(self.index):        #팩맨이 이동할 수 있는 방향 수만큼 반복
            nextState = gameState.generateSuccessor(0, action)      #팩맨이 이동 가능한 상태 저장
            FinalValue = self.Value(nextState, 0, 1)                #AlphaBetaPurning 수행, 이 때, 최종 값은 유령의 exp 값이 나옴.
            if FinalValue > maxValue:     #유령의 min값이 최대의 이득값이면
                maxValue = FinalValue     #변경
                maxAction = action        #움직임도 변경
        return maxAction	#액션 반환

        util.raiseNotDefined()

    def Value(self, gameState, currentDepth, agentIndex):			#팩맨인지 유령인지 터미널인지 비교하는 함수
        if currentDepth == self.depth or gameState.isWin() or gameState.isLose():	#터미널이거나, 이기거나 지면 유틸리티 값 가지고옴.
            return self.evaluationFunction(gameState)
        elif agentIndex == 0:	#팩맨이면 maxValue 수행
            return self.maxValue(gameState,currentDepth)
        else:	#유령이면 expValue 수행
            return self.ExpValue(gameState,currentDepth,agentIndex)

    def maxValue(self, gameState, currentDepth):	#팩맨이 가장 이득이 되는 값을 가지고 오는 구간
        maxValue = -100000000	#max값 선언
        for action in gameState.getLegalActions(self.index):	#팩맨이 움직일 수 있는 액션만큼 반복
            nextValue = self.Value(gameState.generateSuccessor(0, action), currentDepth, 1)	#아래 노드의 유령들의 손해값을 가지고 옴
            maxValue = max(maxValue,nextValue)	#유령들 손해값과, 현재 이득값과 큰 값을 비교하여 적용
        return maxValue	#maxValue 반환

    def ExpValue(self, gameState, currentDepth, agentIndex):            #유령이 가장 손해가 되는 값을 가지고 오는 구간
        ExpValue = 0	#exp 값 선언
        for action in gameState.getLegalActions(agentIndex):                                    #유령이 움직일 수 있는 액션만큼 반복
            if agentIndex == gameState.getNumAgents()-1:                                        #해당 유령이 마지막 유령이면
                probability = len(gameState.getLegalActions(agentIndex))                        #확률 계산, 확률은 액션을 할 수 있는 수만큼            
                nextValue = self.Value(gameState.generateSuccessor(agentIndex, action), currentDepth+1, 0)  	#아래 노드의 팩맨의 이득값을 가지고옴.
                ExpValue = (ExpValue) + (nextValue / probability)	#확률을 이용한 계산, 확률 * 유틸리티를 합침.
            else:
                probability = len(gameState.getLegalActions(agentIndex))	#확률 계산, 확률은 액션을 할 수 있는 수만큼   
                nextValue = self.Value(gameState.generateSuccessor(agentIndex, action), currentDepth, agentIndex+1) #형제 노드의 유령의 다른 손해값을 가지고 옴.
                ExpValue = (ExpValue) + (nextValue / probability)	#확률을 이용한 계산, 확률 * 유틸리티를 합침.
        return ExpValue	#exp 값 출력

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()		#팩맨의 위치
    newGhostStates = currentGameState.getGhostStates()	#유령의 상태
    Powerlist = currentGameState.getCapsules()			#파워의 위치
    score = scoreEvaluationFunction(currentGameState)	#점수, 점수는 GUI에 나오는 점수를 가지고옴.
    currentFood = currentGameState.getFood().asList()	#음식 위치 리스트

    x, y = newPos	#팩맨 위치
    FoodDistance = []   #음식 맨해튼 리스트
    PowerDistance = []  #파워 맨해튼 리스트
    scaredGhosts = []	#두려운 유령들 저장하는 리스트
    activeGhosts = []	#활성화 유령들 저장하는 리스트
    scaredGhostsDistances = []	#두려운 유령들 맨해튼 리스트
    activeGhostsDistances = []	#활성화 유령들 맨해튼 리스트

    if currentGameState.isWin():	#게임에서 이기면 최고 점수 부여
        return 10000000
    if currentGameState.isLose():	#게임에서 지면 최저 점수 부여
        return -10000000

    for ghost in newGhostStates:	#유령의 개수만큼 반복
        if ghost.scaredTimer: # 유령이 무서워하면
            scaredGhosts.append(ghost)	#두려운 유령 리스트에 넣음
        else:					#유령이 안무서워하면 활성화 유령 리스트에 넣음
            activeGhosts.append(ghost)

    for c, d in currentFood:			#음식 위치 리스트만큼 반복
        foodfar = abs(x-c) + abs(y-d)   #음식-팩맨 맨해튼 거리
        FoodDistance.append(foodfar)    #맨해튼 거리 저장

    for e, f in Powerlist:              #파워 위치 리스트만큼 반복
        Powerfar = abs(x-e) + abs(y-f)  #파워-팩맨 맨해튼 거리
        PowerDistance.append(Powerfar)  #맨해튼 거리 저장

    for ghost in activeGhosts:                    #활성화 유령 위치 리스트만큼 반복
        g, h = ghost.getPosition()                #활성화 유령 위치
        activefar = abs(x-g) + abs(y-h)           #활성화 유령-팩맨 맨해튼 거리
        scaredGhostsDistances.append(activefar)   #맨해튼 거리 저장

    for ghost in scaredGhosts:                    #두려운 유령 위치 리스트만큼 반복
        i, j = ghost.getPosition()                #두려운 유령 위치
        scaredfar = abs(x-i) + abs(y-j)           #두려운 유령-팩맨 맨해튼 거리
        scaredGhostsDistances.append(scaredfar)   #맨해튼 거리 저장

    score += -10 * len(FoodDistance)			#먹을거가 많으면 점수를 깍아서 중요도 결정
    score += -20 * len(PowerDistance)			#파워가 많으면 점수를 깍아서 중요도 결정
    
	#팩맨이 벽을 만나면 엄청난 점수를 깍음.
    if currentGameState.getPacmanState().getDirection() == Directions.WEST:	#서쪽
        if currentGameState.hasWall(x+1, y):
            print("WallW")
            score -= 200000
    elif currentGameState.getPacmanState().getDirection() == Directions.EAST:	#동쪽
        if currentGameState.hasWall(x-1, y):
            print("WallE")
            score -= 200000
    elif currentGameState.getPacmanState().getDirection() == Directions.SOUTH:	#남쪽
        if currentGameState.hasWall(x, y+1):
            print("WallS")
            score -= 200000
    elif currentGameState.getPacmanState().getDirection() == Directions.NORTH:	#북쪽
        if currentGameState.hasWall(x, y-1):
            print("WallN")
            score -= 200000

    for Food in FoodDistance:	#음식 맨해튼 거리 개수만큼 반복
        if Food < 3:			#음식이 가까이 있으면 점수를 많이 깍음
            score += -1 * Food
        elif Food < 7:          #음식이 중간정도만 있으면 점수를 중간정도만 깍음
            score += -0.5 * Food
        else:                   #음식이 멀리 있으면 점수를 적게 깍음, 단 거리가 길수록 점수를 많이 깍음.
            score += -0.3 * Food

    for Power in PowerDistance:    #파워가 맨해튼 거리 개수만큼 반복
        if Power < 3:              #파워가 가까이 있으면 점수를 많이 깍음
            score += -1 * Power    
        elif Power < 7:            #파워가 중간정도만 있으면 점수를 중간정도만 깍음
            score += -0.5 * Power  
        else:                      #파워가 멀리 있으면 점수를 적게 깍음, 단 거리가 길수록 점수를 많이 깍음.
            score += -0.3 * Power

    for ghost in scaredGhostsDistances:	  #유령이 맨해튼 거리 개수만큼 반복
        if ghost < 3:                     #유령이 가까이 있으면 점수를 엄청많이 깍음
            score += -20 * ghost          
        else:                             #유령이 멀리 있으면 점수를 상당히 많이 깍음
            score += -10 * ghost          
                                          
    for ghost in activeGhostsDistances:   #유령이 맨해튼 거리 개수만큼 반복
        if ghost < 3:                     #유령이 가까이 있으면 점수를 많이 깍음
            score += -3 * ghost           
        elif ghost < 7:                   #유령이 가까이 있으면 점수를 중간정도만 깍음
            score += -2 * ghost           
        else:                             #유령이 멀리 있어도 음식 수준으로 깍음.
            score += -1 * ghost

    #print(score)
    return score	#점수반환
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
