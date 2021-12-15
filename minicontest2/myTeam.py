# myTeam.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random

import util
from captureAgents import CaptureAgent
from game import Directions
from util import nearestPoint


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='MainAgent', second='MainAgent'):
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
    firstAgent = eval(first)(firstIndex)
    secondAgent = eval(second)(secondIndex)

    firstAgent.northBias = True
    secondAgent.northBias = False

    firstAgent.otherAgent = secondAgent
    secondAgent.otherAgent = firstAgent

    firstAgent.isInDanger = False
    secondAgent.isInDanger = False

    return [firstAgent, secondAgent]


##########
# Agents #
##########


class MainAgent(CaptureAgent):
    def __init__(self, index, timeForComputing=.1):
        CaptureAgent.__init__(self, index, timeForComputing=.1)
        self.otherAgent = None
        self.northBias = None

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).
        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)
        IMPORTANT: This method may run for at most 15 seconds.
        """

        ''' 
        Make sure you do not delete the following line. If you would like to
        use Manhattan distances instead of maze distances in order to save
        on initialization time, please take a look at
        CaptureAgent.registerInitialState in captureAgents.py. 
        '''
        CaptureAgent.registerInitialState(self, gameState)

        ''' 
        Your initialization code goes here, if you need any.
        '''
        if self.red:
            self.teamIndices = gameState.getRedTeamIndices()

        else:
            self.teamIndices = gameState.getBlueTeamIndices()

        x = int(gameState.getWalls().width / 2)
        y = int(gameState.getWalls().height / 2)

        self.mazeCenter = (x, y)

        self.goToCenter(gameState)

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        evaluateType = 'attack'

        if not self.reachedCenter:
            evaluateType = 'goToCenter'

        agentCurrentPosition = self.getCurrentAgentPosition(gameState)

        if agentCurrentPosition == self.mazeCenter and not self.reachedCenter:
            evaluateType = 'attack'
            self.reachedCenter = True

        enemiesIndexesAndPositions = self.getEnemiesIndexesAndPositions(gameState)
        safeEnemyDistance = 6

        if enemiesIndexesAndPositions:
            if [(index, enemyPosition) for index, enemyPosition in enemiesIndexesAndPositions
                if self.getMazeDistance(agentCurrentPosition, enemyPosition) < safeEnemyDistance
                   and self.isInOwnTerritory(gameState)]:
                evaluateType = 'defense'

        actions = gameState.getLegalActions(self.index)
        values = [self.evaluate(gameState, action, evaluateType) for action in actions]

        maxValue = max(values)
        bestActions = [action for action, value in zip(actions, values) if value == maxValue]

        return random.choice(bestActions)

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        agentCurrentPosition = successor.getAgentState(self.index).getPosition()

        if agentCurrentPosition != nearestPoint(agentCurrentPosition):
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action, agentType='attack'):
        """
        Computes a linear combination of features and feature weights
        """
        global features
        weights = self.getWeights()

        if agentType == 'attack':
            features = self.getFeaturesForAttack(gameState, action)

        elif agentType == 'defense':
            features = self.getFeaturesForDefense(gameState, action)

        elif agentType == 'goToCenter':
            features = self.getFeaturesForGoingToCenter(gameState, action)

        return sum(features[key] * weights.get(key, 0) for key in features)

    def getFeaturesForAttack(self, gameState, action):
        attackFeatures = util.Counter()
        successor = self.getSuccessor(gameState, action)

        attackFeatures['successorScore'] = self.getScore(successor)

        attackFeatures['distanceToFood'] = self.getDistanceToClosestFood(successor)

        closestCapsuleDistance = self.getDistanceToClosestCapsule(gameState, successor)

        if closestCapsuleDistance is not None:
            try:
                attackFeatures['distanceToCapsule'] = 1.0 / closestCapsuleDistance

            except ZeroDivisionError:
                attackFeatures['distanceToCapsule'] = 10

        enemyIndex, enemyDistance = self.getEnemiesIndexesAndDistances(successor)

        safeClosestCapsuleDistance = 6

        if closestCapsuleDistance is not None:
            if (enemyDistance is not None and self.getEnemyScaredTimer(gameState, enemyIndex) > 1) \
                    or (self.otherAgent.isInDanger and closestCapsuleDistance < safeClosestCapsuleDistance):
                attackFeatures['distanceToCapsuleInDanger'] = closestCapsuleDistance

        safeEnemyDistance = 4
        self.isInDanger = False

        if enemyDistance is not None:
            if enemyDistance <= safeEnemyDistance:
                self.isInDanger = True

                if self.getEnemyScaredTimer(gameState, enemyIndex) <= 1:
                    attackFeatures['distanceToEnemy'] = 1.0 / enemyDistance

                    if enemyDistance <= 1:
                        attackFeatures['deadlyAction'] = 1

                    if self.isInOwnTerritory(successor) and self.getCurrentAgentScaredTimer(successor) < 2:
                        attackFeatures['inOwnTerritoryInDanger'] = 1

                else:
                    attackFeatures['distanceToEnemy'] = 0

        if not self.isInOwnTerritory(successor):
            distanceToAgent = self.getDistanceBetweenTeammates(successor)

            if distanceToAgent is not None:
                attackFeatures['distanceToTeammate'] = 1.0 / distanceToAgent
                
        if action == Directions.STOP:
            attackFeatures['stop'] = 1

        reverse = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]

        if action == reverse:
            attackFeatures['reverse'] = 1

        return attackFeatures

    def getFeaturesForDefense(self, gameState, action):
        defenseFeatures = util.Counter()
        successor = self.getSuccessor(gameState, action)

        attackers = list(filter(lambda an_enemy: an_enemy.isPacman and an_enemy.getPosition() is not None,
                                [successor.getAgentState(enemy) for enemy in self.getOpponents(successor)]))
        defenseFeatures['numberOfAttackers'] = len(attackers)

        if attackers:
            minimumEnemyIndex, minimumDistance = self.getEnemiesIndexesAndDistances(successor)
            defenseFeatures['distanceToClosestAttacker'] = minimumDistance

            if minimumDistance <= 1:
                if (self.getCurrentAgentScaredTimer(successor) > 0 and self.isInOwnTerritory(successor)) \
                        or (not self.isInOwnTerritory(successor)):
                    defenseFeatures['deadlyAction'] = 1

        reverse = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]

        if action == reverse:
            defenseFeatures['reverse'] = 1

        return defenseFeatures

    def getFeaturesForGoingToCenter(self, gameState, action):
        gotToCenterFeatures = util.Counter()
        successor = self.getSuccessor(gameState, action)

        currentAgentPosition = self.getCurrentAgentPosition(successor)
        gotToCenterFeatures['distanceToCenter'] = self.getMazeDistance(currentAgentPosition, self.mazeCenter)

        if currentAgentPosition == self.mazeCenter:
            gotToCenterFeatures['atCenter'] = 1

        return gotToCenterFeatures

    def getWeights(self):
        return {
            'successorScore': 0.04,
            'numberOfAttackers': -0.2,
            'distanceToClosestAttacker': -0.002,
            'distanceToFood': -0.0005,
            'distanceToTeammate': -0.008,
            'distanceToEnemy': -0.045,
            'distanceToCapsule': 0.009,
            'distanceToCapsuleInDanger': -0.046,
            'inOwnTerritoryInDanger': 0.08,
            'distanceToCenter': -0.002,
            'atCenter': 0.02,
            'deadlyAction': -1.0,
            'stop': -0.02,
            'reverse': -0.0004
        }

    def getCurrentAgentPosition(self, gameState):
        return gameState.getAgentState(self.index).getPosition()

    def isInEnemyTerritory(self, gameState):
        return gameState.getAgentState(self.index).isPacman

    def isInOwnTerritory(self, gameState):
        return not self.isInEnemyTerritory(gameState)

    def getCurrentAgentScaredTimer(self, gameState):
        return gameState.getAgentState(self.index).scaredTimer

    def getEnemyScaredTimer(self, gameState, enemyIndex):
        return gameState.getAgentState(enemyIndex).scaredTimer

    def getEnemiesIndexesAndPositions(self, gameState):
        enemiesAndPositions = [(enemy, gameState.getAgentPosition(enemy)) for enemy in self.getOpponents(gameState)]
        return [enemyTuple for enemyTuple in enemiesAndPositions if None not in enemyTuple]

    def getEnemiesIndexesAndDistances(self, gameState):
        agentCurrentPosition = self.getCurrentAgentPosition(gameState)

        try:
            enemyIndex, enemyDistance = min([(index, self.getMazeDistance(agentCurrentPosition, enemyPosition))
                                             for (index, enemyPosition)
                                             in self.getEnemiesIndexesAndPositions(gameState)])

            return enemyIndex, max(enemyDistance, 0.5)

        except ValueError:
            return None, None

    def getDistanceBetweenTeammates(self, gameState):
        agentDistance = None

        if self.index != self.teamIndices[0]:
            otherAgentIndex = self.teamIndices[0]
            otherAgentPosition = gameState.getAgentState(otherAgentIndex).getPosition()
            agentDistance = self.getMazeDistance(self.getCurrentAgentPosition(gameState), otherAgentPosition)

        if agentDistance is not None:
            return max(agentDistance, 0.5)
        else:
            return None

    def getDistanceToClosestFood(self, gameState):
        try:
            return min([self.getMazeDistance(self.getCurrentAgentPosition(gameState), food)
                        for food in self.getFood(gameState).asList()])

        except ValueError:
            return None

    def getDistanceToClosestCapsule(self, gameState, successor):
        if len(self.getCapsules(successor)) < len(self.getCapsules(gameState)):
            return 0

        else:
            try:
                return min([self.getMazeDistance(self.getCurrentAgentPosition(successor), capsulePosition)
                            for capsulePosition in self.getCapsules(successor)])
            except ValueError:
                return None

    def goToCenter(self, gameState):
        self.reachedCenter = False

        x = int(gameState.getWalls().width / 2)
        y = int(gameState.getWalls().height / 2)

        if self.red:
            x = x - 1

        if self.northBias:
            maxHeight = gameState.getWalls().height
            locations = [(x, thisY) for thisY in range(maxHeight - y, maxHeight, +1) if not gameState.hasWall(x, thisY)]
        else:
            locations = [(x, thisY) for thisY in range(y, 0, -1) if not gameState.hasWall(x, thisY)]

        self.mazeCenter = sorted(locations, key=lambda location: self.getMazeDistance(
            self.getCurrentAgentPosition(gameState), location))[0]