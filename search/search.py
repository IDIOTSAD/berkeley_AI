# search.py
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    state = problem.getStartState() #현재 상태
    mazestack = util.Stack()    #스택 선언
    visit = []  #방문 리스트 선언

    mazestack.push((state, [])) #스택에 루트 노드 삽입

    item = 0

    while True:
        if mazestack.isEmpty(): #스택이 비어 있으면 종료
            break

        node, direction = mazestack.pop()   #스택에 있는 값을 빼옴.
        visit.insert(item, node)    #방문한 노드라고 판별시킴. 방문 리스트에 삽입함.
        item = item + 1 #방문한 노드 수

        if problem.isGoalState(node):   #문제의 골에 도착하면  여태까지 간 거리(노드)를 반환시킴.
            return direction
        else:
            for successor, action, cost in problem.getSuccessors(node): #방문 할 수 있는 노드들 반복
                if not successor in visit:  #방문하지 않은 노드면 스택에 삽입
                    mazestack.push((successor, direction + [action]))
    return []   #문제 탐색 실패
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    state = problem.getStartState() #현재 상태
    mazequeue = util.Queue()    #큐 선언
    visit = []  #방문 리스트 선언
    mazequeue.push((state, [])) #큐에 루트 노드 삽입
    visit.append(state) #방문 리스트에 루트노드 삽입

    while True:
        if mazequeue.isEmpty(): #큐가 비어 있으면 종료
            break

        node, direction = mazequeue.pop()   #큐에 있는 값을 빼옴.

        if problem.isGoalState(node):   #문제의 골에 도착하면 여태까지 간 거리(노드)를 반환시킴.
            return direction
        else:
            for successor, action, stepcost in problem.getSuccessors(node):#방문 할 수 있는 노드들 반복
                if successor not in visit:  #방문하지 않은 노드면 큐에 삽입
                    mazequeue.push((successor, direction + [action]))
                    visit.append(successor) #방문하였기 때문에 노드를 방문 리스트에 삽입
    return []
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    state = problem.getStartState() #현재 상태
    mazequeue = util.PriorityQueue()    #큐 선언
    visit = set()  #방문 집합 선언
    mazequeue.push((state, []), 0) #큐에 루트 노드 삽입, 초기 거리는 0

    while True:
        if mazequeue.isEmpty(): #큐가 비어 있으면 종료
            break

        node, direction = mazequeue.pop()    #큐에 있는 값을 빼옴.
        if(node not in visit):  #해당 노드가 방문하지 않았으면
            visit.add(node) #방문 집합에 넣음
        else:
            continue    #그렇지 않으면 아래 코드전부 무시하고 다시 시작

        if problem.isGoalState(node):   #문제의 골에 도착하면  여태까지 간 거리(노드)를 반환시킴. 
            return direction
        else:
            for successor, action, stepcost in problem.getSuccessors(node): #방문 할 수 있는 노드들 반복
                if successor not in visit:  #해당 위치가 방문되어 있지 않으면 비용 계산하여 업데이트
                    cost = problem.getCostOfActions(direction + [action])   #새로운 비용 계산
                    mazequeue.update((successor, direction + [action]), stepcost + cost)    #큐에 있는 값을 수정, 큐에 없으면 추가

    return []
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    state = problem.getStartState() #현재 상태
    mazequeue = util.PriorityQueue()    #큐 선언
    visit = []  #방문 리스트 선언
    mazequeue.push((state, []), heuristic(state, problem)) #큐에 루트 노드 삽입, 휴리스틱 삽입

    while True:
        if mazequeue.isEmpty(): #큐가 비어 있으면 종료
            break

        node, direction = mazequeue.pop()    #큐에 있는 값을 빼옴.

        if(node not in visit):  #해당 노드가 방문하지 않았으면
            visit.append(node) #방문 리스트에 넣음
        else:
            continue    #그렇지 않으면 아래 코드전부 무시하고 다시 시작

        if problem.isGoalState(node):   #문제의 골에 도착하면  여태까지 간 거리(노드)를 반환시킴. 
            return direction
        else:
            for successor, action, stepcost in problem.getSuccessors(node): #방문 할 수 있는 노드들 반복
                if successor not in visit:  #해당 위치가 방문되어 있지 않으면 비용 계산하여 업데이트
                    cost = problem.getCostOfActions(direction) + heuristic(successor, problem)   #새로운 비용 계산
                    mazequeue.update((successor, direction + [action]), (stepcost + cost))    #큐에 있는 값을 수정, 큐에 없으면 추가
    return []
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
