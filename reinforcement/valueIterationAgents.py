# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*
        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.
          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):		#위에 정해진 수만큼 MDP 반복
            counter = util.Counter()			#맵을 만듬
            for state in self.mdp.getStates():	#모든 상태의 수만큼 반복, 각 상태를 가지고 옴
                max_val = -999999				#최대값
                for action in self.mdp.getPossibleActions(state):	#위의 상태에서 모든 가능한 액션 수만큼 반복
                    q_val = self.computeQValueFromValues(state, action)	#Q-value값을 구함.
                    if q_val > max_val:	#Q-value값이 최대값보다 크면 갱신
                        max_val = q_val
                    counter[state] = max_val	#해당 맵에 최대값 집어넣음, 이를 통해 최선의 정책을 얻을 수 있음.

            self.values = counter	#나중에 모든 최선의 정책값들을 대입넣음



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        q_value = 0
        for ele in self.mdp.getTransitionStatesAndProbs(state, action):	#nextState로 갈 확률 수만큼 반복
            next_s = ele[0]	#다음 상태 정보
            prob = ele[1]	#다음 상태 갈 확률 정보
            reward = self.mdp.getReward(state, action, next_s)	#그에 따른 보상값 가져오기
            q_value += prob * (reward + self.discount * self.values[next_s])	#Q-value 구함

        return q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.
          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        act_dic = {}	#액션리스트

        for action in self.mdp.getPossibleActions(state):	#움직일 수 있는 액션리스트
            tmp_q = self.computeQValueFromValues(state, action)	#임시 Q-value 값
            act_dic[action] = tmp_q	#임시 Q-value 값을 액션리스트에 넣음
        if not act_dic:	#액션 리스트에 값이 아무것도 없으면 없다고 출력
            return None
        return max(act_dic, key=lambda x: act_dic[x])	#그게 아니라면 액션리스트의 값중 가장 큰값 출력

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*
        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.
          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        state_lst = self.mdp.getStates()	#상태 리스트
        # update one state each iteration
        for i in range(self.iterations):	#반복 횟수에 따른 반복
            state = state_lst[i % len(state_lst)]	#순차적으로 state를 가져옴
            if self.mdp.isTerminal(state):	#터미널이면 종료
                pass
            else:
                action = self.computeActionFromValues(state)	#Value를 기반으로 액션가져오기
                self.values[state] = self.computeQValueFromValues(state, action)	#Q-value 값 가져와서 value에 저장


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*
        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # Compute predecessors of all states.
        pre_lst = {}	#error 검사 리스트
        for s in self.mdp.getStates():	#모든 state 반복
            for action in self.mdp.getPossibleActions(s):	#모든 가능한 action 반복
                for succ, prob in self.mdp.getTransitionStatesAndProbs(s, action):	#다음 state와 확률을 가져옴
                    if succ in pre_lst:	#다음 state가 이미 리스트 있다면
                        pre_lst[succ].add(s)	#해당 위치에 state값을 집어넣음
                    else:
                        pre_lst[succ] = {s}	#그게 아니라면 초기이기 때문에 state 리스트 1개로 정의

        # Initialize an empty priority queue.
        prique = util.PriorityQueue()	#큐

        for s in self.mdp.getStates():	#모든 state 반복
            if self.mdp.isTerminal(s):	#state가 터미널이면 종료
                pass
            else:
                action = self.computeActionFromValues(s)	#액션을 가져옴
                maximum = self.computeQValueFromValues(s, action)	#Q-value 값으로 최대값을 구함
                diff = abs(maximum - self.values[s])	#차이값을 구함
                prique.update(s, -diff)	#큐에 갱신시킴, -가 클수록 늦게 처리

        for i in range(self.iterations):	#반복 횟수만큼 반복
            if prique.isEmpty():	#큐에 아무것도 없으면 종료
                return

            cur_s = prique.pop()	#큐에 있는 값을 가져옴
            if self.mdp.isTerminal(cur_s):	#근데 그게 터미널이면 종료
                pass
            else:
                action = self.computeActionFromValues(cur_s)	#액션을 가져옴
                self.values[cur_s] = self.computeQValueFromValues(cur_s, action)	#value 값 갱신

            for pre in pre_lst[cur_s]:	#위에서 error 검사 리스트 가져옴
                if not self.mdp.isTerminal(pre):	#터미널이면 안됨
                    action = self.computeActionFromValues(pre)	#액션을 가지고 옴
                    maximum = self.computeQValueFromValues(pre, action)	#Q-value 값으로 최대값을 구함
                    diff = abs(maximum - self.values[pre])	#diff 공식 구하기
                    if diff > self.theta:	#만약 위에 정의한 세타값보다 적으면 큐에 다시 업데이터함
                        prique.update(pre, -diff)