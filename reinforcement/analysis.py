# analysis.py
# -----------
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


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():	#직선으로만 가야함
    answerDiscount = 0.9
    answerNoise = 0.0
    return answerDiscount, answerNoise

def question3a():	#가까운 거리로 가야함
    answerDiscount = 0.4
    answerNoise = 0.0
    answerLivingReward = -0.5
    return answerDiscount, answerNoise, answerLivingReward

def question3b():	#가까운 거리로 가되, 위로 가기 위해 어긋나게 가는 노이즈 추가
    answerDiscount = 0.4
    answerNoise = 0.2
    answerLivingReward = -0.5
    return answerDiscount, answerNoise, answerLivingReward

def question3c():	#멀리 있는 보상으로 가되, 빠른길로 가게 만듬.
    answerDiscount = 1.0
    answerNoise = 0.0
    answerLivingReward = -0.5
    return answerDiscount, answerNoise, answerLivingReward

def question3d():	#멀리 있는 보상으로 가되, 위로 가기 위해 어긋나게 가는 노이즈 추가
    answerDiscount = 1.0
    answerNoise = 0.2
    answerLivingReward = -0.5
    return answerDiscount, answerNoise, answerLivingReward

def question3e():	#아무데도 안가게 만들기
    answerDiscount = 0.0
    answerNoise = 0.0
    answerLivingReward = 10000
    return answerDiscount, answerNoise, answerLivingReward

def question8():	#입실론과 학습률을 어떻게 줘도 터미널에 보상이 있으므로 불가능
    answerEpsilon = None
    answerLearningRate = None
    # return answerEpsilon, answerLearningRate
    return 'NOT POSSIBLE'

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))