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
        # V_k+1 = max()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        for i in range(self.iterations):
            cur = self.values.copy()
            for state in states:
                if self.mdp.isTerminal(state):
                    continue
                actions = self.mdp.getPossibleActions(state)
                qValue = []
                for action in actions:
                    qValue.append(self.computeQValueFromValues(state, action))
                cur[state] = max(qValue)
            self.values = cur   


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
        if self.mdp.isTerminal(state):
            return 0
        values = 0
        statesProb = self.mdp.getTransitionStatesAndProbs(state, action)
        for s, prob in statesProb:
            if not self.mdp.isTerminal(state):
                value = self.mdp.getReward(state, action, s) + self.discount *  self.getValue(s)
                value = value * prob
                values += value
        return values
        

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        results = {}
        actions = self.mdp.getPossibleActions(state)
        for action in actions:
            # children = self.mdp.getTransitionStatesAndProbs(state, action)
            # for child, prob in children:
            #     # if child:
            #     #     value += self.values.get(child) * prob
            #     results[action] = self.getValue(child)
            results[action] = self.getQValue(state, action)
        c = util.Counter(results)        
        return c.argMax()


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
        states = self.mdp.getStates()

        for i in range(self.iterations):
            index = i % len(states)
            state = states[index] 
            if self.mdp.isTerminal(state):
                continue  
            actions = self.mdp.getPossibleActions(state)
            qValue = []
            for action in actions:
                qValue.append(self.computeQValueFromValues(state, action))
            self.values[state] = max(qValue)     



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
        predecessors = {}
        states = self.mdp.getStates()
        for state in states:
            predecessors[state] = []
            
        queue = util.PriorityQueue()
        for state in states:
            if not self.mdp.isTerminal(state):
                for action in self.mdp.getPossibleActions(state):
                    successors = self.mdp.getTransitionStatesAndProbs(state, action)
                    for successor, prob in successors:
                        if not prob == 0.0:
                            
                            if state not in predecessors[successor]:
                                curList = predecessors[successor]
                                curList.append(state)
                                predecessors[successor] = curList
        for state in states:
            if self.mdp.isTerminal(state):
                continue
            qValue = []
            for action in self.mdp.getPossibleActions(state):
                qValue.append(self.getQValue(state, action))
            highest = max(qValue)    
            queue.push(state, 0 - abs(self.getValue(state) - highest))

        for i in range(self.iterations):
            if queue.isEmpty():
                break
            cur = queue.pop()
            actions = self.mdp.getPossibleActions(cur)
            qValue = []
            for action in actions:
                qValue.append(self.computeQValueFromValues(cur, action))
            self.values[cur] = max(qValue)  

            for predecessor in predecessors[cur]:
                qValue = []
                for action in self.mdp.getPossibleActions(predecessor):
                    qValue.append(self.getQValue(predecessor, action))
                highest = max(qValue)
                diff = abs(self.getValue(predecessor) - highest)
                if diff > self.theta:
                    queue.update(predecessor, 0 - diff)





