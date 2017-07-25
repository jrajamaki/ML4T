"""
Q-learning algorithm with Dyna capabilities
"""

import numpy as np
import random as rand


class QLearner(object):

    def __init__(self,
                 num_states=100,
                 num_actions=4,
                 alpha=0.2,
                 gamma=0.9,
                 rar=0.5,
                 radr=0.99,
                 dyna=0,
                 verbose=False):
        """
        @summary: Constructor for the Q learner
        @param num_states: integer, the number of states to consider
        @param num_actions integer, the number of actions available.
        @param alpha float [0, 1], the learning rate used in the update rule.
        @param gamma float [0, 1], the discount rate used in the update rule.
        @param rar float [0, 1], random action rate.
        @param radr float [0, 1], random action decay rate: rar = rar * radr.
        @param dyna integer, conduct this number of dyna updates.
        @param verbose boolean, if True, debugging statements are allowed.
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.verbose = verbose

        self.Q = np.zeros(shape=(num_states, num_actions))
        self.state = 0
        self.action = 0

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.state = s
        self.action = self._next_action()

        if self.verbose:
            print "s =", self.state, "a =", self.action

        return self.action

    def query(self, s_prime, r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The reward in the new state
        @returns: The selected action
        """

        # Q-learning algorithm steps:
        # 1) Update the Q table with new experience tuple <s,a,s_prime,r>
        # 2) Decide whether to take random action or not, update rar
        # 3) If yes return random action
        # 4) If no, query Q table to find optimal action
        self._update_Q_table(self.state, self.action, s_prime, r)

        self.state = s_prime
        self.action = self._next_action()
        self.rar *= self.radr

        # Q-learning Dyna algorithm
        # In addition hallucinate an experience:
        # 1) Choose random s and a
        # 2) Query table T and determine new state prime_s according to probs
        # 3) Query table R and determine reward r
        # 4) Update Q table
        # Iterate over above algorithm

        # Tip: Dyna hallucinates from experiences already had
        # it is not necessary to create T or R tables
        self._dyna()

        if self.verbose:
            print "s =", s_prime, "a =", self.action, "r =", r

        return self.action

    def _next_action(self):
        """
        @summary: Decides next action either through look up or randomly
        @returns: Action to take
        """
        probs = [self.rar, 1 - self.rar]
        random_action = np.random.choice([True, False], p=probs)
        if random_action:
            action = np.random.randint(self.num_actions)
        else:
            action = np.argmax(self.Q[self.state, :])

        return action

    def _update_Q_table(self, s, a, s_prime, reward):
        """
        @summary: Updates lookup table Q
        """
        a_prime = np.argmax(self.Q[s_prime, :])
        new_Q_value = (1 - self.alpha) * self.Q[s, a] + \
                        self.alpha * (reward +
                                      self.gamma * self.Q[s_prime, a_prime])

        self.Q[s, a] = new_Q_value

    def _dyna(self):
        pass
        '''
        random_s = np.random.randint(self.num_states, size=self.dyna)
        random_a = np.random.randint(self.num_actions, size=self.dyna)

            for d in range(self.dyna):
                # random initial state and action
                random_s = np.random.randint(self.num_states)
                random_a = np.random.randint(self.num_actions)

                #
                states, counts = np.unique(self.Q[random_s, :], return_counts=True)
                for i in range(len(states)):
                    gen_s = int(states[i])
                    gen_r = self.Q[gen_s, random_a]
                    gen_r *= counts[i] / counts.sum()
                    self._update_Q_table(random_s, random_a, gen_s, gen_r)
        '''

if __name__ == "__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
