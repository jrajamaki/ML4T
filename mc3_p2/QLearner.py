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
        @param dyna integer, run this number of dyna optimisations.
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
        self.dyna_model = np.empty((0, 4), dtype=int)
        self.state = 0
        self.action = 0

    def querysetstate(self, s):
        """
        @summary: Updates the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.state = s
        self.action = self._generate_next_action()

        return self.action

    def query(self, s_prime, r):
        """
        @summary: Updates learner status, runs optionally Dyna optimisation
        @param s_prime: The new state
        @param r: The reward in the new state
        @returns: The next action to take
        """

        # Q-learning algorithm steps:
        # 1) Update the Q table with new experience tuple <s,a,s_prime,r>
        # 2) Take a next action from new state either through
        # look-up table Q or by random generation of random chance self.rar
        # 3) Update rar by deteriorate rate of radr
        self._update_Q_table(self.state, self.action, s_prime, r)

        # Optional Q-learning Dyna algorithm
        # 1) Choose random s and a
        # 2) Query transition table T and determine new state prime_s
        # 3) Query reward table R and determine reward r
        # 4) Update Q table
        # Iterate over above algorithm

        # In this implementation each experience tuple is saved
        # and data is chosen from experience tuple list,
        # thus every time new experience tuple is added and only then
        # Dyna optimisation is run
        if self.dyna != 0:
            new_exp_tuple = np.array([[self.state, self.action, s_prime, r]])
            self.dyna_model = np.r_[self.dyna_model, new_exp_tuple]
            self._run_dyna_optimisation()

        # Update learner status
        self.state = s_prime
        self.action = self._generate_next_action()
        self.rar *= self.radr

        return self.action

    def _generate_next_action(self):
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
        if type(s_prime) is np.ndarray:
            a_prime = np.argmax(self.Q[s_prime, :], axis=1)
        else:
            a_prime = np.argmax(self.Q[s_prime, :])

        new_Q_value = (1 - self.alpha) * self.Q[s, a] + \
                        self.alpha * (reward +
                                      self.gamma * self.Q[s_prime, a_prime])
        self.Q[s, a] = new_Q_value

    def _run_dyna_optimisation(self):
        """
        @summary: Runs optional Dyna optimisation
        """

        # From within all experience tuples randomly choose 
        # 'self.dyna' amount by index
        random_index = np.random.choice(self.dyna_model.shape[0],
                                        size=self.dyna,
                                        replace=True)

        # Set dyna variables for optimisation
        s = self.dyna_model[random_index, 0]
        a = self.dyna_model[random_index, 1]
        s_prime = self.dyna_model[random_index, 2]
        r = self.dyna_model[random_index, 3]

        self._update_Q_table(s, a, s_prime, r)


if __name__ == "__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
