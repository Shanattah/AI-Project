import numpy as np


class SarsaAgent:
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.Q = np.zeros((num_states, num_actions))
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate

    def act(self, state):
        if np.random.uniform() < self.epsilon:
            # Explore by choosing a random action
            action = np.random.randint(self.Q.shape[1])
        else:
            # Exploit by choosing the best action based on Q-values
            action = np.argmax(self.Q[state])
        return action

    def update(self, state, action, reward, next_state, next_action):
        # SARSA update rule
        td_error = (
            reward
            + self.gamma * self.Q[next_state][next_action]
            - self.Q[state][action]
        )
        self.Q[state][action] += self.alpha * td_error
