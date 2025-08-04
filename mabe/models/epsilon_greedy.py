from typing import Union

import numpy as np

# TODO: Add save_model and load_model methods.
class EpsilonGreedyPolicy:
    """Class containing an implementation of the Epsilon-Greedy algorithm for Multi-Armed Bandits.
    
    This implementation initialises the average rewards of all the arms at 0, and updates them after each time step.
    
    Attributes:
        epsilon: Exploration parameter of the model. Defines the probability with which the model chooses an action
        different from the one that is expected to be the best.
        n_arms: Number of arms of the bandit problem.
        q_values: Vector with the experimental average rewards of all arms.
        action_counts: Vector with the count of how many times each action has been chosen.
        """
    def __init__(self, n_arms: int, epsilon: float = 0.1) -> None:
        """Initialisation of a class instance.
        
        Args:
            epsilon: Exploration parameter of the model. Defines the probability with which the model chooses an action
            different from the one that is expected to be the best.
            n_arms: Number of arms of the bandit problem.
        """
        self.epsilon = epsilon
        self.n_arms = n_arms
        self.q_values = np.zeros(n_arms)
        self.action_counts = np.zeros(n_arms)

    def select_action(self, x_contexts: np.ndarray) -> Union[int, np.int64]:
        """Method to choose an action or arm following the algorithm policy.
        
        This method suggests which arm to choose following the equations of the Epsilon-Greedy algorithm.
        
        Args:
            x_contexts: Vector with the context for an specific time step. It is ignored given that the algorithm
            is not contextual.
        
        Returns:
            The number of the arm that should be chosen according to the algorithm.
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_arms)
        return np.argmax(self.q_values)

    def update(self, action: int, x: np.ndarray, reward: float) -> None:
        """Update method of the model.
        
        This method is responsible for updating the parameters of the model after each time step (after an action is 
        selected and the reward of that choice is returned).
        
        Args:
            action: Action or arm chosen.
            x: Context associated to that time step.
            reward: Reward returned from the action chosen.
        """
        self.action_counts[action] += 1
        n_t_a = 1 / self.action_counts[action]
        self.q_values[action] += n_t_a * (reward - self.q_values[action])