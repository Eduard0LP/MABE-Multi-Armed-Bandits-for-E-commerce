import numpy as np

class ThompsonSampling:
    """Class containing an implementation of the Thompson Sampling (TS) algorithm for Multi-Armed Bandits.
    
    This implementation considers a beta distribution B(alpha, beta) as the prior distribution of the rewards. 
    The parameters of the distribution are both initialised to 1.

    Attributes:
        n_arms: Number of arms of the bandit problem.
        successes: Vector with the amount of positive rewards of the suggestions of each arm plus one. This corresponds
        to the alpha parameter of the beta distribution. 
        failures: Vector with the amount of negative rewards (no reward) of the suggestions of each arm plus one. This 
        corresponds to the beta parameter of the beta distribution. .
    """
    def __init__(self, n_arms: int) -> None:
        """Initialisation of a class instance.
        
        Args:
            n_arms: Number of arms of the bandit problem.
        """
        self.n_arms = n_arms
        self.successes = np.ones(n_arms)
        self.failures = np.ones(n_arms)

    def select_action(self, x_contexts: np.ndarray) -> np.int64:
        """Method to choose an action or arm following the algorithm policy.
        
        This method suggests which arm to choose following the equations of the TS algorithm.
        
        Args:
            x_contexts: Vector with the context for an specific time step. It is ignored given that the algorithm
            is not contextual.
        
        Returns:
            The number of the arm that should be chosen according to the algorithm.
        """
        sampled_means = np.random.beta(self.successes, self.failures)
        return np.argmax(sampled_means)

    def update(self, action: int, x: np.ndarray, reward: float) -> None:
        """Update method of the model.
        
        This method is responsible for updating the parameters of the model after each time step (after an action is 
        selected and the reward of that choice is returned).
        
        Args:
            action: Action or arm chosen.
            x: Context associated to that time step.
            reward: Reward returned from the action chosen.
        """
        if reward == 1:
            self.successes[action] += 1
        else:
            self.failures[action] += 1