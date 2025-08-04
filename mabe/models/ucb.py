import numpy as np

class UCBBandit:
    """Class containing an implementation of the Upper Confidence Bound (UCB) algorithm for Multi-Armed Bandits.
    
    This implementation initialises the average rewards of all the arms at 0, and updates them after each time step.
    
    Attributes:
        n_arms: Number of arms of the bandit problem.
        q_values: Vector with the experimental average rewards of all arms.
        action_counts: Vector with the count of how many times each action has been chosen.
        total_counts: Total amount of time steps the algorithm has taken.
        alpha: Parameter of the UCB algorithm that controls exploration.
    """
    def __init__(self, n_arms: int, alpha: float = 1.0) -> None:
        """Initialisation of a class instance.
        
        Args:
            n_arms: Number of arms of the bandit problem.
            alpha: Parameter of the UCB algorithm that controls exploration.
        """
        self.n_arms = n_arms
        self.q_values = np.zeros(n_arms)
        self.action_counts = np.zeros(n_arms)
        self.total_counts = 0
        self.alpha = alpha

    def select_action(self, x_contexts: np.ndarray) -> np.int64:
        """Method to choose an action or arm following the algorithm policy.
        
        This method suggests which arm to choose following the equations of the UCB algorithm.
        
        Args:
            x_contexts: Vector with the context for an specific time step. It is ignored given that the algorithm
            is not contextual.
        
        Returns:
            The number of the arm that should be chosen according to the algorithm.
        """
        self.total_counts += 1
        # The value 1e-5 is a small value used to avoid dividing by 0
        ucb_values = self.q_values + self.alpha * np.sqrt(np.log(self.total_counts) / (self.action_counts + 1e-5)) 
        return np.argmax(ucb_values)

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