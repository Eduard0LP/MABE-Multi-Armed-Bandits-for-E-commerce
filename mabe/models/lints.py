import numpy as np

class LinTS:
    """Class containing an implementation of the Linear Thompson Sampling (LinTS) algorithm for Contextual
    Multi-Armed Bandits.
    
    This implementation initialises the linear regresion coefficient matrices to the identity matrix and the vectors of
    independent terms to 0 and considers a multivariate normal distribution N(mu, sigma) as the prior distribution of
    the rewards. Both the coefficient matrices and vectors of independent terms get updated after each time step 
    based on the reward obtained.
    
    Attributes:
        n_arms: Number of arms of the bandit problem.
        d: Dimension of the context vector that the algorithm should expect.
        v: Parameter of the LinTS algorithm that scales the values of the covariance matrix of the multivariate normal
        distribution. It controls exploration.
        A: Linear regresion coefficient matrices.
        b: Vectors of independent terms of the linear regressions.
    """
    def __init__(self, n_arms: int, context_dim: int, v: float = 1.0) -> None:
        """Initialisation of a class instance.
        
        Args:
            n_arms: Number of arms of the bandit problem.
            context_dim: Dimension of the context vector that the algorithm should expect.
            v: Parameter of the LinTS algorithm that scales the values of the covariance matrix of the multivariate 
            normal distribution. It controls exploration.
        """
        self.n_arms = n_arms
        self.d = context_dim
        self.v = v

        self.A = [np.identity(self.d) for _ in range(n_arms)]  # A_a = d x d
        self.b = [np.zeros((self.d, 1)) for _ in range(n_arms)]  # b_a = d x 1

    def select_action(self, x_contexts: np.ndarray) -> np.int64:
        """Method to choose an action or arm following the algorithm policy.
        
        This method suggests which arm to choose following the equations of the LinTS algorithm.
        
        Args:
            x_contexts: Vector with the context for an specific time step. It has shape (d, 1).
        
        Returns:
            The number of the arm that should be chosen according to the algorithm.
        """
        p_values = []

        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            mu = A_inv @ self.b[a]
            cov = self.v ** 2 * A_inv

            theta_sample = np.random.multivariate_normal(mu.flatten(), cov).reshape(-1, 1)
            x = x_contexts[a]

            p = (theta_sample.T @ x)[0, 0]
            p_values.append(p)

        return np.argmax(p_values)

    def update(self, action: int, x: np.ndarray, reward: float) -> None:
        """Update method of the model.
        
        This method is responsible for updating the parameters of the linear regression of the model after each time
        step (after an action is selected and the reward of that choice is returned).
        
        Args:
            action: Action or arm chosen.
            x: Context associated to that time step.
            reward: Reward returned from the action chosen.
        """
        self.A[action] += x @ x.T
        self.b[action] += reward * x