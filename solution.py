"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, DotProduct, RBF, WhiteKernel, DotProduct
from scipy.stats import norm


# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA


# TODO: implement a self-contained solution in the BO_algo class.
# NOTE: main() is not called by the checker.
class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        # TODO: Define all relevant class members for your BO algorithm here.
        # we need to keep track of observed data points {x, f, v}
        self.X_observed = None  # shape (n_samples, n_features)
        self.f_observed = None  # shape (n_samples, )
        self.v_observed = None  # shape (n_samples, )

        # It is recommended to tune the kernel hyperparameters and lengthscales. We also have noisy observations so we can add a WhiteKernel if needed. Noise for f is gaussian with std 0.15 and for v is gaussian with std 0.0001
        # the mapping f can be modeled with a Matern kernel of nu=2.5 or RBF kernel with variance 0.5 and lengthscale 10, 1 or 0.5.
        self.f_kernel = ConstantKernel(
            constant_value=0.5) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.15**2)
        self.f_gp = GaussianProcessRegressor(
            kernel=self.f_kernel, n_restarts_optimizer=5, normalize_y=True)

        # the mapping v can be modeled with a Linear kernel + Mathern kernel of nu=2.5 or RBF kernel with variance sqrt(0.1) and lengthscale 10, 1 or 0.5. The mean should be 4 but is usually 0 so need to offset when outputting the prediction and when adding data points.
        self.v_kernel = ConstantKernel(constant_value=0.1) * DotProduct(
            sigma_0=0.0) + Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.0001**2)
        # no normalize_y here since we need to offset the mean to 4
        self.v_gp = GaussianProcessRegressor(
            kernel=self.v_kernel, n_restarts_optimizer=5)

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            the next point to evaluate
        """
        # TODO: Implement the function which recommends the next point to query
        # using functions f and v.
        # In implementing this function, you may use
        # optimize_acquisition_function() defined below.

        x_opt = self.optimize_acquisition_function()
        return np.array([[x_opt]])

    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick the best solution
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                np.random.rand(DOMAIN.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return x_opt

    def acquisition_function(self, x: np.ndarray):
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        x = np.atleast_2d(x)
        # TODO: Implement the acquisition function you want to optimize.

        mu_f, sigma_f = self.f_gp.predict(x, return_std=True)
        mu_v, sigma_v = self.v_gp.predict(x, return_std=True)
        mu_v += 4  # offset mean to 4

        # to avoid division by zero
        sigma_v = np.maximum(sigma_v, 1e-9)

        # Lagrangian relaxation max f(x) - lambda * max(v(x) - 4, 0)
        relax_lambda = 1.75  # starting with 2.0

        # Probability of safety P(v(x) <= SAFETY_THRESHOLD)
        probability_of_safety = norm.cdf(
            (SAFETY_THRESHOLD - mu_v) / sigma_v
        )

        # Expected Improvement from exercise 4.2
        valid_indexes = np.where(self.v_observed <= SAFETY_THRESHOLD)[0]
        best_f_safe = np.max(self.f_observed[valid_indexes])

        improvement = mu_f - best_f_safe
        frac = improvement / sigma_f
        expected_improvment = improvement * \
            norm.cdf(frac) + sigma_f * norm.pdf(frac)

        # return expected_improvment * probability_of_safety
        # return expected_improvment * relax_lambda * probability_of_safety

        # Expected penalty E[max(v(x) - 4, 0)]
        z = (mu_v - SAFETY_THRESHOLD) / sigma_v
        expected_penalty = (mu_v - SAFETY_THRESHOLD) * \
            norm.cdf(z) + sigma_v * norm.pdf(z)

        # UCB from exercise 4.2
        beta = 0.75  # starting with 2.0
        ucb_f = mu_f + beta * sigma_f

        # lagrangian acquisition function
        # return ucb_f - relax_lambda * np.maximum(mu_v - SAFETY_THRESHOLD, 0)
        return ucb_f - relax_lambda * expected_penalty
        # return expected_improvment * relax_lambda * expected_penalty
        # return expected_improvment * probability_of_safety - relax_lambda * expected_penalty

    def add_data_point(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float or np.ndarray
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """
        # TODO: Add the observed data {x, f, v} to your model.
        self.X_observed = np.vstack(
            [self.X_observed, x]) if self.X_observed is not None else x
        self.f_observed = np.vstack(
            [self.f_observed, f]) if self.f_observed is not None else f
        self.v_observed = np.vstack(
            [self.v_observed, v]) if self.v_observed is not None else v

        # needs to be 2D arrays for sklearn
        self.X_observed = self.X_observed.reshape(-1, 1)
        self.f_observed = self.f_observed.reshape(-1, 1)
        self.v_observed = self.v_observed.reshape(-1, 1)

        self.f_gp.fit(self.X_observed, self.f_observed)
        self.v_gp.fit(self.X_observed, self.v_observed - 4)  # offset mean to 4

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        # TODO: Return your predicted safe optimum of f.
        valid_indexes = np.where(self.v_observed <= SAFETY_THRESHOLD)[0]
        best_index = valid_indexes[np.argmax(self.f_observed[valid_indexes])]

        return self.X_observed[best_index]

    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        pass


# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---

def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])


def f(x: float):
    """Dummy logP objective"""
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return - np.linalg.norm(x - mid_point, 2)


def v(x: float):
    """Dummy SA"""
    return 2.0


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]

    return x_init


def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, DOMAIN.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {DOMAIN.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x) + np.randn()
        cost_val = v(x) + np.randn()
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = agent.get_solution()
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the' \
        f'DOMAIN, {solution} returned instead'

    # Compute regret
    regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}\nUnsafe-evals TODO\n')


if __name__ == "__main__":
    main()
