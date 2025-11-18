# Hyperparameter tuning with Bayesian optimization

## **Task**

In this task, you will use Bayesian optimization to tune the structural features of a drug candidate, which affects its absorption and distribution. These features should be optimized subject to a constraint on how difficult the candidate is to synthesize. Let $x \in \mathcal{X}$ be a parameter that quantifies such structural features. We want to find a candidate with $x$ that is 1) bioavailable enough to reach its intended target, and 2) easy to synthesize. We use logP as our objective — a coarse proxy for bioavailability.

To this end, for a specific $x$, we simulate the candidate's corresponding logP as well as its synthetic accessibility score (SA), which is a proxy for how difficult it is to synthesize. Our goal is to find the structural features $x^*$ that induce the highest possible logP while satisfying a constraint on synthesizability.

More formally, let us denote with
$f : \mathcal{X} \rightarrow [0,1]$
the mapping from structural features to the candidate’s corresponding bioavailability (logP), which we assume is bounded. Given a candidate with $x$, we observe

$$
y_f = f(x) + \varepsilon_f,
\quad \varepsilon_f \sim \mathcal{N}(0, \sigma_f^2),
$$

which is zero mean Gaussian i.i.d. noise.

Moreover, we denote with
$v : \mathcal{X} \rightarrow \mathbb{R}^+$
the mapping from structural features to the corresponding synthetic accessibility (SA). Similar to our objective logP, we observe a noisy value of this score,

$$
y_v = v(x) + \varepsilon_v,
\quad \varepsilon_v \sim \mathcal{N}(0, \sigma_v^2).
$$

The problem is formalized as:

$$
x^* \in \arg\max_{x \in \mathcal{X},, v(x) < \kappa} f(x),
$$

where $\kappa$ is the maximum tolerated synthetic accessibility (SA). Compounds with higher SA are more difficult to synthesize. The objective has no analytical expression, is computationally expensive to evaluate, and is only accessible through noisy evaluations. Therefore, it is well suited for Bayesian optimization (see [1] for further reading on an example of constrained Bayesian optimization in drug discovery).

You need to solve the drug discovery problem presented above with Bayesian optimization. Let $x_i$ be evaluated at the $i^{th}$ iteration of the Bayesian optimization algorithm. While running the algorithm, you have a fixed budget for trying hyperparameters for which the synthesizability constraint is violated, i.e.

$$
v(x_i) \ge \kappa.
$$

Furthermore, the final solution must satisfy:

$$
v(x) < \kappa.
$$

### **Remarks**

In the motivating example above, $x$ takes discrete values (e.g., the number of functional groups would be a natural number) and the objective and constraint can be evaluated independently. However, to keep the problem simple, we let $x$ be continuous and we evaluate $f$ and $v$ simultaneously. Moreover, to avoid unfair advantages due to differences in computational power, the physiochemical properties of the molecule are simulated, therefore the time required for this step is platform-independent. This task does not have a private score.

Below, you can find the quantitative details of this problem:

* The domain is $\mathcal{X} = [0,10]$.
* The noise perturbing the observation is Gaussian with standard deviation
  $\sigma_f = 0.15$ and $\sigma_v = 0.0001$ for logP and SA, respectively.
* The mapping $f$ can be effectively modeled with a Matérn kernel ($\nu = 2.5$) or an RBF kernel with variance $0.5$, lengthscale $10$, $1$, or $0.5$. Kernel and lengthscale should be tuned for best results.
* The mapping $v$ can be effectively modeled with an additive kernel composed of a Linear kernel and a Matérn kernel ($\nu = 2.5$) or an RBF kernel with variance $\sqrt{2}$, lengthscale $10$, $1$, or $0.5$. The prior mean should be $4$. Kernel and lengthscale should be tuned for best results.
* The maximum tolerated SA is $\kappa = 4$.

We specify the following baseline, which needs to be beaten to achieve a passing grade: **0.785**.

---

## **Submission Workflow**

1. Install and start Docker.
2. Download the handout.
3. Implement all TODOs in `solution.py` within the `BO_algo` class. The `main()` method is ignored by the checker.
4. Use Python 3.8.5. Additional libraries must be listed in the Dockerfile.
5. Run the checker using Docker (instructions differ for Linux, macOS, and Windows).
6. If the checker runs successfully, it generates a `results_check.byte` file. Upload it along with your code and a one-minute video.
7. You pass this task if your score is above the baseline.
8. Submission limits: 40 per team, with at most 20 in 24 hours.

---

## **Evaluation**

We are interested in minimizing the normalized regret for not knowing the best hyperparameter value.

Let $\tilde{x}_j$ be the final optimal solution suggested by the algorithm for a randomly initialized drug discovery task $j$. Let $N_j$ be the number of unsafe evaluations that violate the constraint $v(x) < \kappa$.

We do not penalize staying within the safe set around the initial safe point $x^{\blacktriangle}$, and we use the local optimum $x^\bullet$.

Define normalized regret:

$$
r_j = \max\left(
\frac{f(x^\bullet) - f(\tilde{x}_j)}{f(x^\bullet)},, 0
\right).
$$
