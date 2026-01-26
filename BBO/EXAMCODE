# =============================================================
# MLCE_Final_Submission_BO_FINAL_30S.py
#
# Constraints satisfied
# 6 init evaluations
# 15 batches, each batch is one objective call with 5 points
# no cross run memory
# no pandas
#
# Key performance drivers combined
# cached global Sobol pool per run
# dynamic slice of global pool per batch
# dynamic local cloud around within run best
# shortlist once per batch, then Kriging Believer only on shortlist
# mean gated uncertainty for batches 1 and 2
# EI for batches 3 to 15 with mild UCB boost for early picks of batch 3
# rescue trigger after batch 3 if best_overall < 300, applied in batches 4 and 5
# safe handling of invalid outputs, and avoidance of repeats and known bad points
# time aware downscaling when close to target runtime
# =============================================================

import os, sys, time, random
import numpy as np
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from scipy.optimize import minimize
from scipy.stats import norm
from sobol_seq import i4_sobol

sys.path.append(os.path.abspath(
    r"C:\Imperial\Year 4\ML\Imperial-ML4CE-Course\BatchBayesianOptimization"
))
import MLCE_CWBO2025.virtual_lab as virtual_lab


def objective_func(X_list):
    custom_init = [0, 0.4e9, 0.4e6, 0, 20, 3.5, 0, 1.8]
    y = np.array(
        virtual_lab.conduct_experiment(X_list, initial_conditions=custom_init),
        dtype=float
    )
    return y


BOUNDS = np.array([
    [30.0, 40.0],
    [6.0,  8.0],
    [0.0, 50.0],
    [0.0, 50.0],
    [0.0, 50.0],
], dtype=float)

CELLTYPES = ["celltype_1", "celltype_2", "celltype_3"]
#What this section does

# This defines the optimisation domain.

# There are 5 continuous variables, each with lower and upper bounds.

# There is 1 categorical variable with 3 valid labels.

# This immediately tells us:

# The problem is mixed-variable

# The optimisation space is bounded

# The optimiser must never evaluate outside this region.

# Why this exists

# Bayesian Optimisation:

# needs bounds to generate candidates

# needs bounds to scale inputs

# must avoid invalid experiments/simulations

# Oral exam explanation (what to say)

# “This section defines the feasible optimisation space. The problem consists of five bounded continuous variables and one categorical variable. These bounds are enforced throughout the optimisation to guarantee validity of all candidate points.”


def _safe_y(y, bad_value=-1e6):
    y = np.asarray(y, dtype=float)
    y[~np.isfinite(y)] = bad_value
    return y


def clip_x(x):
    t, pH, f1, f2, f3, ct = x
    t = float(np.clip(t, BOUNDS[0, 0], BOUNDS[0, 1]))
    pH = float(np.clip(pH, BOUNDS[1, 0], BOUNDS[1, 1]))
    f1 = float(np.clip(f1, BOUNDS[2, 0], BOUNDS[2, 1]))
    f2 = float(np.clip(f2, BOUNDS[3, 0], BOUNDS[3, 1]))
    f3 = float(np.clip(f3, BOUNDS[4, 0], BOUNDS[4, 1]))
    if ct not in CELLTYPES:
        ct = "celltype_3"
    return [t, pH, f1, f2, f3, ct]

# What this section does

# This is the defensive layer of the optimiser.

# _safe_y:

# Handles failed simulations / experiments

# Converts NaN or infinite outputs into large penalties

# clip_x:

# Forces all candidate points back into valid bounds

# Fixes invalid categorical values

# Why this exists

# Real BO problems are messy:

# Simulators fail

# Experiments return NaN

# Random sampling overshoots bounds

# Without this:

# GP training crashes

# Acquisition values become meaningless

# Optimisation fails silently

# Oral exam explanation

# “This section ensures robustness. Invalid objective values are penalised so the GP remains stable, and all candidate points are clipped to ensure feasibility. This is essential for real-world Bayesian optimisation.”
def encode_point(x):
    t, pH, f1, f2, f3, ct = x
    cont = np.array([
        (t - 30.0) / 10.0,
        (pH - 6.0) / 2.0,
        f1 / 50.0,
        f2 / 50.0,
        f3 / 50.0
    ], dtype=float)
    cat = np.array([1.0 if ct == c else 0.0 for c in CELLTYPES], dtype=float)
    return np.concatenate([cont, cat])


def encode_batch(X):
    return np.vstack([encode_point(x) for x in X])
# What this section does

# This converts mixed-variable inputs into numeric vectors usable by the Gaussian Process.

# Continuous variables are scaled to [0,1]

# Categorical variable is one-hot encoded

# Final GP input dimension = 8

# Why this exists

# Gaussian Processes:

# operate on Euclidean distances

# are extremely sensitive to variable scaling

# cannot directly handle categorical variables

# This encoding ensures:

# fair contribution from all variables

# meaningful kernel distances

# Oral exam explanation

# “This encoding maps mixed continuous–categorical inputs into a scaled numeric feature space suitable for kernel-based learning. Continuous variables are normalised and the categorical variable is one-hot encoded.”

def sobol_pool_balanced(n_total, sobol_offset, sobol_stride):
    per = n_total // 3
    X = []
    for ct_idx, ct in enumerate(CELLTYPES):
        for j in range(per):
            idx = ct_idx * per + j
            u, _ = i4_sobol(5, sobol_offset + sobol_stride * idx)
            c = BOUNDS[:, 0] + u * (BOUNDS[:, 1] - BOUNDS[:, 0])
            X.append([float(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]), ct])

    rem = n_total - 3 * per
    for r in range(rem):
        u, _ = i4_sobol(5, sobol_offset + sobol_stride * (3 * per + r))
        c = BOUNDS[:, 0] + u * (BOUNDS[:, 1] - BOUNDS[:, 0])
        X.append([float(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]), CELLTYPES[r % 3]])
    return X
# What this does

# Generates a large global candidate pool

# Uses Sobol sequences for uniform coverage

# Explicitly balances candidates across celltypes

# Why Sobol is used

# Lower discrepancy than random sampling

# Better space filling

# Faster global exploration

# Oral exam explanation

# “This function builds a globally diverse candidate pool using Sobol sampling, ensuring uniform coverage of the continuous space and balanced representation of categorical variables.”

def local_cloud_around(x_center, n_pts, radius_frac, keep_ct_prob=0.72):
    t, pH, f1, f2, f3, ct0 = x_center
    span = (BOUNDS[:, 1] - BOUNDS[:, 0])
    r = radius_frac * span
    out = []
    n_pts = int(max(0, n_pts))
    for _ in range(n_pts):
        tt = t + random.uniform(-r[0], r[0])
        pp = pH + random.uniform(-r[1], r[1])
        a1 = f1 + random.uniform(-r[2], r[2])
        a2 = f2 + random.uniform(-r[3], r[3])
        a3 = f3 + random.uniform(-r[4], r[4])
        ct = ct0 if (random.random() < keep_ct_prob) else random.choice(CELLTYPES)
        out.append(clip_x([tt, pp, a1, a2, a3, ct]))
    return out
# What this does

# Generates candidates near the current best solution

# Radius shrinks as optimisation progresses

# Mostly preserves the best celltype

# Why this exists

# Global exploration is inefficient late in BO.
# This implements local refinement similar to trust-region BO.

# Oral exam explanation

# “This function performs local exploitation by sampling in a shrinking neighbourhood around the best observed point, enabling fine-grained optimisation.”

def biased_init_points(n=6):
    """
    Balanced and not severely biased.
    2 Sobol points
    4 mild mixture points that still cover mid and upper feed basins
    Ensures 2 per celltype in the final 6
    """
    n = int(n)
    if n != 6:
        n = 6

    X = []
    base = random.randint(0, 80_000)
    stride = random.choice([1, 3, 5, 7, 9, 11])

    for i in range(2):
        u, _ = i4_sobol(5, base + stride * i)
        c = BOUNDS[:, 0] + u * (BOUNDS[:, 1] - BOUNDS[:, 0])
        X.append([float(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]), "celltype_1"])

    mu1 = np.array([33.6, 6.35, 36.0, 36.0, 34.0], dtype=float)
    sd1 = np.array([1.6, 0.32, 14.0, 14.0, 14.0], dtype=float)

    mu2 = np.array([35.0, 6.55, 28.0, 38.0, 24.0], dtype=float)
    sd2 = np.array([1.8, 0.36, 16.0, 14.0, 16.0], dtype=float)

    p_comp1 = 0.50

    for _ in range(4):
        if random.random() < p_comp1:
            z = np.random.randn(5) * sd1 + mu1
        else:
            z = np.random.randn(5) * sd2 + mu2
        z = np.clip(z, BOUNDS[:, 0], BOUNDS[:, 1])
        X.append([float(z[0]), float(z[1]), float(z[2]), float(z[3]), float(z[4]), "celltype_1"])

    cts = [
        "celltype_1", "celltype_2", "celltype_3",
        "celltype_1", "celltype_2", "celltype_3"
    ]
    for i in range(6):
        X[i][-1] = cts[i]

    return X[:n]
# What this does

# Creates 6 initial evaluation points

# Mixes:

# Sobol global points

# Heuristically promising region

# Forces balanced categorical coverage

# Why this matters

# BO is extremely sensitive to initial data.
# This avoids:

# cold-start failure

# GP bias

# early convergence to poor regions

# Oral exam explanation

# “This initial design balances exploration, prior knowledge, and categorical coverage to give the Gaussian Process a strong starting model.”

class GP:
    def __init__(self):
        self.theta = np.log([0.5, 0.5, 1.0, 1e-4])
        self._bounds = [(-3.0, 2.0), (-3.0, 2.0), (-3.0, 3.0), (-15.0, -1.0)]
        # What this block does

        # This defines a Gaussian Process regression model from scratch.

        # self.theta stores log-hyperparameters

        # The GP has four hyperparameters:

        # lc → lengthscale for continuous variables

        # lk → lengthscale for categorical variables

        # ls → signal variance (function amplitude)

        # ln → noise variance

        # All stored in log-space.

        # Why log-space is used

        # Hyperparameters must be positive

        # Optimisation in log-space:

        # avoids constraints

        # improves numerical stability

        # Why _bounds exist

        # These bounds constrain hyperparameter optimisation:

        # Prevent absurd lengthscales

        # Avoid numerical blow-ups

        # Speed up convergence

        # Oral exam explanation

        # “This class implements a Gaussian Process surrogate. Hyperparameters are stored in log-space to ensure positivity and stable optimisation, and bounded to avoid pathological solutions.”
    def kernel(self, A, B, lc, lk, s2):
        A1, A2 = A[:, :5] / lc, A[:, 5:] / lk
        B1, B2 = B[:, :5] / lc, B[:, 5:] / lk
        sq = np.sum(A1**2, 1)[:, None] + np.sum(B1**2, 1)[None, :] - 2.0 * (A1 @ B1.T)
        sq += np.sum(A2**2, 1)[:, None] + np.sum(B2**2, 1)[None, :] - 2.0 * (A2 @ B2.T)
        sq = np.maximum(sq, 0.0)
        d = np.sqrt(sq)
        sd = np.sqrt(5.0) * d
        return s2 * (1.0 + sd + 5.0 * d * d / 3.0) * np.exp(-sd)
        # What this block does

        # This defines the covariance kernel of the GP — how similar two points are.

        # Key ideas:

        # Inputs are encoded vectors of length 8

        # First 5 dimensions → continuous variables

        # Last 3 dimensions → one-hot categorical variables

        # Separate lengthscales for each part

        # Kernel type

        # This is a Matérn 5/2 kernel:

        # Smooth but not overly smooth

        # Very common in Bayesian optimisation

        # Allows moderate function roughness

        # Why split continuous & categorical parts
        # A[:, :5] / lc
        # A[:, 5:] / lk


        # This allows the GP to learn:

        # “How sensitive is the objective to small continuous changes?”

        # “How important is switching celltype?”

        # Oral exam explanation

        # “The kernel defines similarity between points using a Matérn 5/2 covariance. Continuous and categorical variables are treated separately with different lengthscales, allowing the GP to learn how influential categorical changes are.”
    def _build_K(self, X, y_std):
        lc, lk, ls, ln = self.theta
        K = self.kernel(X, X, np.exp(lc), np.exp(lk), np.exp(2.0 * ls))
        jitter = 1e-8 + 1e-6 * float(np.var(y_std))
        K[np.diag_indices(len(X))] += np.exp(2.0 * ln) + jitter
        return K
        # What this block does

        # This constructs the training covariance matrix:

        #𝐾=𝐾(𝑋,𝑋)+𝜎𝑛^2𝐼

        # Where:

        # K(X,X) comes from the kernel

        # Noise term + jitter are added to the diagonal

        # Why jitter is added

        # Ensures positive definiteness

        # Prevents Cholesky decomposition failure

        # Scales slightly with data variance

        # Oral exam explanation

        # “This function builds the GP covariance matrix and adds noise and jitter for numerical stability, ensuring reliable Cholesky decomposition.”
    def neg_log_mll(self, theta):
        old = self.theta
        self.theta = theta
        try:
            K = self._build_K(self.X, self.y)
            L, _ = cho_factor(K, lower=True, check_finite=False)
            alpha = cho_solve((L, True), self.y, check_finite=False)
            nll = 0.5 * float(self.y @ alpha) + float(np.sum(np.log(np.diag(L)))) + 0.5 * len(self.y) * np.log(2.0 * np.pi)
        except LinAlgError:
            nll = 1e25
        self.theta = old
        return float(nll)
        # What this block does

        # This computes the negative log marginal likelihood (NLL) of the GP.

        # This is:

        # the objective used to train GP hyperparameters

        # how the GP learns smoothness, noise, and scale

        # Why Cholesky is used

        # Efficient: 

        # Stable

        # Standard in GP implementations

        # Why exceptions return large value

        # If covariance matrix fails:

        # that hyperparameter setting is invalid

        # optimisation should avoid it

        # Oral exam explanation

        # “Hyperparameters are trained by minimising the negative log marginal likelihood, which balances data fit and model complexity.”
    def fit(self, X, y, optimise=True, n_restarts=2, maxiter=140):
        self.X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        self.m = float(np.mean(y))
        self.s = float(np.std(y)) if float(np.std(y)) > 1e-9 else 1.0
        self.y = (y - self.m) / self.s

        if optimise:
            best_theta = self.theta.copy()
            best_val = self.neg_log_mll(best_theta)

            for r in range(max(1, int(n_restarts))):
                if r == 0:
                    x0 = self.theta.copy()
                else:
                    x0 = self.theta.copy() + np.random.randn(4) * 0.16
                    for k in range(4):
                        x0[k] = np.clip(x0[k], self._bounds[k][0], self._bounds[k][1])

                res = minimize(
                    self.neg_log_mll,
                    x0,
                    method="L-BFGS-B",
                    bounds=self._bounds,
                    options={"maxiter": int(maxiter)}
                )
                if float(res.fun) < best_val:
                    best_val = float(res.fun)
                    best_theta = res.x.copy()

            self.theta = best_theta

        K = self._build_K(self.X, self.y)
        self.L, _ = cho_factor(K, lower=True, check_finite=False)
        self.alpha = cho_solve((self.L, True), self.y, check_finite=False)
        # What this block does

        # This:

        # Stores training data

        # Normalises outputs

        # Optionally optimises hyperparameters

        # Precomputes matrices for prediction

        # Why normalise y

        # GP hyperparameters become easier to learn

        # Prevents scale dominance

        # Improves numerical stability

        # Why multiple restarts

        # NLL is non-convex

        # Restarts reduce local minima risk

        # Oral exam explanation

        # “The GP is trained by normalising outputs, optimising hyperparameters via marginal likelihood, and precomputing matrix factorizations for fast prediction.”
    def predict(self, Xs):
        Xs = np.asarray(Xs, dtype=float)
        lc, lk, ls, _ = self.theta
        Kxs = self.kernel(Xs, self.X, np.exp(lc), np.exp(lk), np.exp(2.0 * ls))
        mu = Kxs @ self.alpha
        v = cho_solve((self.L, True), Kxs.T, check_finite=False)
        var = np.exp(2.0 * ls) - np.sum(Kxs * v.T, axis=1)
        mu_out = mu * self.s + self.m
        std_out = np.sqrt(np.maximum(var, 1e-12)) * self.s
        return mu_out, std_out
        # What this block does

        # Given new points:

        # Computes predictive mean

        # Computes predictive uncertainty

        # Unnormalises outputs

        # This is exactly what BO needs to:

        # rank candidates

        # balance exploration vs exploitation

        # Oral exam explanation

        # “The GP prediction provides both mean and uncertainty estimates, which are essential for acquisition functions like Expected Improvement.”

def expected_improvement(mu, sigma, best, xi):
    sigma = np.maximum(sigma, 1e-12)
    z = (mu - best - xi) / (sigma + 1e-12)
    return (mu - best - xi) * norm.cdf(z) + sigma * norm.pdf(z)
# What this does

# Computes Expected Improvement (EI)

# Balances exploitation (high mean) and exploration (high uncertainty)

# Why EI is used

# EI:

# is principled

# has closed form

# works well with GPs

# Oral exam explanation

# “Expected Improvement selects points that are likely to outperform the current best while accounting for predictive uncertainty.”

class BO_Run:
    def __init__(self,
                 global_pool_size=12000,
                 shortlist_M=1200,
                 div_weight=0.22,
                 target_time_s=32.0):

        self.global_pool_size = int(global_pool_size)
        self.shortlist_M = int(shortlist_M)
        self.div_weight = float(div_weight)
        self.target_time_s = float(target_time_s)

        self.N_MIN_OPT = 15
        self.UCB_KAPPA = 1.0
        self.UCB_PICKS = 1
        self.score_from_batch = 3

        self.gp = GP()

        self.X_obs = []
        self.Y_obs = []
        self.batch_best = []
        self.best_overall = -1e18

        self.global_pool_X = None
        self.global_pool_enc = None

        self.bad_keys = set()
        # What this section does

        # This sets up everything the optimiser needs before optimisation begins.

        # Key responsibilities:

        # Stores optimisation budget:

        # iterations = number of BO rounds

        # batch = number of points per round

        # Stores history:

        # X_obs, Y_obs = evaluated data

        # Sets strategy hyperparameters:

        # exploration vs exploitation

        # diversity

        # rescue logic

        # Instantiates the Gaussian Process

        # Important exam notes

        # X_searchspace is not actually used later → legacy/template compatibility

        # bad_keys tracks failed or invalid evaluations

        # global_pool_* will hold Sobol candidates

        # Oral explanation

        # “The BO class manages the entire optimisation process. It stores observations, controls budgets, sets exploration–exploitation strategy, and orchestrates interaction between the GP surrogate and the true objective.”
    def _pt_key(self, x, nd=6):
        return (
            round(float(x[0]), nd),
            round(float(x[1]), nd),
            round(float(x[2]), nd),
            round(float(x[3]), nd),
            round(float(x[4]), nd),
            str(x[5])
        )
            # What this section does

            # Creates a hashable key for a candidate point.

            # Continuous values are rounded to nd decimal places

            # Category is converted to string

            # Returned as a tuple → usable in sets

            # Why rounding is used

            # Floating-point values are not reliably comparable

            # Two mathematically identical values may differ numerically

            # Rounding gives robust equality checks

            # Why this matters

            # Used to:

            # avoid re-evaluating the same point

            # track failed experiments (bad_keys)

            # Oral exam explanation

            # “This function provides a robust way to identify duplicate points using rounding, preventing redundant or failed evaluations.”
    def _tr_radius_for_best(self, best_overall):
        if best_overall >= 335.0:
            return 0.03
        if best_overall >= 305.0:
            return 0.05
        return 0.08
        # What this section does

        # Defines trust-region radius as a function of performance.

        # Poor performance → large radius (exploration)

        # Good performance → small radius (fine-tuning)

        # Why thresholds are hard-coded

        # This is problem-specific tuning:

        # Based on expected objective scale

        # Encodes empirical knowledge

        # Oral exam explanation

        # “This function adapts the local search radius based on optimisation progress, enabling coarse exploration early and precise refinement later.”
    def _local_cloud_n_for_best(self, best_overall):
        if best_overall >= 335.0:
            return 2200
        if best_overall >= 305.0:
            return 1500
        if best_overall >= 285.0:
            return 900
        return 500
        # What this section does

        # Controls how many local candidates are generated.

        # Better performance → more local samples

        # Poor performance → fewer local samples

        # Why this matters

        # Near optimum → dense local search

        # Early stage → avoid overspending locally

        # Oral exam explanation

        # “The number of local candidates increases as the optimiser gains confidence, focusing computational effort where it matters most.”
    def _apply_mean_gated_uncertainty(self, score, mu, gate_q):
        cut = float(np.quantile(mu, gate_q))
        out = score.copy()
        out[mu < cut] = -np.inf
        return out
        # What this section does

        # This modifies uncertainty-based exploration by:

        # rejecting candidates with very low predicted mean

        # How it works

        # Computes a quantile of predicted means (gate_q)

        # Any point with mean below this threshold:

        # gets score -∞

        # cannot be selected

        # Why this exists

        # Pure uncertainty sampling:

        # can explore obviously bad regions

        # wastes evaluations

        # This is a smart hybrid:

        # explore uncertain points

        # but only if they’re not predicted to be terrible

        # Oral exam explanation

        # “Mean-gated uncertainty avoids wasting exploration on regions the GP already predicts to be poor, improving sampling efficiency.”
    def _ensure_global_pool(self):
        if self.global_pool_X is not None:
            return
        base = random.randint(0, 450_000)
        stride = random.choice([1, 3, 5, 7, 9])
        self.global_pool_X = sobol_pool_balanced(self.global_pool_size, sobol_offset=base, sobol_stride=stride)
        self.global_pool_enc = encode_batch(self.global_pool_X)
            # What this section does

            # Creates the global Sobol candidate pool, but only once.

            # Important details

            # Uses random:

            # Sobol offset

            # Sobol stride

            # Ensures:

            # different runs explore different patterns

            # Encodes pool once for efficiency

            # Why lazy initialisation

            # Sobol pool is large (12,000 points)

            # Expensive to regenerate

            # No need to rebuild it every iteration

            # Oral exam explanation

            # “This function initialises a large global Sobol candidate pool once and caches both raw and encoded versions for efficient reuse.”
    def _global_pool_take(self, batch_idx):
        if batch_idx <= 5:
            return min(self.global_pool_size, 12000)
        if batch_idx <= 10:
            return min(self.global_pool_size, 8500)
        return min(self.global_pool_size, 6500)
            # What this section does

            # Controls how much global exploration is allowed per iteration.

            # Early iterations → large global pool

            # Later iterations → smaller global pool

            # Why this exists

            # Early BO needs exploration

            # Late BO should focus locally

            # Reduces computational cost over time

            # Oral exam explanation

            # “This function gradually reduces global exploration as optimisation progresses, shifting focus toward exploitation.”
    def _ml2_plan(self, batch_idx, elapsed_s):
        if elapsed_s < 0.70 * self.target_time_s:
            return True, 3, 170
        if elapsed_s < 0.88 * self.target_time_s:
            return True, 2, 140
        return True, 1, 110
        # What this section does

        # Schedules how aggressively GP hyperparameters are optimised.

        # Returns:

        # whether to optimise

        # number of restarts

        # max iterations

        # Why time-aware

        # GP optimisation is expensive.
        # This prevents:

        # timeouts

        # wasted compute near deadline

        # Oral exam explanation

        # “Hyperparameter optimisation is adaptively scheduled based on remaining time to balance model quality and computational budget.”
    def _rescue_pool(self, n_pts):
        muR = np.array([33.4, 6.30, 46.0, 46.0, 40.0], dtype=float)
        sdR = np.array([1.0, 0.20, 7.5, 7.5, 9.5], dtype=float)
        X = []
        for i in range(int(n_pts)):
            z = np.random.randn(5) * sdR + muR
            z = np.clip(z, BOUNDS[:, 0], BOUNDS[:, 1])
            ct = "celltype_3" if random.random() < 0.75 else random.choice(["celltype_1", "celltype_2"])
            X.append([float(z[0]), float(z[1]), float(z[2]), float(z[3]), float(z[4]), ct])
        return X
            # What this section does

            # Creates emergency exploration points.

            # Samples from a Gaussian distribution

            # Centred at a known promising region

            # Biased heavily toward celltype_3

            # Why this exists

            # If BO:

            # gets stuck early

            # makes poor predictions

            # converges to a bad region

            # This injects domain-informed randomness to escape.

            # Oral exam explanation

            # “The rescue pool injects aggressive exploration around a known promising region when the optimiser detects stagnation.”
    def run(self, iters=15, batch_size=5, verbose=True):
        X_init = biased_init_points(n=6)
        Y_init = _safe_y(objective_func(X_init)).tolist()

        for i, y in enumerate(Y_init):
            if (not np.isfinite(y)) or (y < -1e5):
                self.bad_keys.add(self._pt_key(X_init[i]))

        self.X_obs = list(X_init)
        self.Y_obs = list(Y_init)
        self.best_overall = float(np.max(np.array(self.Y_obs, dtype=float)))

        self.gp.fit(encode_batch(self.X_obs), np.array(self.Y_obs, dtype=float), optimise=False)
        self._ensure_global_pool()

        mark_score = 0.0
        t_start = time.time()
        rescue_mode = False
            # This part of the code initialises and prepares the Bayesian Optimisation process before the main optimisation loop begins.

            # At a high level, it:

            # Generates a small set of initial experiments using a biased sampling strategy.
            # These points give the optimiser a reasonable starting understanding of the objective function instead of starting completely blind.

            # Evaluates the objective function safely at those initial points.
            # Any invalid, NaN, or failed evaluations are penalised and recorded so the optimiser does not try them again.

            # Initialises the dataset of observed inputs and outputs that will be used to train the Gaussian Process surrogate model.

            # Fits an initial Gaussian Process model using the initial data, without optimising hyperparameters yet.
            # This provides a rough probabilistic model of the objective function that can already make predictions and uncertainty estimates.

            # Creates a global candidate pool using Sobol sampling.
            # This pool is later used for global exploration when selecting new points to evaluate.

            # Resets optimisation state variables, including timing, scoring, and rescue-mode flags, so the optimisation starts from a clean and controlled state.

            # 🎤 One-minute oral exam answer (polished)

            # “This code block bootstraps the Bayesian Optimisation process. It generates initial design points, evaluates the objective safely, and uses those observations to train an initial Gaussian Process surrogate. It also prepares the global Sobol candidate pool for exploration and resets all runtime and control variables. Once this setup is complete, the optimiser is ready to enter the main optimisation loop.”
            #         for b in range(1, iters + 1):
            elapsed = time.time() - t_start
            explore_mode = (b <= 2)

            if b <= 2:
                xi = 0.30
            elif b <= 5:
                xi = 0.06
            else:
                xi = 0.02
            # What this does

            # Each loop iteration is one BO batch.

            # Early iterations:

            # heavy exploration

            # Later iterations:

            # exploitation-focused

            # Why xi is scheduled

            # Large xi → exploration

            # Small xi → exploitation

            # Prevents early overconfidence

            # Oral explanation

            # “The optimisation follows a scheduled exploration–exploitation strategy, encouraging uncertainty-driven exploration early and improvement-focused sampling later.”
            rescue_active = rescue_mode and (b in [4, 5])

            best_idx = int(np.argmax(np.array(self.Y_obs, dtype=float)))
            x_best = self.X_obs[best_idx]

            tr_r = self._tr_radius_for_best(self.best_overall)
            n_loc = self._local_cloud_n_for_best(self.best_overall)
            
            takeN = self._global_pool_take(b)
            if elapsed > 0.90 * self.target_time_s:
                takeN = int(max(2500, 0.65 * takeN))
            if elapsed > 0.90 * self.target_time_s:
                n_loc = int(max(250, 0.55 * n_loc))
            # What this does

            # Identifies the current best observed solution

            # Determines:

            # trust-region radius

            # number of local samples

            # size of global candidate pool

            # Why adaptive

            # As performance improves:

            # radius shrinks

            # local exploitation increases

            # global exploration reduces

            # Oral explanation

            # “The optimiser dynamically adapts its sampling strategy based on the best observed performance, focusing increasingly on local refinement.”
            X_local = local_cloud_around(x_best, n_loc, tr_r, keep_ct_prob=0.72)
            enc_local = encode_batch(X_local) if len(X_local) > 0 else None


            X_cand = self.global_pool_X[:takeN]
            enc_cand = self.global_pool_enc[:takeN]

            X_pool = []
            enc_parts = []
            
            if len(X_local) > 0:
                X_pool.extend(X_local)
                enc_parts.append(enc_local)

            X_pool.extend(X_cand)
            enc_parts.append(enc_cand)

            enc_pool = np.vstack(enc_parts)
            # What this does

            # Builds the candidate set by combining:

            # Local exploitation points

            # Global Sobol exploration points

            # All candidates are encoded once for efficiency.

            # Oral explanation

            # “Each iteration constructs a hybrid candidate pool combining local trust-region samples and global Sobol samples to balance exploitation and exploration.”
            if rescue_active:
                n_res = int(max(1200, 1.2 * takeN))
                if elapsed > 0.90 * self.target_time_s:
                    n_res = int(max(900, 0.65 * n_res))
                X_res = self._rescue_pool(n_res)
                enc_res = encode_batch(X_res)
                X_pool.extend(X_res)
                enc_parts.append(enc_res)

            obs_keys = set([self._pt_key(x) for x in self.X_obs])
            taken0 = np.zeros(len(X_pool), dtype=bool)
            for i, x in enumerate(X_pool):
                k = self._pt_key(x)
                if (k in obs_keys) or (k in self.bad_keys):
                    taken0[i] = True
                # What this does

                # Prevents re-evaluating:

                # already-tested points

                # previously failed points

                # Why rounding keys

                # Floating-point exact equality is unreliable.

                # Oral explanation

                # “This step ensures efficiency by avoiding redundant or known-bad evaluations.”
            mu0, std0 = self.gp.predict(enc_pool)
            best0 = float(np.max(np.array(self.Y_obs, dtype=float)))

            if explore_mode or rescue_active:
                score0 = std0.copy()
                gate = 0.55 if explore_mode else 0.72
                score0 = self._apply_mean_gated_uncertainty(score0, mu0, gate_q=gate)
                score0 = score0 + 0.01 * (mu0 - float(np.mean(mu0)))
            else:
                mu_eff0 = mu0
                if b == 3:
                    mu_eff0 = mu0 + self.UCB_KAPPA * std0
                score0 = expected_improvement(mu_eff0, std0, best0, xi)
            # What this does

            # Predicts mean and uncertainty

            # Uses:

            # uncertainty-only scoring early

            # Expected Improvement later

            # Why gate uncertainty

            # Avoids exploring regions predicted to be very poor.

            # Oral explanation

            # “The acquisition strategy shifts from uncertainty-driven exploration to Expected Improvement as confidence in the surrogate increases.”
            score0[taken0] = -np.inf

            poolN = len(X_pool)
            M = int(min(self.shortlist_M, poolN))
            finite = np.isfinite(score0)
            if int(np.sum(finite)) > M:
                idx_short = np.argpartition(score0, -M)[-M:]
            else:
                idx_short = np.where(finite)[0]
                if idx_short.size == 0:
                    idx_short = np.argpartition(std0, -M)[-M:]

            X_short = [X_pool[i] for i in idx_short.tolist()]
            enc_short = enc_pool[idx_short]
                # What this does

                # Keeps only top-scoring candidates

                # Reduces computational cost for batch selection

                # Oral explanation

                # “Shortlisting reduces acquisition optimisation cost by focusing only on the most promising candidates.”
            taken_s = np.zeros(len(X_short), dtype=bool)

            fake_X, fake_Y = [], []

            for j in range(batch_size):
                mu, std = self.gp.predict(enc_short)
                best = float(np.max(np.array(self.Y_obs + fake_Y, dtype=float)))

                if explore_mode or rescue_active:
                    score = std.copy()
                    gate = 0.55 if explore_mode else 0.72
                    score = self._apply_mean_gated_uncertainty(score, mu, gate_q=gate)
                    score = score + 0.01 * (mu - float(np.mean(mu)))
                else:
                    mu_eff = mu
                    if (b == 3) and (j < self.UCB_PICKS):
                        mu_eff = mu + self.UCB_KAPPA * std
                    score = expected_improvement(mu_eff, std, best, xi)

                if len(fake_X) > 0:
                    enc_fake = encode_batch(fake_X)
                    diff = enc_short[:, None, :] - enc_fake[None, :, :]
                    dist = np.linalg.norm(diff, axis=2).min(axis=1)
                    score = score * (1.0 + self.div_weight * dist)

                if b == 1 and j < 3:
                    chosen_cts = set([x[-1] for x in fake_X])
                    missing = set(CELLTYPES) - chosen_cts
                    if len(missing) > 0:
                        allowed = np.array([X_short[i][-1] in missing for i in range(len(X_short))], dtype=bool)
                        score[~allowed] = -np.inf

                score[taken_s] = -np.inf

                idxj = int(np.argmax(score))
                taken_s[idxj] = True
                x_new = X_short[idxj]
                fake_X.append(x_new)

                mu_fake, _ = self.gp.predict(encode_batch([x_new]))
                fake_Y.append(float(mu_fake[0]))

                self.gp.fit(
                    encode_batch(self.X_obs + fake_X),
                    np.array(self.Y_obs + fake_Y, dtype=float),
                    optimise=False
                )
                # What this does

                # This is batch BO done properly.

                # Selects batch points sequentially

                # Uses fantasy observations (predicted means)

                # Updates GP between selections

                # Why this is essential

                # Without this:

                # Batch points collapse to same location

                # Waste evaluation budget

                # Oral explanation

                # “Batch selection uses fantasy models, also known as Kriging Believer, to simulate the effect of evaluating points sequentially and maintain batch diversity.”
            Y_real = _safe_y(objective_func(fake_X)).tolist()

            for i, y in enumerate(Y_real):
                if (not np.isfinite(y)) or (y < -1e5):
                    self.bad_keys.add(self._pt_key(fake_X[i]))

            self.batch_best.append(float(np.max(np.array(Y_real, dtype=float))))
            self.X_obs.extend(fake_X)
            self.Y_obs.extend(Y_real)
            self.best_overall = float(np.max(np.array(self.Y_obs, dtype=float)))
            # What this does

            # Evaluates real objective

            # Updates observation history

            # Updates global best

            # Flags failed points

            # Oral explanation

            # “The selected batch is evaluated against the true objective, and results are incorporated into the optimisation history.”
            if b == 3 and self.best_overall < 300.0:
                rescue_mode = True
            if self.best_overall >= 305.0:
                rescue_mode = False

            if b >= self.score_from_batch:
                mark_score += self.best_overall

            elapsed = time.time() - t_start
            optimise_now, restarts, maxiter = self._ml2_plan(b, elapsed)

            if optimise_now and (len(self.Y_obs) >= self.N_MIN_OPT):
                self.gp.fit(
                    encode_batch(self.X_obs),
                    np.array(self.Y_obs, dtype=float),
                    optimise=True,
                    n_restarts=restarts,
                    maxiter=maxiter
                )
            else:
                self.gp.fit(
                    encode_batch(self.X_obs),
                    np.array(self.Y_obs, dtype=float),
                    optimise=False
                )
                # What this does

                # Detects early stagnation

                # Triggers aggressive exploration

                # Schedules GP hyperparameter optimisation

                # Oral explanation

                # “Rescue mode prevents premature convergence, while adaptive hyperparameter optimisation balances model accuracy and computational cost.”
            if verbose:
                s = f"  Batch {b:02d}/15 | batch_best={self.batch_best[-1]:.3f} | best_overall={self.best_overall:.3f} | mark_so_far={mark_score:.1f}"
                if rescue_active:
                    s += " | RESCUE"
                print(s)

        return {
            "best": float(self.best_overall),
            "mark_score": float(mark_score),
            "time_s": float(time.time() - t_start),
        }


def main():
    N_RUNS = 15
    bests, marks, times = [], [], []

    for r in range(1, N_RUNS + 1):
        print(f"\n=== RUN {r}/{N_RUNS} ===")
        bo = BO_Run(
            global_pool_size=12000,
            shortlist_M=1200,
            div_weight=0.22,
            target_time_s=32.0
        )
        out = bo.run(iters=15, batch_size=5, verbose=True)
        bests.append(out["best"])
        marks.append(out["mark_score"])
        times.append(out["time_s"])
        print(f"RUN {r}: best={out['best']:.3f} | mark_score={out['mark_score']:.1f} | time={out['time_s']:.1f}s")

    bests = np.array(bests, dtype=float)
    marks = np.array(marks, dtype=float)
    times = np.array(times, dtype=float)

    print(f"\n=== SUMMARY ({N_RUNS} independent runs) ===")
    print(f"avg best:      {bests.mean():.3f} (std {bests.std(ddof=1):.3f})")
    print(f"avg markscore: {marks.mean():.1f} (std {marks.std(ddof=1):.1f})")
    print(f"avg time/run:  {times.mean():.1f}s")
# 🔍 High-level overview of main()

# This function acts as a driver and evaluator for the Bayesian Optimisation algorithm.
# It repeatedly runs the optimiser under the same settings to measure average performance, stability, and runtime, rather than relying on a single run.
# By aggregating results across multiple independent executions, it provides a statistically meaningful assessment of how well the optimisation strategy performs overall.

# 🎤 Exam-ready version (say this)

# “This function runs the Bayesian Optimisation algorithm multiple times to evaluate its average performance, robustness, and computational cost, and then reports summary statistics across all runs.”

if __name__ == "__main__":
    main()
