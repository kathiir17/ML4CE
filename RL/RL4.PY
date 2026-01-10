# algorithms/your_algorithm.py
# Only numpy + torch + stdlib (time/os) as required.

import os
import time
import numpy as np
import torch


# ============================================================
# Helpers: vectorize / devectorize policy parameters
# ============================================================
def _get_param_shapes(policy_net):
    names, shapes = [], []
    for n, p in policy_net.named_parameters():
        names.append(n)
        shapes.append(tuple(p.data.shape))
    return names, shapes


def _state_dict_to_vec(state_dict, names):
    parts = []
    for n in names:
        parts.append(state_dict[n].detach().cpu().numpy().ravel())
    return np.concatenate(parts, axis=0).astype(np.float32)


def _vec_to_state_dict(vec, template_state_dict, names, shapes):
    out = {}
    idx = 0
    for n, shp in zip(names, shapes):
        size = int(np.prod(shp))
        chunk = vec[idx: idx + size].reshape(shp)
        ref = template_state_dict[n]
        out[n] = torch.from_numpy(chunk).to(dtype=ref.dtype, device=ref.device)
        idx += size
    return out


def _clip_and_round_action(action_cont, action_ub):
    # action_cont: (n_actions,) float
    a = np.rint(action_cont).astype(np.int16)
    a = np.clip(a, 0, action_ub.astype(np.int16))
    return a


# ============================================================
# Heuristic (s,S) policy
# ============================================================
def _ss_action_from_state(state, action_ub, n_entities, ss_params):
    """
    ss_params = [s_0, S_0, s_1, S_1, ...] for each entity.
    state = [invpos_entity_0, ..., invpos_entity_{n_entities-1}, dow]
    """
    inv = state[:n_entities].astype(np.float32)
    a = np.zeros(n_entities, dtype=np.float32)

    for i in range(n_entities):
        s_i = float(ss_params[2 * i + 0])
        S_i = float(ss_params[2 * i + 1])
        if inv[i] <= s_i:
            a[i] = max(0.0, S_i - inv[i])
        else:
            a[i] = 0.0

    return _clip_and_round_action(a, action_ub)


# ============================================================
# Environment rollouts / evaluation
# ============================================================
def _run_episode(env, policy_net=None, action_ub=None, ss_params=None, rng=None):
    """
    Run one full episode and return the total reward.
    Either:
      - ss_params is provided => use (s,S) heuristic
      - otherwise use policy_net forward pass
    """
    env.reset()
    done = False
    ep_ret = 0.0

    n_actions = env.action_space.shape[0]
    n_entities = n_actions  # retailers + DCs

    while not done:
        s = env.state  # IMPORTANT: raw state, matches evaluation pipeline

        if ss_params is not None:
            a = _ss_action_from_state(s, action_ub, n_entities, ss_params)
        else:
            with torch.no_grad():
                x = torch.tensor(s, dtype=torch.float32)
                y = policy_net(x).detach().cpu().numpy()
            # Safety clip (prevents huge actions -> huge loops in env internals)
            a = _clip_and_round_action(y, action_ub)

        _, r, done, _ = env.step(a)
        ep_ret += float(r)

    return ep_ret


def _eval_mean_return(env, n_eval_episodes, policy_net=None, action_ub=None, ss_params=None, rng=None):
    rets = np.zeros(n_eval_episodes, dtype=np.float32)
    for i in range(n_eval_episodes):
        rets[i] = float(_run_episode(env, policy_net=policy_net, action_ub=action_ub, ss_params=ss_params, rng=rng))
    return float(rets.mean()), float(rets.std(ddof=0))


# ============================================================
# CEM optimiser (generic)
# ============================================================
def _cem_optimize(
    fitness_fn,
    dim,
    x_init,
    sigma_init,
    pop_size=32,
    elite_frac=0.2,
    n_iters=40,
    x_min=None,
    x_max=None,
    seed=0,
    sigma_decay=0.95,
    min_sigma=1e-3,
    time_budget_s=None,
    start_time=None,
):
    rng = np.random.default_rng(seed)
    mu = x_init.astype(np.float32).copy()
    sigma = float(sigma_init)

    n_elite = max(2, int(pop_size * elite_frac))
    best_x = mu.copy()
    best_f = -np.inf

    for _ in range(n_iters):
        if time_budget_s is not None and start_time is not None:
            if (time.time() - start_time) >= time_budget_s:
                break

        eps = rng.standard_normal(size=(pop_size, dim)).astype(np.float32)
        xs = mu[None, :] + sigma * eps

        if x_min is not None:
            xs = np.maximum(xs, x_min[None, :])
        if x_max is not None:
            xs = np.minimum(xs, x_max[None, :])

        fs = np.zeros(pop_size, dtype=np.float32)
        for i in range(pop_size):
            fs[i] = float(fitness_fn(xs[i]))

        j = int(np.argmax(fs))
        if fs[j] > best_f:
            best_f = float(fs[j])
            best_x = xs[j].copy()

        elite_idx = np.argsort(fs)[-n_elite:]
        mu = xs[elite_idx].mean(axis=0)

        sigma = max(min_sigma, sigma * sigma_decay)

    return best_x, best_f


# ============================================================
# Behaviour cloning (imitate best heuristic into NN)
# ============================================================
def _collect_bc_data(env, action_ub, ss_params, n_episodes=40):
    X, Y = [], []
    n_entities = env.action_space.shape[0]

    for _ in range(n_episodes):
        env.reset()
        done = False
        while not done:
            s = env.state
            a = _ss_action_from_state(s, action_ub, n_entities, ss_params)
            X.append(s.astype(np.float32))
            Y.append(a.astype(np.float32))  # IMPORTANT: true action units
            _, _, done, _ = env.step(a)

    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    return X, Y


def _train_bc(
    policy_net,
    X,
    Y,
    action_ub,
    lr=3e-3,
    epochs=25,
    batch_size=256,
    weight_decay=1e-4,
    seed=0,
):
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(policy_net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()

    ub = torch.tensor(action_ub.astype(np.float32), dtype=torch.float32).view(1, -1)

    n = X.shape[0]
    for _ in range(epochs):
        idx = rng.permutation(n)
        for i in range(0, n, batch_size):
            b = idx[i: i + batch_size]
            xb = torch.tensor(X[b], dtype=torch.float32)
            yb = torch.tensor(Y[b], dtype=torch.float32)

            pred = policy_net(xb)

            # MSE imitation loss
            loss = loss_fn(pred, yb)

            # Soft constraints to keep outputs inside [0, action_ub]
            loss = loss + 0.01 * torch.mean(torch.relu(pred - ub) ** 2)
            loss = loss + 0.01 * torch.mean(torch.relu(-pred) ** 2)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 5.0)
            opt.step()


# ============================================================
# Main entry point (MANDATORY signature)
# ============================================================
def your_optimization_alg(env, policy_net, *args, **hyperparam):
    """
    Mandatory signature.

    Returns
    -------
    policy_param : dict
        State dict of the best policy network found.
    plot_info : dict
        Dict with keys used by utils.plot_reward_evolution:
        - episodes
        - reward_history
        - std_history
        - best_reward_history
        - save_path
    """
    # ------------------------------
    # Reproducibility (IMPORTANT)
    # ------------------------------
    seed = int(hyperparam.get("seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(env, "seed"):
        try:
            env.seed = seed
        except Exception:
            pass
    rng = np.random.default_rng(seed)

    # ------------------------------
    # Budgets (grading constraints)
    # ------------------------------
    max_episodes = int(hyperparam.get("max_episodes", 5000))
    max_time = float(hyperparam.get("max_time", 60 * 5))
    start_time = time.time()

    # "episode" budget here counts full environment episodes simulated
    episodes_used = 0

    # ------------------------------
    # Spaces
    # ------------------------------
    action_ub = env.action_space.high.astype(np.int16)
    obs_ub = env.observation_space.high.astype(np.int16)  # not used directly (kept for clarity)

    n_actions = env.action_space.shape[0]
    n_entities = n_actions
    ss_dim = 2 * n_entities

    # ------------------------------
    # Plot buffers (like SA / REINFORCE)
    # ------------------------------
    episodes_curve = []
    reward_history = []
    std_history = []
    best_reward_history = []

    best_policy_sd = {k: v.detach().clone() for k, v in policy_net.state_dict().items()}
    best_return = -np.inf

    # ------------------------------
    # Utility: checkpoint evaluation
    # ------------------------------
    def checkpoint_eval(ss_params=None):
        nonlocal episodes_used, best_return, best_policy_sd

        n_eval = int(hyperparam.get("num_episodes_avg", 3))
        if episodes_used + n_eval > max_episodes:
            n_eval = max(1, max_episodes - episodes_used)

        if ss_params is not None:
            mean_r, std_r = _eval_mean_return(env, n_eval, action_ub=action_ub, ss_params=ss_params, rng=rng)
        else:
            mean_r, std_r = _eval_mean_return(env, n_eval, policy_net=policy_net, action_ub=action_ub, rng=rng)

        episodes_used += n_eval

        episodes_curve.append(int(episodes_used))
        reward_history.append(float(mean_r))
        std_history.append(float(std_r))

        if mean_r > best_return:
            best_return = float(mean_r)
            best_policy_sd = {k: v.detach().clone() for k, v in policy_net.state_dict().items()}
        best_reward_history.append(float(best_return))

    # ============================================================
    # (1) CEM tune a strong (s,S) heuristic
    # ============================================================
    # Initial guess (generic)
    x0 = np.zeros(ss_dim, dtype=np.float32)
    for i in range(n_entities):
        # simple generic guesses: reorder point 1/4 of cap, target 3/4 of cap
        cap_i = float(obs_ub[i])
        x0[2 * i + 0] = 0.25 * cap_i
        x0[2 * i + 1] = 0.75 * cap_i

    # Bounds for s, S (and we enforce S >= s in the fitness function)
    cap_plus = obs_ub[:-1].astype(np.float32)  # per-entity inv_pos upper bounds
    x_min = np.zeros(ss_dim, dtype=np.float32)
    x_max = np.zeros(ss_dim, dtype=np.float32)
    for i in range(n_entities):
        hi = float(cap_plus[i])
        x_min[2 * i + 0] = 0.0
        x_max[2 * i + 0] = hi
        x_min[2 * i + 1] = 0.0
        x_max[2 * i + 1] = hi

    ss_pop = int(hyperparam.get("ss_pop", 24))
    ss_elite = float(hyperparam.get("ss_elite_frac", 0.25))
    ss_iters = int(hyperparam.get("ss_cem_iters", 30))
    ss_eval_eps = int(hyperparam.get("ss_eval_eps", 2))

    def ss_fitness(x):
        nonlocal episodes_used

        if (time.time() - start_time) >= max_time:
            return -1e18
        if episodes_used + ss_eval_eps > max_episodes:
            return -1e18

        # enforce S >= s
        xx = x.copy()
        for i in range(n_entities):
            s_i = xx[2 * i + 0]
            xx[2 * i + 1] = max(xx[2 * i + 1], s_i)

        mean_r, _ = _eval_mean_return(env, ss_eval_eps, action_ub=action_ub, ss_params=xx, rng=rng)
        episodes_used += ss_eval_eps
        return float(mean_r)

    # Allocate only a fraction of time to heuristic stage
    ss_time_budget = max_time * float(hyperparam.get("ss_time_frac", 0.35))
    ss_best, _ = _cem_optimize(
        ss_fitness,
        dim=ss_dim,
        x_init=x0,
        sigma_init=float(hyperparam.get("ss_sigma_init", 15.0)),
        pop_size=ss_pop,
        elite_frac=ss_elite,
        n_iters=ss_iters,
        x_min=x_min,
        x_max=x_max,
        seed=seed,
        sigma_decay=float(hyperparam.get("ss_sigma_decay", 0.93)),
        min_sigma=float(hyperparam.get("ss_min_sigma", 1.0)),
        time_budget_s=ss_time_budget,
        start_time=start_time,
    )

    # evaluate tuned heuristic (checkpoint)
    checkpoint_eval(ss_params=ss_best)

    # ============================================================
    # (2) Behaviour cloning: imitate tuned heuristic into NN
    # ============================================================
    bc_episodes = int(hyperparam.get("bc_episodes", 40))
    # fit to remaining episode budget
    bc_episodes = int(min(bc_episodes, max(0, max_episodes - episodes_used - 1)))
    if bc_episodes > 0 and (time.time() - start_time) < max_time:
        X, Y = _collect_bc_data(env, action_ub, ss_best, n_episodes=bc_episodes)
        # data collection uses full episodes
        episodes_used += bc_episodes

        _train_bc(
            policy_net,
            X,
            Y,
            action_ub=action_ub,
            lr=float(hyperparam.get("bc_lr", 3e-3)),
            epochs=int(hyperparam.get("bc_epochs", 25)),
            batch_size=int(hyperparam.get("bc_batch", 256)),
            weight_decay=float(hyperparam.get("bc_wd", 1e-4)),
            seed=seed,
        )

        # checkpoint NN
        checkpoint_eval()

    # ============================================================
    # (3) CEM in weight-space (derivative-free policy optimisation)
    # ============================================================
    # Weight bounds (keeps network outputs safe for the env internal loops)
    param_min = float(hyperparam.get("param_min", -1.0))
    param_max = float(hyperparam.get("param_max", 1.0))

    w_pop = int(hyperparam.get("w_pop", 20))
    w_elite = float(hyperparam.get("w_elite_frac", 0.2))
    w_iters = int(hyperparam.get("w_cem_iters", 60))
    w_eval_eps = int(hyperparam.get("w_eval_eps", 2))

    names, shapes = _get_param_shapes(policy_net)
    template_sd = policy_net.state_dict()
    w0 = _state_dict_to_vec(template_sd, names).astype(np.float32)

    mu = w0.copy()
    sigma = float(hyperparam.get("w_sigma_init", 0.10))
    sigma_decay = float(hyperparam.get("w_sigma_decay", 0.97))
    min_sigma = float(hyperparam.get("w_min_sigma", 0.01))
    n_elite = max(2, int(w_pop * w_elite))

    # Track best in weight search
    best_w = mu.copy()
    best_w_ret = -np.inf

    while True:
        if (time.time() - start_time) >= max_time:
            break
        if episodes_used + w_pop * w_eval_eps > max_episodes:
            break
        if w_iters <= 0:
            break

        w_iters -= 1

        eps = rng.standard_normal(size=(w_pop, mu.size)).astype(np.float32)
        cand = mu[None, :] + sigma * eps
        cand = np.clip(cand, param_min, param_max)

        fs = np.zeros(w_pop, dtype=np.float32)
        for i in range(w_pop):
            sd = _vec_to_state_dict(cand[i], template_sd, names, shapes)
            policy_net.load_state_dict(sd, strict=True)

            mean_r, _ = _eval_mean_return(env, w_eval_eps, policy_net=policy_net, action_ub=action_ub, rng=rng)
            episodes_used += w_eval_eps
            fs[i] = float(mean_r)

        j = int(np.argmax(fs))
        if fs[j] > best_w_ret:
            best_w_ret = float(fs[j])
            best_w = cand[j].copy()

        elite_idx = np.argsort(fs)[-n_elite:]
        mu = cand[elite_idx].mean(axis=0)

        sigma = max(min_sigma, sigma * sigma_decay)

        # Load current best and checkpoint it (with one cheap eval)
        sd_best = _vec_to_state_dict(best_w, template_sd, names, shapes)
        policy_net.load_state_dict(sd_best, strict=True)
        checkpoint_eval()

    # Final: ensure policy_net holds best policy
    sd_best = _vec_to_state_dict(best_w, template_sd, names, shapes)
    policy_net.load_state_dict(sd_best, strict=True)

    # Update global best policy_sd from final network weights
    if (time.time() - start_time) < max_time and episodes_used < max_episodes:
        checkpoint_eval()

    # ============================================================
    # Save best policy (unique filename)
    # ============================================================
    save_dir = os.path.join(".", "your_policies")
    os.makedirs(save_dir, exist_ok=True)

    existing = [f for f in os.listdir(save_dir) if f.startswith("optimal_policy_") and f.endswith(".pt")]
    run_num = len(existing)
    save_path = os.path.join(save_dir, f"optimal_policy_{run_num}.pt")
    torch.save(best_policy_sd, save_path)

    plot_info = {
        "episodes": np.array(episodes_curve, dtype=np.int32),
        "reward_history": np.array(reward_history, dtype=np.float32),
        "std_history": np.array(std_history, dtype=np.float32),
        "best_reward_history": np.array(best_reward_history, dtype=np.float32),
        "save_path": save_path,
        "episodes_used": int(episodes_used),
        "ss_params_best": ss_best.astype(np.float32),
    }

    return best_policy_sd, plot_info


# ============================================================
# REQUIRED METADATA (EDIT THESE)
# ============================================================
team_names = ["YOUR NAME 1", "YOUR NAME 2"]  # Imperial format
cids = ["YOUR CID 1", "YOUR CID 2"]
question = [1, 0]
