# algorithms/your_algorithm.py
# Uses only packages available in the provided environment (numpy/torch/matplotlib + stdlib)
# and respects the required template: returns (best_policy_state_dict, plot_info_dict).

import os
import time
import numpy as np
import torch

# matplotlib is available per requirements.txt; keep import local-safe for headless runs
import matplotlib
matplotlib.use("Agg")  # save figures without needing a display
import matplotlib.pyplot as plt


# -----------------------------
# Helpers: vectorize state_dict
# -----------------------------
def _get_param_shapes(policy_net):
    shapes = []
    names = []
    for n, p in policy_net.named_parameters():
        shapes.append(tuple(p.data.cpu().numpy().shape))
        names.append(n)
    return names, shapes


def _state_dict_to_vec(state_dict, names):
    vecs = []
    for n in names:
        v = state_dict[n].detach().cpu().numpy().ravel()
        vecs.append(v)
    return np.concatenate(vecs, axis=0).astype(np.float32)


def _vec_to_state_dict(vec, template_state_dict, names, shapes):
    out = {}
    idx = 0
    for n, shp in zip(names, shapes):
        size = int(np.prod(shp))
        chunk = vec[idx : idx + size].reshape(shp)
        out[n] = torch.tensor(chunk, dtype=template_state_dict[n].dtype)
        idx += size
    return out


def _clip_and_round_action(action_cont, action_ub):
    # action_cont: (n_actions,) float
    a = np.rint(action_cont).astype(np.int16)
    a = np.clip(a, 0, action_ub.astype(np.int16))
    return a


def _normalize_state(state, obs_ub):
    obs_ub = np.maximum(obs_ub.astype(np.float32), 1.0)
    return (state.astype(np.float32) / obs_ub).astype(np.float32)


# -----------------------------
# Heuristic (s,S) policy
# -----------------------------
def _ss_action_from_state(state, action_ub, n_retailers, n_dcs, ss_params):
    """
    ss_params = [s_r1, S_r1, s_r2, S_r2, ..., s_dc1, S_dc1, ...]
    state = [invpos_r1, invpos_r2, ..., invpos_dc1, ..., dow]
    """
    inv = state[:-1].astype(np.float32)
    n_entities = n_retailers + n_dcs
    a = np.zeros(n_entities, dtype=np.float32)

    for i in range(n_entities):
        s_i = float(ss_params[2 * i + 0])
        S_i = float(ss_params[2 * i + 1])
        if inv[i] <= s_i:
            a[i] = max(0.0, S_i - float(inv[i]))
        else:
            a[i] = 0.0

    return _clip_and_round_action(a, action_ub)


def _eval_episode_env(
    env,
    policy_net=None,
    action_ub=None,
    obs_ub=None,
    ss_params=None,
    deterministic=True,
):
    """
    Runs one episode, returns total reward.
    Either:
      - use ss_params (heuristic), or
      - use policy_net forward pass (NN).
    """
    env.reset()
    done = False
    ep_ret = 0.0

    n_actions = env.action_space.shape[0]
    n_retailers = len(env.retailers) if hasattr(env, "retailers") else (n_actions - 1)
    n_dcs = len(env.DCs) if hasattr(env, "DCs") else 1

    while not done:
        s = env.state

        if ss_params is not None:
            a = _ss_action_from_state(s, action_ub, n_retailers, n_dcs, ss_params)
        else:
            x = torch.tensor(_normalize_state(s, obs_ub), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                y = policy_net(x).squeeze(0).cpu().numpy()  # nonnegative due to ReLU in PolicyNetwork
            if not deterministic:
                y = y + np.random.normal(0.0, 0.10, size=y.shape).astype(np.float32) * action_ub.astype(np.float32)
                y = np.maximum(y, 0.0)
            a = _clip_and_round_action(y, action_ub)

        _, r, done, _ = env.step(a)
        ep_ret += float(r)

    return ep_ret


def _eval_mean_return(
    env,
    policy_net=None,
    action_ub=None,
    obs_ub=None,
    ss_params=None,
    n_episodes=3,
    deterministic=True,
):
    rets = []
    for _ in range(n_episodes):
        rets.append(
            _eval_episode_env(
                env,
                policy_net=policy_net,
                action_ub=action_ub,
                obs_ub=obs_ub,
                ss_params=ss_params,
                deterministic=deterministic,
            )
        )
    rets = np.array(rets, dtype=np.float32)
    return float(rets.mean()), float(rets.std(ddof=0))


# -----------------------------
# CEM optimiser (generic)
# -----------------------------
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
    eval_budget=None,  # mutable dict: {"used": int, "max": int}
    eval_cost_per_call=1,
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

        # Episode budget guard
        if eval_budget is not None:
            if eval_budget["used"] + eval_cost_per_call * pop_size > eval_budget["max"]:
                break

        # Sample
        eps = rng.standard_normal(size=(pop_size, dim)).astype(np.float32)
        xs = mu[None, :] + sigma * eps

        if x_min is not None:
            xs = np.maximum(xs, x_min[None, :])
        if x_max is not None:
            xs = np.minimum(xs, x_max[None, :])

        # Evaluate
        fs = np.zeros(pop_size, dtype=np.float32)
        for i in range(pop_size):
            fs[i] = float(fitness_fn(xs[i]))

        if eval_budget is not None:
            eval_budget["used"] += eval_cost_per_call * pop_size

        # Track best
        j = int(np.argmax(fs))
        if fs[j] > best_f:
            best_f = float(fs[j])
            best_x = xs[j].copy()

        # Elite update
        elite_idx = np.argsort(fs)[-n_elite:]
        elite = xs[elite_idx]
        mu = elite.mean(axis=0)

        # avoid collapse
        sigma = max(min_sigma, sigma * sigma_decay)

    return best_x, best_f


# -----------------------------
# Behaviour cloning
# -----------------------------
def _collect_bc_data(env, action_ub, obs_ub, ss_params, n_episodes=40):
    X = []
    Y = []
    n_actions = env.action_space.shape[0]
    n_retailers = len(env.retailers) if hasattr(env, "retailers") else (n_actions - 1)
    n_dcs = len(env.DCs) if hasattr(env, "DCs") else 1

    for _ in range(n_episodes):
        env.reset()
        done = False
        while not done:
            s = env.state
            a = _ss_action_from_state(s, action_ub, n_retailers, n_dcs, ss_params)
            X.append(_normalize_state(s, obs_ub))
            Y.append(a.astype(np.float32) / np.maximum(action_ub.astype(np.float32), 1.0))
            _, _, done, _ = env.step(a)

    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    return X, Y


def _train_bc(policy_net, X, Y, lr=3e-3, epochs=25, batch_size=256, weight_decay=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(policy_net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()

    n = X.shape[0]
    for _ in range(epochs):
        idx = rng.permutation(n)
        for i in range(0, n, batch_size):
            b = idx[i : i + batch_size]
            xb = torch.tensor(X[b], dtype=torch.float32)
            yb = torch.tensor(Y[b], dtype=torch.float32)
            pred = policy_net(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 5.0)
            opt.step()


# -----------------------------
# Plotting utilities (saved to disk)
# -----------------------------
def _save_training_plots(plot_info, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    episodes = plot_info["episodes"]
    mean_rewards = plot_info["reward_history"]
    std_rewards = plot_info["std_history"]
    best_so_far = plot_info["best_reward_history"]

    # 1) Learning curve (mean +/- std)
    p1 = os.path.join(out_dir, "learning_curve_mean_std.png")
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.plot(episodes, mean_rewards, label="Mean return")
    ax.fill_between(
        episodes,
        np.array(mean_rewards) - np.array(std_rewards),
        np.array(mean_rewards) + np.array(std_rewards),
        alpha=0.2,
    )
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("Episode return")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(p1, dpi=200)
    plt.close(fig)

    # 2) Best-so-far curve
    p2 = os.path.join(out_dir, "best_so_far.png")
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.plot(episodes, best_so_far, label="Best-so-far")
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("Best episode return")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(p2, dpi=200)
    plt.close(fig)

    # 3) Final reward distribution (optional; uses final_eval_rewards if present)
    p3 = None
    if "final_eval_rewards" in plot_info and plot_info["final_eval_rewards"] is not None:
        p3 = os.path.join(out_dir, "final_reward_distribution.png")
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        ax.hist(plot_info["final_eval_rewards"], density=True, edgecolor="black", alpha=0.6)
        ax.set_xlabel("Episode return")
        ax.set_ylabel("Density")
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(p3, dpi=200)
        plt.close(fig)

    return {"learning_curve": p1, "best_so_far": p2, "final_hist": p3}


# ============================================================
# Main entry point (MANDATORY signature)
# ============================================================
def your_optimization_alg(env, policy_net, *args, **hyperparam):
    """
    Returns
    -------
    policy_param : dict
        state_dict of best policy_net
    plot_info : dict
        dict containing arrays for plots + (optionally) saved plot paths
    """

    # -------- defaults (safe under 5 min / 5000 eps) --------
    max_episodes = int(hyperparam.get("max_episodes", 5000))
    max_time = float(hyperparam.get("max_time", 60 * 5))

    # Evaluation batching
    eval_eps_ss = int(hyperparam.get("eval_eps_ss", 4))
    eval_eps_nn = int(hyperparam.get("eval_eps_nn", 3))

    # Stage budgets
    ss_cem_iters = int(hyperparam.get("ss_cem_iters", 35))
    ss_pop = int(hyperparam.get("ss_pop", 30))

    bc_episodes = int(hyperparam.get("bc_episodes", 35))
    bc_epochs = int(hyperparam.get("bc_epochs", 25))

    nn_cem_iters = int(hyperparam.get("nn_cem_iters", 55))
    nn_pop = int(hyperparam.get("nn_pop", 26))

    # Plot saving
    save_plots = bool(hyperparam.get("save_plots", True))
    plot_dir = hyperparam.get("plot_dir", os.path.join(".", "training_plots"))

    # --------- spaces ---------
    action_ub = env.action_space.high.astype(np.int16)
    obs_ub = env.observation_space.high.astype(np.int16)

    n_actions = env.action_space.shape[0]
    n_entities = n_actions  # retailers + DCs
    ss_dim = 2 * n_entities

    start = time.time()

    # Episode-budget tracker (best-effort; evaluation calls are the main cost here)
    eval_budget = {"used": 0, "max": max_episodes}

    # --------- plot buffers (checkpoints) ---------
    checkpoints = []
    reward_history = []
    std_history = []
    best_reward_history = []

    best_policy_sd = {k: v.detach().clone() for k, v in policy_net.state_dict().items()}
    best_return = -np.inf

    # ============================================================
    # (1) CEM tune a strong (s,S) heuristic
    # ============================================================
    x0 = np.zeros(ss_dim, dtype=np.float32)
    for i in range(n_entities):
        if i < n_entities - 1:  # typically retailers
            s0, S0 = 10.0, 45.0
        else:  # typically DC
            s0, S0 = 60.0, 140.0
        x0[2 * i + 0] = s0
        x0[2 * i + 1] = S0

    # Bounds: s,S in [0, inv_pos_ub] (inv_pos_ub = capacity + order_limit)
    cap_plus = (obs_ub[:-1]).astype(np.float32)
    x_min = np.zeros(ss_dim, dtype=np.float32)
    x_max = np.zeros(ss_dim, dtype=np.float32)
    for i in range(n_entities):
        hi = float(cap_plus[i])
        x_min[2 * i + 0] = 0.0
        x_max[2 * i + 0] = hi
        x_min[2 * i + 1] = 0.0
        x_max[2 * i + 1] = hi

    def ss_fitness(x):
        xx = x.copy()
        for i in range(n_entities):
            s_i = xx[2 * i + 0]
            xx[2 * i + 1] = max(xx[2 * i + 1], s_i)  # enforce S >= s
        m, _ = _eval_mean_return(
            env,
            ss_params=xx,
            action_ub=action_ub,
            obs_ub=obs_ub,
            n_episodes=eval_eps_ss,
            deterministic=True,
        )
        return m

    ss_best, ss_best_ret = _cem_optimize(
        ss_fitness,
        dim=ss_dim,
        x_init=x0,
        sigma_init=15.0,
        pop_size=ss_pop,
        elite_frac=0.25,
        n_iters=ss_cem_iters,
        x_min=x_min,
        x_max=x_max,
        seed=0,
        sigma_decay=0.93,
        min_sigma=1.0,
        time_budget_s=max_time * 0.35,
        start_time=start,
        eval_budget=eval_budget,
        eval_cost_per_call=eval_eps_ss,
    )

    checkpoints.append(0)
    reward_history.append(ss_best_ret)
    std_history.append(0.0)
    best_return = max(best_return, ss_best_ret)
    best_reward_history.append(best_return)

    # ============================================================
    # (2) Behaviour cloning: imitate tuned heuristic into NN
    # ============================================================
    # Reduce BC episodes if budget is tight
    bc_episodes_eff = int(min(bc_episodes, max(1, (eval_budget["max"] - eval_budget["used"]) // 2)))
    X, Y = _collect_bc_data(env, action_ub, obs_ub, ss_best, n_episodes=bc_episodes_eff)
    _train_bc(policy_net, X, Y, lr=3e-3, epochs=bc_epochs, batch_size=256, weight_decay=1e-4, seed=0)

    def _eval_nn_det():
        m, s = _eval_mean_return(
            env,
            policy_net=policy_net,
            action_ub=action_ub,
            obs_ub=obs_ub,
            n_episodes=eval_eps_nn,
            deterministic=True,
        )
        return m, s

    nn_mean, nn_std = _eval_nn_det()
    eval_budget["used"] += eval_eps_nn

    checkpoints.append(1)
    reward_history.append(nn_mean)
    std_history.append(nn_std)
    if nn_mean > best_return:
        best_return = nn_mean
        best_policy_sd = {k: v.detach().clone() for k, v in policy_net.state_dict().items()}
    best_reward_history.append(best_return)

    # ============================================================
    # (3) CEM in weight-space (derivative-free policy optimisation)
    # ============================================================
    names, shapes = _get_param_shapes(policy_net)
    template_sd = policy_net.state_dict()
    w0 = _state_dict_to_vec(template_sd, names)

    def w_fitness(wvec):
        sd = _vec_to_state_dict(wvec, template_sd, names, shapes)
        policy_net.load_state_dict(sd, strict=True)
        m, _ = _eval_mean_return(
            env,
            policy_net=policy_net,
            action_ub=action_ub,
            obs_ub=obs_ub,
            n_episodes=eval_eps_nn,
            deterministic=True,
        )
        return m

    remaining_time = max(1.0, max_time - (time.time() - start))
    w_best, w_best_ret = _cem_optimize(
        w_fitness,
        dim=w0.size,
        x_init=w0,
        sigma_init=0.08,  # small steps; BC already decent
        pop_size=nn_pop,
        elite_frac=0.2,
        n_iters=nn_cem_iters,
        x_min=None,
        x_max=None,
        seed=1,
        sigma_decay=0.96,
        min_sigma=0.01,
        time_budget_s=remaining_time * 0.95,
        start_time=time.time(),
        eval_budget=eval_budget,
        eval_cost_per_call=eval_eps_nn,
    )

    best_sd_candidate = _vec_to_state_dict(w_best, template_sd, names, shapes)
    policy_net.load_state_dict(best_sd_candidate, strict=True)
    final_mean, final_std = _eval_nn_det()
    eval_budget["used"] += eval_eps_nn

    checkpoints.append(2)
    reward_history.append(final_mean)
    std_history.append(final_std)
    if final_mean > best_return:
        best_return = final_mean
        best_policy_sd = {k: v.detach().clone() for k, v in policy_net.state_dict().items()}
    best_reward_history.append(best_return)

    # Final eval distribution (small, to keep within budget/time)
    final_eval_n = int(hyperparam.get("final_eval_n", 10))
    # budget guard
    final_eval_n = int(min(final_eval_n, max(1, eval_budget["max"] - eval_budget["used"])))
    final_rewards = None
    if final_eval_n > 0:
        final_rewards = []
        for _ in range(final_eval_n):
            final_rewards.append(
                _eval_episode_env(
                    env,
                    policy_net=policy_net,
                    action_ub=action_ub,
                    obs_ub=obs_ub,
                    ss_params=None,
                    deterministic=True,
                )
            )
        eval_budget["used"] += final_eval_n
        final_rewards = np.array(final_rewards, dtype=np.float32)

    # ============================================================
    # Save best policy (do not overwrite if possible)
    # ============================================================
    save_dir = os.path.join(".", "your_policies")
    os.makedirs(save_dir, exist_ok=True)
    # unique-ish filename
    save_path = os.path.join(save_dir, f"optimal_policy_{int(time.time())}.pt")
    torch.save(best_policy_sd, save_path)

    # ============================================================
    # Assemble plot_info (REQUIRED output)
    # ============================================================
    plot_info = {
        "episodes": np.array(checkpoints, dtype=np.int32),  # checkpoint index (not env episodes)
        "reward_history": np.array(reward_history, dtype=np.float32),
        "std_history": np.array(std_history, dtype=np.float32),
        "best_reward_history": np.array(best_reward_history, dtype=np.float32),
        "save_path": save_path,
        "ss_params_best": ss_best.astype(np.float32),
        "final_eval_rewards": None if final_rewards is None else final_rewards,
        "eval_budget_used": int(eval_budget["used"]),
        "eval_budget_max": int(eval_budget["max"]),
        "time_used_s": float(time.time() - start),
        "time_budget_s": float(max_time),
    }

    # Save plots to disk and include file paths (nice for notebook/report)
    if save_plots:
        plot_paths = _save_training_plots(plot_info, plot_dir)
        plot_info["plot_paths"] = plot_paths
    else:
        plot_info["plot_paths"] = None

    return best_policy_sd, plot_info


if __name__ == "__main__":
    # Local quick test
    from ML4CE_RL_environment import MESCEnv
    from common import PolicyNetwork

    env = MESCEnv([[2], [1], 1], num_periods=28)
    policy = PolicyNetwork(
        input_size=env.observation_space.shape[0],
        output_size=env.action_space.shape[0],
    )

    best_sd, plot_info = your_optimization_alg(
        env,
        policy,
        max_episodes=500,
        max_time=30,
        save_plots=True,
        plot_dir="./training_plots_demo",
        final_eval_n=10,
    )

    print("\n=== TRAINING RESULTS ===")
    print("Reward checkpoints:", plot_info["reward_history"])
    print("Best-so-far:", plot_info["best_reward_history"])
    print("Saved policy:", plot_info["save_path"])
    print("Saved plots:", plot_info.get("plot_paths", None))


# ============================================================
# REQUIRED METADATA (EDIT THESE)
# ============================================================
team_names = ["YOUR NAME 1", "YOUR NAME 2"]  # Imperial format
cids = ["YOUR CID 1", "YOUR CID 2"]
question = [1, 0]
