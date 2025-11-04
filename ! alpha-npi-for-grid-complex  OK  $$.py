# alpha_npi_repro.py
# Research-grade reproducible prototype:
# - True program stack (call_sub pushes, return pops)
# - Parameterized subprograms, per-subprogram parameter distributions
# - MCTS used to produce supervised policy targets (AlphaZero style)
# - Mixed training: PPO (actor-critic) + MCTS supervised loss (cross-entropy)
# - Replay buffers for transitions and MCTS targets
#
# Usage: python alpha_npi_repro.py
# Requirements: torch, numpy

import math
import time
import random
from collections import deque, Counter, defaultdict
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Task (3x3 grid transforms)
# ----------------------------
class Task:
    class Action:
        ATOMIC_ACTIONS = [
            ('translate', (-1, 0)),
            ('translate', (+1, 0)),
            ('translate', (0, -1)),
            ('translate', (0, +1)),
            ('rotate', (90,)),
            ('rotate', (180,)),
            ('rotate', (270,)),
            ('flip', ('horizontal',)),
            ('flip', ('vertical',))
        ]
        ACTION_TO_ID = {action: idx for idx, action in enumerate(ATOMIC_ACTIONS)}
        ID_TO_ACTION = {idx: action for action, idx in ACTION_TO_ID.items()}

        @staticmethod
        def translate(state, dx, dy):
            new_grid = np.roll(state.real, shift=(dy, dx), axis=(0, 1))
            return Task.State(np.array(new_grid), state.goal)

        @staticmethod
        def rotate(state, angle):
            k = int((angle // 90) % 4)
            new_grid = np.rot90(state.real, k=k)
            return Task.State(np.array(new_grid), state.goal)

        @staticmethod
        def flip(state, axis):
            new_grid = np.fliplr(state.real) if axis == 'horizontal' else np.flipud(state.real)
            return Task.State(np.array(new_grid), state.goal)

    class State:
        GRID_X = 3
        GRID_Y = 3

        def __init__(self, real, goal):
            self.real = np.array(real)
            self.goal = np.array(goal) if goal is not None else None

        def has_solved(self):
            return np.array_equal(self.real, self.goal)

        def get_reward(self):
            return 1.0 if self.has_solved() else 0.0

        def get_penalty(self):
            return 0.01

        def get_valid_actions(self):
            return Task.get_whole_actions()

        def mix_tensor(self):
            real_flat = self.real.flatten().astype(np.float32)
            goal_flat = self.goal.flatten().astype(np.float32) if self.goal is not None else np.zeros_like(real_flat)
            return torch.tensor(np.concatenate([real_flat, goal_flat]), dtype=torch.float32).unsqueeze(0)

    @staticmethod
    def get_dimension_state():
        return Task.State.GRID_X * Task.State.GRID_Y

    @staticmethod
    def get_whole_actions():
        return Task.Action.ATOMIC_ACTIONS

    @staticmethod
    def run_action_state(action, state):
        act_name, params = action
        return getattr(Task.Action, act_name)(state, *params)

    @staticmethod
    def gen_task(max_complex):
        init = np.random.randint(0, 2, (Task.State.GRID_Y, Task.State.GRID_X))
        init_state = Task.State(init, None)
        goal_state = init_state
        atomic_seq = []
        for _ in range(np.random.randint(1, max_complex + 1)):
            actions = Task.get_whole_actions()
            action = random.choice(actions)
            atomic_seq.append(action)
            goal_state = Task.run_action_state(action, goal_state)
        init_state.goal = goal_state.real
        return init_state, goal_state, atomic_seq

# ----------------------------
# Program Library: stores programs, param spec, embeddings, stats
# ----------------------------
class ProgramLibrary:
    def __init__(self, max_subs=10, emb_dim=32, max_param_dim=2, device='cpu'):
        self.max_subs = max_subs
        self.emb_dim = emb_dim
        self.max_param_dim = max_param_dim
        self.device = device
        self.programs = {}  # pid -> dict
        # register root program id 0
        self.register(sequence=[], param_dim=0, param_type='noop', is_root=True)
        # stats
        self.usage = Counter()
        self.success = Counter()

    def register(self, sequence, param_dim=0, param_type='noop', is_root=False):
        if len(self.programs) >= self.max_subs:
            print("[ProgramLib] max reached")
            return None
        pid = max(self.programs.keys()) + 1 if self.programs else 0
        emb = torch.randn(self.emb_dim, device=self.device) * 0.01
        # program param_logstd is per-program; initialize small
        param_logstd = torch.zeros(self.max_param_dim, device=self.device) - 1.0
        self.programs[pid] = {
            'sequence': sequence,
            'param_dim': param_dim,
            'param_type': param_type,
            'embedding': emb,
            'param_logstd': param_logstd,
            'created_at': time.time(),
            'is_root': is_root
        }
        print(f"[ProgramLib] Registered pid={pid} seq_len={len(sequence)} param_dim={param_dim} type={param_type}")
        return pid

    def get_embedding(self, pid):
        if pid not in self.programs:
            pid = 0
        return self.programs[pid]['embedding'].unsqueeze(0)

    def get_param_logstd(self, pid):
        if pid not in self.programs:
            pid = 0
        return self.programs[pid]['param_logstd']

    def get(self, pid):
        return self.programs.get(pid, None)

    def num_programs(self):
        return len(self.programs)

    def record_usage(self, pid, success):
        self.usage[pid] += 1
        if success:
            self.success[pid] += 1

# ----------------------------
# Replay buffers
# - trans_buffer: for PPO (transitions)
# - mcts_buffer: (state, prog_id, pi_mcts, v_mcts) for supervised loss
# ----------------------------
class TransitionBuffer:
    def __init__(self, capacity=20000):
        self.capacity = capacity
        self.buf = deque(maxlen=capacity)

    def add(self, item):
        self.buf.append(item)

    def sample_all(self):
        return list(self.buf)

    def clear(self):
        self.buf.clear()

    def __len__(self):
        return len(self.buf)

class MctsTargetBuffer:
    def __init__(self, capacity=5000):
        self.capacity = capacity
        self.buf = deque(maxlen=capacity)

    def add(self, item):
        self.buf.append(item)

    def sample(self, k):
        k = min(k, len(self.buf))
        idxs = random.sample(range(len(self.buf)), k)
        return [self.buf[i] for i in idxs]

    def __len__(self):
        return len(self.buf)

# ----------------------------
# Mind network: policy logits, value, per-program param means; per-program embeddings are provided by ProgramLibrary
# Output ordering: [call_sub_0..call_sub_max-1, atomic_0..atomic_K-1, return]
# ----------------------------
class Mind(nn.Module):
    def __init__(self, state_dim, max_subs, n_atomic, max_param_dim=2, emb_dim=32, hidden=256, device='cpu'):
        super().__init__()
        self.device = device
        self.state_dim = state_dim
        self.max_subs = max_subs
        self.n_atomic = n_atomic
        self.max_param_dim = max_param_dim
        self.emb_dim = emb_dim
        self.out_dim = self.max_subs + self.n_atomic + 1
        # simple MLP
        self.fc1 = nn.Linear(state_dim + emb_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.policy_head = nn.Linear(hidden, self.out_dim)
        self.value_head = nn.Linear(hidden, 1)
        # per-subprogram param means: output max_subs * max_param_dim
        self.param_head = nn.Linear(hidden, self.max_subs * self.max_param_dim)
        # initialize
        self.to(self.device)

    def forward(self, state: Task.State, prog_emb: torch.Tensor):
        # state.mix_tensor -> [1, state_dim]; prog_emb -> [1, emb_dim]
        s = state.mix_tensor().to(self.device)
        x = torch.cat([s, prog_emb.to(self.device)], dim=-1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        logits = self.policy_head(h)
        probs = F.softmax(logits, dim=-1)
        value = torch.tanh(self.value_head(h))
        params = self.param_head(h).view(self.max_subs, self.max_param_dim)  # [max_subs, max_param_dim]
        return probs, value, params  # params are means; logstds are per-program in ProgramLibrary

# ----------------------------
# Single-thread MCTS (AlphaZero style) that supports (state, prog_stack) nodes and param actions
# - For param actions, we treat the param continuous component by using network's param means and per-program logstd to sample
# - MCTS uses PUCT with dirichlet noise at root
# - Returns: root node (with visit counts) and value estimate
# ----------------------------
class MCTS:
    class Node:
        def __init__(self, state, prog_stack, parent=None, action_from_parent=None, depth=0):
            self.state = state
            self.prog_stack = list(prog_stack)  # copy
            self.parent = parent
            self.action_from_parent = action_from_parent
            self.children = {}  # action_key -> Node
            self.N = defaultdict(int)   # visit count per action_key
            self.W = defaultdict(float) # total value per action_key
            self.Q = defaultdict(float) # mean value per action_key
            self.P = {}  # prior prob per action_key
            self.expanded = False
            self.depth = depth

        def best_child_by_ucb(self, c_puct):
            best_score = -float('inf')
            best_key = None
            total_N = sum(self.N.values()) + 1e-8
            for k, prior in self.P.items():
                q = self.Q.get(k, 0.0)
                n = self.N.get(k, 0)
                u = c_puct * prior * math.sqrt(total_N) / (1 + n)
                score = q + u
                if score > best_score:
                    best_score = score
                    best_key = k
            return best_key

    def __init__(self, mind: Mind, lib: ProgramLibrary, c_puct=1.0, max_depth=30, dirichlet_alpha=0.3, dirichlet_eps=0.25):
        self.mind = mind
        self.lib = lib
        self.c_puct = c_puct
        self.max_depth = max_depth
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps

    def search(self, root_state: Task.State, root_stack, n_sim=50):
        root = MCTS.Node(root_state, root_stack, depth=0)
        for _ in range(n_sim):
            node = root
            # selection
            path = [node]
            while node.expanded and node.depth < self.max_depth:
                key = node.best_child_by_ucb(self.c_puct)
                if key is None:
                    break
                # move to child (create placeholder if not exist or is None)
                if key not in node.children or node.children[key] is None:
                    # create child node; child prog_stack and state depend on action
                    child_state, child_stack = self._simulate_action_result(node.state, node.prog_stack, key)
                    node.children[key] = MCTS.Node(child_state, child_stack, parent=node, action_from_parent=key, depth=node.depth + 1)
                node = node.children[key]
                path.append(node)

            # expand and evaluate
            leaf = node
            if leaf.state.has_solved():
                leaf_value = leaf.state.get_reward()
            else:
                # network prior & value
                top_pid = leaf.prog_stack[-1] if len(leaf.prog_stack) > 0 else 0
                prog_emb = self.lib.get_embedding(top_pid).detach()
                probs, v, param_means = self.mind(leaf.state, prog_emb)
                probs = probs.squeeze(0).detach().cpu().numpy()
                v = float(v.item())
                param_means = param_means.detach().cpu().numpy()  # [max_subs, max_param_dim]

                # build action keys
                keys = []
                for pid in range(self.mind.max_subs):
                    keys.append(('call_sub', pid))
                for ai in range(self.mind.n_atomic):
                    keys.append(('atomic', ai))
                keys.append(('return', None))

                # root dirichlet noise
                priors = {}
                if leaf is root:
                    valid = len(keys)
                    dir_noise = np.random.dirichlet([self.dirichlet_alpha] * valid)
                    for i, k in enumerate(keys):
                        priors[k] = (1 - self.dirichlet_eps) * float(probs[i]) + self.dirichlet_eps * dir_noise[i]
                else:
                    for i, k in enumerate(keys):
                        priors[k] = float(probs[i])

                # fill node.P and create children placeholders (keep None)
                leaf.P = priors
                for k in keys:
                    if k not in leaf.children:
                        leaf.children[k] = None
                leaf.expanded = True
                leaf_value = v

            # backup
            self._backup(path, leaf_value)
        return root

    def _simulate_action_result(self, state: Task.State, prog_stack, action_key):
        # simulate effect of action without running policy: atomic immediate state change; call_sub: push; return: pop
        atype = action_key[0]
        if atype == 'atomic':
            ai = action_key[1]
            action = Task.Action.ATOMIC_ACTIONS[ai]
            new_state = Task.run_action_state(action, state)
            new_stack = list(prog_stack)
            return new_state, new_stack
        elif atype == 'call_sub':
            pid = action_key[1]
            new_stack = list(prog_stack) + [pid]
            # no immediate state change (child will decide actions)
            return state, new_stack
        else:  # return
            new_stack = list(prog_stack)
            if len(new_stack) > 1:
                new_stack.pop()
            return state, new_stack

    def _backup(self, path, value, gamma=0.99):
        for node in reversed(path):
            # parent update using action_from_parent
            if node.parent is not None:
                k = node.action_from_parent
                node.parent.N[k] += 1
                node.parent.W[k] += value
                node.parent.Q[k] = node.parent.W[k] / node.parent.N[k]
            value *= gamma

# ----------------------------
# Agent with true stack execution and parameterized calls
# - Uses MCTS to produce supervised targets at decision points
# - Stores transitions for PPO and (state, prog_id, pi_mcts, v_mcts) for supervised loss
# ----------------------------
class AlphaNPIAgent:
    def __init__(self, lib: ProgramLibrary, mind: Mind, trans_buf: TransitionBuffer, mcts_buf: MctsTargetBuffer,
                 mcts_simulations=50, c_puct=1.0, device='cpu'):
        self.lib = lib
        self.mind = mind
        self.trans_buf = trans_buf
        self.mcts_buf = mcts_buf
        self.device = device
        self.mcts = MCTS(mind, lib, c_puct=c_puct)
        self.mcts_simulations = mcts_simulations

    def rollout_episode(self, init_state: Task.State, max_steps=50, use_mcts=True):
        state = copy.deepcopy(init_state)
        prog_stack = [0]
        done = False
        steps = 0
        episode = []

        # === 新增：调用轨迹记录 ===
        call_trace = []

        while not done and steps < max_steps:
            top_pid = prog_stack[-1]
            # run MCTS to get improved policy & value
            if use_mcts:
                root = self.mcts.search(state, prog_stack, n_sim=self.mcts_simulations)
                keys = []
                for pid in range(self.mind.max_subs):
                    keys.append(('call_sub', pid))
                for ai in range(self.mind.n_atomic):
                    keys.append(('atomic', ai))
                keys.append(('return', None))
                counts = np.array([root.N.get(k, 0) for k in keys], dtype=np.float32)
                if counts.sum() == 0:
                    pi_mcts = np.ones_like(counts) / len(counts)
                else:
                    pi_mcts = counts / counts.sum()
                prog_emb = self.lib.get_embedding(top_pid).detach()
                probs_net, v_net, param_means = self.mind(state, prog_emb)
                v_mcts = float(v_net.item())
            else:
                prog_emb = self.lib.get_embedding(top_pid).detach()
                probs_net, v_net, param_means = self.mind(state, prog_emb)
                pi_mcts = None
                v_mcts = float(v_net.item())

            if pi_mcts is not None:
                self.mcts_buf.add((copy.deepcopy(state), top_pid, pi_mcts.astype(np.float32), v_mcts))

            if pi_mcts is not None:
                idx = np.random.choice(len(pi_mcts), p=pi_mcts)
            else:
                probs = probs_net.squeeze(0).cpu().numpy()
                idx = np.random.choice(len(probs), p=probs)
            action_key = self._index_to_action_key(idx)

            joint_logp = 0.0
            sampled_params = None
            if action_key[0] == 'call_sub':
                pid = action_key[1]
                prog = self.lib.get(pid)
                if prog is None:
                    sampled_params = None
                    action_logp = math.log(1e-8)
                    joint_logp = action_logp
                else:
                    pdim = prog['param_dim']
                    if pi_mcts is not None:
                        action_logp = math.log(max(pi_mcts[idx], 1e-8))
                    else:
                        probs = probs_net.squeeze(0).cpu().numpy()
                        action_logp = math.log(max(probs[idx], 1e-8))
                    if pdim > 0:
                        param_means_arr = param_means
                        mean = param_means_arr[pid, :pdim].detach().cpu().numpy()
                        logstd = prog['param_logstd'].detach().cpu().numpy()[:pdim]
                        std = np.exp(logstd)
                        sampled_params = np.random.normal(loc=mean, scale=std, size=(pdim,))
                        logp_params = -0.5 * (((sampled_params - mean) / (std + 1e-8)) ** 2 +
                                              2 * np.log(std + 1e-8) + np.log(2 * np.pi))
                        logp_params = float(np.sum(logp_params))
                        joint_logp = action_logp + logp_params
                    else:
                        joint_logp = action_logp
            elif action_key[0] == 'atomic':
                if pi_mcts is not None:
                    action_logp = math.log(max(pi_mcts[idx], 1e-8))
                else:
                    probs = probs_net.squeeze(0).cpu().numpy()
                    action_logp = math.log(max(probs[idx], 1e-8))
                joint_logp = action_logp
            else:
                if pi_mcts is not None:
                    action_logp = math.log(max(pi_mcts[idx], 1e-8))
                else:
                    probs = probs_net.squeeze(0).cpu().numpy()
                    action_logp = math.log(max(probs[idx], 1e-8))
                joint_logp = action_logp

            # === 在执行动作前记录轨迹 ===
            call_trace.append((action_key[0], action_key[1], list(prog_stack)))

            # execute action with true stack semantics
            if action_key[0] == 'atomic':
                atom_idx = action_key[1]
                action = Task.Action.ATOMIC_ACTIONS[atom_idx]
                next_state = Task.run_action_state(action, state)
                reward = next_state.get_reward() - next_state.get_penalty()
            elif action_key[0] == 'call_sub':
                pid = action_key[1]
                prog_stack.append(pid)
                next_state = state
                reward = 0.0
            else:  # return
                if len(prog_stack) > 1:
                    prog_stack.pop()
                next_state = state
                reward = 0.0

            done = next_state.has_solved()
            logp_tensor = torch.tensor(joint_logp, dtype=torch.float32)
            val_est = v_mcts
            self.trans_buf.add((copy.deepcopy(state), list(prog_stack),
                                action_key, sampled_params, logp_tensor,
                                reward, copy.deepcopy(next_state), done, val_est))
            state = next_state
            steps += 1

        success = state.has_solved()
        for trans in list(self.trans_buf.buf)[-steps:]:
            _, prog_stack_snap, action_key, _, _, _, _, _, _ = trans
            if prog_stack_snap:
                pid_used = prog_stack_snap[-1]
                self.lib.record_usage(pid_used, success)

        return success, call_trace


    def _index_to_action_key(self, idx):
        if idx < self.mind.max_subs:
            return ('call_sub', int(idx))
        elif idx < self.mind.max_subs + self.mind.n_atomic:
            return ('atomic', int(idx - self.mind.max_subs))
        else:
            return ('return', None)

# ----------------------------
# Trainer: mixes PPO and MCTS supervised loss
# - Uses transitions buffer for PPO (with GAE)
# - Uses mcts buffer for supervised cross-entropy towards pi_mcts
# ----------------------------
class Trainer:
    def __init__(self, lib: ProgramLibrary, mind: Mind, trans_buf: TransitionBuffer, mcts_buf: MctsTargetBuffer,
                 device='cpu', gamma=0.99, lam=0.95, ppo_epochs=4, mini_batch=64,
                 clip_eps=0.2, lr=2e-4, mcts_supervision_coef=1.0, value_coef=0.5, entropy_coef=0.01):
        self.lib = lib
        self.mind = mind
        self.trans_buf = trans_buf
        self.mcts_buf = mcts_buf
        self.device = device
        self.gamma = gamma
        self.lam = lam
        self.ppo_epochs = ppo_epochs
        self.mini_batch = mini_batch
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.mcts_supervision_coef = mcts_supervision_coef
        self.optimizer = torch.optim.Adam(mind.parameters(), lr=lr)

    def compute_gae(self, rewards, values, dones):
        adv = []
        gae = 0.0
        next_v = 0.0
        for r, v, done in zip(reversed(rewards), reversed(values), reversed(dones)):
            mask = 0.0 if done else 1.0
            delta = r + self.gamma * next_v * mask - v
            gae = delta + self.gamma * self.lam * mask * gae
            adv.insert(0, gae)
            next_v = v
        returns = [a + v for a, v in zip(adv, values)]
        return adv, returns

    def ppo_update(self):
        # aggregate episodes from transitions buffer by episode (split by done)
        data = list(self.trans_buf.buf)
        if not data:
            return
        episodes = []
        cur = []
        for it in data:
            cur.append(it)
            if it[7]:
                episodes.append(cur)
                cur = []
        if cur:
            episodes.append(cur)  # include last partial episode

        # build flat lists
        flat_states = []
        flat_prog_embs = []
        flat_actions_idx = []
        flat_old_logp = []
        flat_rewards = []
        flat_dones = []
        flat_values = []

        for ep in episodes:
            rewards = [it[5] for it in ep]
            values = [it[8] for it in ep]
            dones = [it[6] for it in ep]
            advs, rets = self.compute_gae(rewards, values, dones)
            for (state, prog_stack, action_key, params, old_logp, reward, next_state, done, v_est), adv, ret in zip(ep, advs, rets):
                flat_states.append(state)
                pid = prog_stack[-1] if len(prog_stack) > 0 else 0
                flat_prog_embs.append(self.lib.get_embedding(pid).detach().to(self.device))
                flat_actions_idx.append(self._action_key_to_index(action_key))
                flat_old_logp.append(old_logp.to(self.device))
                flat_rewards.append(reward)
                flat_dones.append(done)
                flat_values.append(v_est)
        if not flat_states:
            return

        N = len(flat_states)
        actions_t = torch.tensor(flat_actions_idx, dtype=torch.long, device=self.device)
        old_logp_t = torch.stack(flat_old_logp).to(self.device)
        advs = []
        rets = []
        # Recompute advs/returns properly:
        for ep in episodes:
            rewards = [it[5] for it in ep]
            values = [it[8] for it in ep]
            dones = [it[6] for it in ep]
            adv_ep, ret_ep = self.compute_gae(rewards, values, dones)
            advs.extend(adv_ep)
            rets.extend(ret_ep)
        adv_tensor = torch.tensor(advs, dtype=torch.float32, device=self.device)
        ret_tensor = torch.tensor(rets, dtype=torch.float32, device=self.device)
        adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

        # training epochs
        idxs = np.arange(N)
        for _ in range(self.ppo_epochs):
            np.random.shuffle(idxs)
            # =============================
            # PPO 更新部分
            # =============================
            for start in range(0, N, self.mini_batch):
                mb = idxs[start:start + self.mini_batch]
                mb_states = [flat_states[i] for i in mb]
                mb_embs = [flat_prog_embs[i] for i in mb]  # list of tensors
                mb_actions = actions_t[mb]
                mb_old_logp = old_logp_t[mb]
                mb_adv = adv_tensor[mb]
                mb_ret = ret_tensor[mb]

                # forward per sample
                new_probs = []
                new_vals = []
                for s, p_emb in zip(mb_states, mb_embs):
                    # 确保 p_emb 是 [1, emb_dim]
                    if p_emb.dim() == 1:
                        p_emb = p_emb.unsqueeze(0)
                    probs, val, param_means = self.mind(s, p_emb)
                    new_probs.append(probs)
                    new_vals.append(val)
                new_probs = torch.cat(new_probs, dim=0)  # [B, out_dim]
                new_vals = torch.cat(new_vals, dim=0).view(-1)

                # PPO 损失计算
                new_logp = torch.log(torch.gather(new_probs, 1, mb_actions.unsqueeze(1)).squeeze(1) + 1e-8)
                ratio = torch.exp(new_logp - mb_old_logp)
                s1 = ratio * mb_adv
                s2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_adv
                policy_loss = -torch.mean(torch.min(s1, s2))
                value_loss = F.mse_loss(new_vals, mb_ret)
                entropy = -(new_probs * torch.log(new_probs + 1e-8)).sum(dim=1).mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.mind.parameters(), 0.5)
                self.optimizer.step()


            # =============================
            # MCTS 监督损失部分
            # =============================
            if len(self.mcts_buf) > 0:
                mcts_samples = self.mcts_buf.sample(min(len(self.mcts_buf), 256))
                states = [it[0] for it in mcts_samples]
                pids = [it[1] for it in mcts_samples]
                pi_targets = np.stack([it[2] for it in mcts_samples], axis=0)  # [B, out_dim]
                v_targets = [it[3] for it in mcts_samples]

                # convert to tensors
                pi_targets_t = torch.tensor(pi_targets, dtype=torch.float32, device=self.device)
                v_targets_t = torch.tensor(v_targets, dtype=torch.float32, device=self.device)

                policy_preds = []
                value_preds = []
                for s, pid in zip(states, pids):
                    emb = self.lib.get_embedding(pid).detach().to(self.device)
                    # 确保 emb 是 [1, emb_dim]
                    if emb.dim() == 1:
                        emb = emb.unsqueeze(0)
                    probs, v, param_means = self.mind(s, emb)
                    policy_preds.append(probs)
                    value_preds.append(v)
                policy_preds = torch.cat(policy_preds, dim=0)  # [B, out_dim]
                value_preds = torch.cat(value_preds, dim=0).view(-1)

                # cross-entropy loss
                ce = -torch.sum(pi_targets_t * torch.log(policy_preds + 1e-8), dim=1).mean()
                v_loss = F.mse_loss(value_preds, v_targets_t)
                loss_sup = self.mcts_supervision_coef * ce + 0.5 * v_loss

                self.optimizer.zero_grad()
                loss_sup.backward()
                nn.utils.clip_grad_norm_(self.mind.parameters(), 0.5)
                self.optimizer.step()

    def _action_key_to_index(self, action_key):
        if action_key is None:
            return self.mind.out_dim - 1
        if action_key[0] == 'call_sub':
            return int(action_key[1])
        elif action_key[0] == 'atomic':
            return self.mind.max_subs + int(action_key[1])
        else:
            return self.mind.out_dim - 1

# ----------------------------
# Program discovery (simple frequent n-gram on successful episodes)
# ----------------------------
def program_discovery(lib: ProgramLibrary, trans_buf: TransitionBuffer, min_freq=6, min_success_rate=0.7, ngram_max=4, recent_n=1000):
    # collect recent episodes
    data = list(trans_buf.buf)[-recent_n:]
    episodes = []
    cur = []
    for it in data:
        cur.append(it)
        if it[7]:
            episodes.append(cur)
            cur = []
    # build atomic sequences + success flags
    sequences = []
    for ep in episodes:
        seq = []
        success = False
        for (s, prog_stack, action_key, params, logp, reward, ns, done, v) in ep:
            if action_key[0] == 'atomic':
                seq.append(action_key[1])
            if done:
                success = ns.has_solved()
        sequences.append((seq, success))
    # ngram counts
    ngram_counts = Counter()
    ngram_success = Counter()
    for seq, success in sequences:
        L = len(seq)
        for n in range(1, min(ngram_max, L) + 1):
            for i in range(0, L - n + 1):
                ng = tuple(seq[i:i + n])
                ngram_counts[ng] += 1
                if success:
                    ngram_success[ng] += 1
    candidates = []
    for ng, cnt in ngram_counts.items():
        succ = ngram_success.get(ng, 0)
        rate = succ / cnt if cnt > 0 else 0.0
        if cnt >= min_freq and rate >= min_success_rate:
            candidates.append((ng, cnt, rate))
    candidates.sort(key=lambda x: (-x[1], -x[2]))
    for ng, cnt, rate in candidates:
        if lib.num_programs() < lib.max_subs:
            lib.register(sequence=list(ng), param_dim=0, param_type='noop')
        else:
            break

def train(episodes, max_complex, max_steps, mcts_sims=50, ppo_update_interval=8, discovery_interval=200, device='cpu'):
    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)

    state_dim = Task.get_dimension_state() * 2
    n_atomic = len(Task.get_whole_actions())
    MAX_SUBS = 6
    MAX_PARAM_DIM = 2
    EMB = 32

    lib = ProgramLibrary(max_subs=MAX_SUBS, emb_dim=EMB, max_param_dim=MAX_PARAM_DIM, device=device)
    # register some parameterized subprograms: pid=1 move, pid=2 rotate
    lib.register(sequence=[], param_dim=2, param_type='move')
    lib.register(sequence=[], param_dim=1, param_type='rotate_angle')

    mind = Mind(state_dim=state_dim, max_subs=MAX_SUBS, n_atomic=n_atomic, max_param_dim=MAX_PARAM_DIM, emb_dim=EMB, hidden=256, device=device)
    trans_buf = TransitionBuffer(capacity=20000)
    mcts_buf = MctsTargetBuffer(capacity=5000)
    agent = AlphaNPIAgent(lib, mind, trans_buf, mcts_buf, mcts_simulations=mcts_sims, c_puct=1.0, device=device)
    trainer = Trainer(lib, mind, trans_buf, mcts_buf, device=device, gamma=0.99, lam=0.95,
                      ppo_epochs=4, mini_batch=64, clip_eps=0.2, lr=2e-4, mcts_supervision_coef=1.0)

    solved = 0
    start_time = time.time()
    for ep in range(1, episodes + 1):
        s0, g0, seq = Task.gen_task(max_complex=max_complex)
        succ, call_trace = agent.rollout_episode(s0, max_steps, use_mcts=True)
        solved += int(succ)

        if ep % ppo_update_interval == 0:
            trainer.ppo_update()
            trans_buf.clear()

        if ep % discovery_interval == 0:
            program_discovery(lib, trans_buf, min_freq=6, min_success_rate=0.7, ngram_max=4)
        if ep % 50 == 0:
            elapsed = time.time() - start_time
            print(f"ep={ep:04d}  solved_rate={solved/ep:.4f}  mcts_buf={len(mcts_buf):04d}  programs={lib.num_programs()}  elapsed={elapsed:5.1f}s")
    # final update
    trainer.ppo_update()
    print("Training finished. programs:", lib.num_programs())
    return lib, mind, trans_buf, mcts_buf, agent

def valid(agent, n, max_complex, max_steps):
    for i in range(n):
        s0, g0, seq = Task.gen_task(max_complex=max_complex)
        print("GroundTruth seq:", seq)
        succ, call_trace = agent.rollout_episode(s0, max_steps, use_mcts=True)
        print("Solved?", succ)
        # === 打印层次调用轨迹 ===
        print("[Program Call Trace]")
        for t, pid, stack in call_trace:
            indent = "  " * (len(stack) - 1)
            if t == 'call_sub':
                prog_name = self.lib.programs.get(pid, {}).get('name', f'sub_{pid}')
                print(f"{indent}CALL {prog_name}")
            elif t == 'return':
                print(f"{indent}RETURN")
            else:
                # 取原子操作名字
                if hasattr(Task.Action, "ATOMIC_ACTIONS"):
                    atomic_name = Task.Action.ATOMIC_ACTIONS[pid]
                else:
                    atomic_name = f'op_{pid}'
                print(f"{indent}ATOMIC {atomic_name}")
        print()

def main(max_complex=3, max_steps=12):
    lib, mind, trans_buf, mcts_buf, agent = train(episodes=1000, max_complex=max_complex, max_steps=max_steps, device='cpu', mcts_sims=30)
    valid(agent, n=3, max_complex=max_complex, max_steps=max_steps)

if __name__ == "__main__":
    main()

