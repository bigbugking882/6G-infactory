"""
MAPPO policy for the warehouse scenario.

Decentralised actors (one shared parameterised head applied to every AGV node embedding)
with a centralised critic over the pooled heterogeneous-graph state. The GNN is part of the
policy and is trained end-to-end: during each PPO update the graph is re-encoded so that
gradients flow through the encoder as well as the actor/critic heads.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from .hgnn import HeteroGNN


class HGNNPolicy(nn.Module):
    def __init__(self, n_actions, hidden=64, gnn_layers=2, n_links=3):
        super().__init__()
        self.gnn = HeteroGNN(hidden, gnn_layers, n_links)
        self.actor = nn.Sequential(
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1),
        )

    def forward(self, graph):
        agv_emb, g = self.gnn(graph)
        logits = self.actor(agv_emb)
        value = self.critic(g)
        return logits, value


class MAPPO:
    def __init__(self, cfg, device='cpu'):
        self.cfg = cfg
        self.device = device
        self.policy = HGNNPolicy(cfg.n_actions, cfg.hidden, cfg.gnn_layers, cfg.n_links).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)
        self.gamma = cfg.gamma
        self.lam = cfg.lam
        self.eps_clip = cfg.eps_clip
        self.ppo_epochs = cfg.ppo_epochs
        self.n_agents = cfg.n_agents
        self.entropy_coef = cfg.entropy_coef

    @torch.no_grad()
    def act(self, graph, greedy=False):
        logits, value = self.policy(graph)
        dist = Categorical(logits=logits)
        if greedy:
            actions = logits.argmax(dim=-1)
        else:
            actions = dist.sample()
        logprobs = dist.log_prob(actions)
        return actions.cpu().numpy(), logprobs, value.squeeze(-1)

    def update(self, graphs, actions, logprobs, values, rewards):
        """One PPO update over a full episode.

        graphs: list of graph dicts (one per step)
        actions: list of (N,) long tensors
        logprobs: list of (N,) tensors
        values: list of scalar tensors (centralised critic)
        rewards: list of (N,) numpy arrays (per-agent rewards)
        """
        n_steps = len(graphs)
        N = self.n_agents

        rewards = torch.tensor(np.stack(rewards), dtype=torch.float32, device=self.device)

        # per-agent discounted returns-to-go: (n_steps, N)
        returns = torch.zeros_like(rewards)
        R = torch.zeros(N, device=self.device)
        for s in reversed(range(n_steps)):
            R = rewards[s] + self.gamma * R
            returns[s] = R

        # centralised critic is trained to predict the mean per-step return
        mean_returns = returns.mean(dim=1)  # (n_steps,)
        old_values = torch.stack(values, dim=0).to(self.device)  # (n_steps,)

        # GAE per-agent advantages w.r.t. the centralised value baseline
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(N, device=self.device)
        for s in reversed(range(n_steps)):
            next_value = old_values[s + 1] if s + 1 < n_steps else 0.0
            delta = rewards[s] + self.gamma * next_value - old_values[s]
            gae = delta + self.gamma * self.lam * gae
            advantages[s] = gae
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.ppo_epochs):
            actor_loss_sum = 0.0
            critic_loss_sum = 0.0
            entropy_sum = 0.0
            self.optimizer.zero_grad()
            for s in range(n_steps):
                logits, value = self.policy(graphs[s])
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(actions[s].to(self.device))
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logp - logprobs[s].to(self.device))
                adv = advantages[s]  # (N,)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * adv
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = ((value.squeeze(-1) - mean_returns[s]) ** 2).mean()

                loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy
                loss.backward()

                actor_loss_sum += actor_loss.item()
                critic_loss_sum += critic_loss.item()
                entropy_sum += entropy.item()

            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

        return {
            'actor_loss': actor_loss_sum / n_steps,
            'critic_loss': critic_loss_sum / n_steps,
            'entropy': entropy_sum / n_steps,
        }

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
        self.policy.eval()
