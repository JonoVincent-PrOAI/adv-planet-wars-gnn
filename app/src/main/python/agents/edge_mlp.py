import math

import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from gym_utils.distributions import MaskedCategorical, SigmoidTransformedDistribution
from gym_utils.gym_wrapper import owner_one_hot_encoding


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PlanetWarsAgentEdgeMLP(nn.Module):
    """
    Edge MLP agent: computes a joint distribution over valid (source, target) planet pairs.

    For each valid (source, target) pair the edge MLP produces a scalar logit.
    Sampling from the resulting distribution chooses the pair in a single step
    rather than hierarchically.  The ratio action is then conditioned on the
    selected pair's node embeddings.

    The critic sums per-edge values over all non-self-loop edges.
    """

    def __init__(self, args, player_id=1, exploit=True):
        super().__init__()
        self.args = args
        self.player_id = player_id
        self.exploit = exploit
        self.num_planets = 30

        # Same preprocessing as MLP: node_feature_dim but we remove destination since we check edges
        self.per_node_input_dim = args.node_feature_dim + 3
        hidden_size = args.hidden_dim
        self.hidden_dim = hidden_size

        # Per-node feature extractor (shared by all heads)
        self.node_embedder = nn.Sequential(
            layer_init(nn.Linear(self.per_node_input_dim, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
        )

        # Edge actor: [src_emb || tgt_emb] -> scalar logit for each (source, target) pair
        self.edge_actor = nn.Sequential(
            layer_init(nn.Linear(2 * hidden_size, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, 1), std=0.01),
        )

        # No-op head: uses global mean-pooled features
        self.noop_actor = nn.Sequential(
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, 1), std=0.01),
        )

        # Edge critic: [src_emb || tgt_emb] -> scalar per-edge value; summed over all edges
        self.edge_critic = nn.Sequential(
            layer_init(nn.Linear(2 * hidden_size, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )

        # Ratio actor: [src_emb || tgt_emb] -> ratio in (0, 1)
        if args.discretized_ratio_bins == 0:
            self.ratio_actor_mean = nn.Sequential(
                layer_init(nn.Linear(2 * hidden_size, hidden_size)),
                nn.ReLU(),
                layer_init(nn.Linear(hidden_size, 1), std=0.01),
            )
            self.ratio_actor_logstd = nn.Parameter(torch.zeros(1))
        else:
            self.ratio_actor = nn.Sequential(
                layer_init(nn.Linear(2 * hidden_size, hidden_size)),
                nn.ReLU(),
                layer_init(
                    nn.Linear(
                        hidden_size,
                        args.discretized_ratio_bins - (0 if args.discretize_include_zero else 1),
                    ),
                    std=0.01,
                ),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess(self, x, player_id=None):
        """One-hot encode owners and destination; return (x_proc, planet_owners, transporter_owners)."""
        pid = player_id if player_id is not None else self.player_id
        planet_owners = x[:, :, 0]
        transporter_owners = x[:, :, 5]
        destination_planet = x[:, :, 6]
        x_proc = torch.cat(
            (
                owner_one_hot_encoding(planet_owners, pid),
                x[:, :, 1:5],
                owner_one_hot_encoding(transporter_owners, pid),
                x[:, :, 7:],
            ),
            dim=-1,
        )
        return x_proc, planet_owners, transporter_owners

    def _get_node_emb(self, x_proc):
        """Apply per-node MLP. x_proc: [B, N, F] -> [B, N, hidden]."""
        b, n, f = x_proc.shape
        return self.node_embedder(x_proc.reshape(b * n, f)).reshape(b, n, self.hidden_dim)

    def _get_edge_features(self, node_emb):
        """All-pairs edge features. node_emb: [B, N, H] -> [B, N, N, 2H]."""
        src = node_emb.unsqueeze(2).expand(-1, -1, self.num_planets, -1)
        tgt = node_emb.unsqueeze(1).expand(-1, self.num_planets, -1, -1)
        return torch.cat([src, tgt], dim=-1)

    def _build_pair_mask(self, source_mask, planet_owners, zero_growth_rate):
        """Build [B, N, N] boolean mask of valid (source, target) pairs.

        Valid iff source is an owned free planet, target satisfies target_mask,
        and source != target.
        """
        if self.args.target_mask == "all":
            target_mask = ~zero_growth_rate
        elif self.args.target_mask == "enemy":
            target_mask = planet_owners == 2
        elif self.args.target_mask == "not_self":
            target_mask = planet_owners != 1
        elif self.args.target_mask == "not_neutral":
            target_mask = planet_owners != 0
        else:
            target_mask = torch.ones_like(source_mask)

        # [B, N, N]: source valid along dim-1, target valid along dim-2
        pair_mask = source_mask.unsqueeze(2) & target_mask.unsqueeze(1)
        # Remove diagonal (cannot send to self)
        diag = torch.eye(self.num_planets, dtype=torch.bool, device=source_mask.device)
        pair_mask = pair_mask & ~diag.unsqueeze(0)
        return pair_mask

    def _build_joint_dist(self, node_emb, pair_mask, planet_owners, zero_growth_rate, temperature=1.0):
        """Build the joint (no-op + pair) masked categorical distribution.

        Returns (joint_probs, edge_logits, global_features) where edge_logits is
        [B, N, N] (un-masked, un-temperature-scaled raw logits for later use).
        """
        batch_size = node_emb.size(0)
        edge_features = self._get_edge_features(node_emb)          # [B, N, N, 2H]
        raw_edge_logits = self.edge_actor(edge_features).squeeze(-1)  # [B, N, N]

        global_features = node_emb.mean(dim=1)                       # [B, H]
        noop_logit = self.noop_actor(global_features)                 # [B, 1]

        flat_edge_logits = (raw_edge_logits / temperature).reshape(batch_size, -1)  # [B, N*N]
        flat_pair_mask = pair_mask.reshape(batch_size, -1)                           # [B, N*N]

        full_logits = torch.cat([noop_logit / temperature, flat_edge_logits], dim=1)  # [B, N*N+1]
        full_mask = torch.cat(
            [torch.ones(batch_size, 1, dtype=torch.bool, device=node_emb.device), flat_pair_mask],
            dim=1,
        )
        return MaskedCategorical(logits=full_logits, mask=full_mask), raw_edge_logits, global_features, edge_features

    @staticmethod
    def _encode_joint(source_action, target_action, num_planets):
        """Encode (source [1-indexed], target [0-indexed]) -> joint index (0 = no-op)."""
        is_noop = source_action == 0
        edge_idx = (source_action - 1).clamp(min=0) * num_planets + target_action
        return torch.where(is_noop, torch.zeros_like(edge_idx), edge_idx + 1)

    @staticmethod
    def _decode_joint(joint_action, num_planets):
        """Decode joint index -> (source [1-indexed], target [0-indexed])."""
        is_noop = joint_action == 0
        edge_idx = (joint_action - 1).clamp(min=0)
        source = torch.where(is_noop, torch.zeros_like(edge_idx), edge_idx // num_planets + 1)
        target = torch.where(is_noop, torch.zeros_like(edge_idx), edge_idx % num_planets)
        return source, target, is_noop

    # ------------------------------------------------------------------
    # Public interface (matches PlanetWarsAgentMLP)
    # ------------------------------------------------------------------

    def get_value(self, x):
        x_proc, _, _ = self._preprocess(x)
        node_emb = self._get_node_emb(x_proc)                        # [B, N, H]
        edge_features = self._get_edge_features(node_emb)             # [B, N, N, 2H]
        edge_vals = self.edge_critic(edge_features).squeeze(-1)        # [B, N, N]
        # Mask out self-loops before summing
        diag = torch.eye(self.num_planets, dtype=torch.bool, device=x.device).unsqueeze(0)
        value = edge_vals.masked_fill(diag, 0.0).sum(dim=[-1, -2]).unsqueeze(-1)  # [B, 1]
        return value

    def get_action_and_value(self, x, action=None):
        batch_size = x.size(0)
        num_planets = self.num_planets

        planet_owners = x[:, :, 0]
        transporter_owners = x[:, :, 5]
        zero_growth_rate = x[:, :, 2] == 0

        source_mask = torch.logical_and(planet_owners == 1, transporter_owners == 0)  # [B, N]

        x_proc, _, _ = self._preprocess(x)
        node_emb = self._get_node_emb(x_proc)                         # [B, N, H]

        pair_mask = self._build_pair_mask(source_mask, planet_owners, zero_growth_rate)  # [B, N, N]
        joint_probs, _, _, edge_features = self._build_joint_dist(node_emb, pair_mask, planet_owners, zero_growth_rate)

        # Critic: sum of all per-edge values
        edge_vals = self.edge_critic(edge_features).squeeze(-1)        # [B, N, N]
        diag = torch.eye(num_planets, dtype=torch.bool, device=x.device).unsqueeze(0)
        value = edge_vals.masked_fill(diag, 0.0).sum(dim=[-1, -2]).unsqueeze(-1)  # [B, 1]

        if action is None:
            joint_sample = joint_probs.sample()                        # [B]
            source_action, target_action, is_noop = self._decode_joint(joint_sample, num_planets)
            ratio_action = torch.zeros(batch_size, 1, dtype=torch.float, device=x.device)
        else:
            source_action = action[:, 0].long()
            target_action = action[:, 1].long()
            ratio_action = action[:, 2].unsqueeze(-1)
            is_noop = source_action == 0
            joint_sample = self._encode_joint(source_action, target_action, num_planets)

        joint_logprob = joint_probs.log_prob(joint_sample)            # [B]
        joint_entropy = joint_probs.entropy()                          # [B]

        ratio_logprob = torch.zeros(batch_size, device=x.device)
        ratio_entropy = torch.zeros(batch_size, device=x.device)

        valid_idx = ~is_noop
        if valid_idx.any():
            valid_batch = torch.where(valid_idx)[0]
            src_idx = source_action[valid_idx] - 1                    # 0-indexed source
            tgt_idx = target_action[valid_idx]
            src_feats = node_emb[valid_batch, src_idx]                 # [V, H]
            tgt_feats = node_emb[valid_batch, tgt_idx]                 # [V, H]
            ratio_input = torch.cat([src_feats, tgt_feats], dim=-1)   # [V, 2H]

            if self.args.discretized_ratio_bins == 0:
                ratio_mean = self.ratio_actor_mean(ratio_input)
                ratio_std = self.ratio_actor_logstd.exp().clamp(max=10.0)
                ratio_probs = SigmoidTransformedDistribution(ratio_mean, ratio_std)
                if action is None:
                    ratio_sample = ratio_probs.sample()
                    ratio_action[valid_idx] = ratio_sample
                    ratio_logprob[valid_idx] = ratio_probs.log_prob(ratio_sample).squeeze(-1)
                else:
                    ratio_logprob[valid_idx] = ratio_probs.log_prob(ratio_action[valid_idx]).squeeze(-1)
                ratio_entropy[valid_idx] = ratio_probs.entropy().squeeze(-1)
            else:
                ratio_probs = Categorical(logits=self.ratio_actor(ratio_input))
                if action is None:
                    ratio_bins = ratio_probs.sample().unsqueeze(-1)
                    ratio_logprob[valid_idx] = ratio_probs.log_prob(ratio_bins.squeeze(-1))
                    ratio_action[valid_idx] = (
                        ratio_bins + (0 if self.args.discretize_include_zero else 1)
                    ).float() / (self.args.discretized_ratio_bins - 1)
                else:
                    discrete_ratio = (
                        (ratio_action[valid_idx] * (self.args.discretized_ratio_bins - 1))
                        .round()
                        .long()
                        - (0 if self.args.discretize_include_zero else 1)
                    )
                    ratio_logprob[valid_idx] = ratio_probs.log_prob(discrete_ratio.squeeze(-1))
                ratio_entropy[valid_idx] = ratio_probs.entropy()

        action_out = torch.stack(
            [source_action.float(), target_action.float(), ratio_action.squeeze(-1)],
            dim=-1,
        )
        total_logprob = joint_logprob + ratio_logprob
        total_entropy = joint_entropy + ratio_entropy
        return action_out, total_logprob, total_entropy, value

    def get_action(self, x):
        """Get action at inference time (single observation, batch size 1)."""
        with torch.no_grad():
            num_planets = self.num_planets

            # Swap owners so player is always represented as player 1
            if self.player_id == 1:
                planet_owners = x[:, :, 0]
                transporter_owners = x[:, :, 5]
            else:
                planet_owners = (x[:, :, 0] * 2) % 3
                transporter_owners = (x[:, :, 5] * 2) % 3
            zero_growth_rate = x[:, :, 2] == 0
            destination_planet = x[:, :, 6]

            source_mask = torch.logical_and(planet_owners == 1, transporter_owners == 0)

            # Preprocess with owners already normalised to player-1 perspective
            x_proc = torch.cat(
                (
                    owner_one_hot_encoding(planet_owners, 1),
                    x[:, :, 1:5],
                    owner_one_hot_encoding(transporter_owners, 1),
                    # torch.nn.functional.one_hot(destination_planet.long(), num_classes=30),
                    x[:, :, 7:],
                ),
                dim=-1,
            )
            node_emb = self._get_node_emb(x_proc)                     # [1, N, H]

            pair_mask = self._build_pair_mask(source_mask, planet_owners, zero_growth_rate)
            joint_probs, _, _, _ = self._build_joint_dist(node_emb, pair_mask, planet_owners, zero_growth_rate)

            if self.exploit:
                joint_action = joint_probs.probs.argmax(dim=-1)        # [1]
            else:
                joint_action = joint_probs.sample()                    # [1]

            source_action, target_action, is_noop = self._decode_joint(joint_action, num_planets)
            ratio_action = torch.zeros(1, dtype=torch.float, device=x.device)

            if not is_noop.item():
                src_feats = node_emb[0, source_action[0] - 1].unsqueeze(0)   # [1, H]
                tgt_feats = node_emb[0, target_action[0]].unsqueeze(0)        # [1, H]
                ratio_input = torch.cat([src_feats, tgt_feats], dim=-1)       # [1, 2H]

                if self.args.discretized_ratio_bins == 0:
                    ratio_logits = self.ratio_actor_mean(ratio_input)
                    if self.exploit:
                        ratio_action = torch.sigmoid(ratio_logits).squeeze(-1)
                    else:
                        ratio_std = self.ratio_actor_logstd.exp().clamp(max=10.0)
                        ratio_action = SigmoidTransformedDistribution(ratio_logits, ratio_std).sample().squeeze(-1)
                else:
                    ratio_logits = self.ratio_actor(ratio_input)
                    if self.exploit:
                        ratio_action = (
                            torch.argmax(ratio_logits, dim=-1)
                            + (0 if self.args.discretize_include_zero else 1)
                        ).float()
                    else:
                        ratio_action = (
                            Categorical(logits=ratio_logits).sample()
                            + (0 if self.args.discretize_include_zero else 1)
                        ).float()
                    ratio_action = ratio_action / (self.args.discretized_ratio_bins - 1)

            action = torch.cat(
                [source_action.float(), target_action.float(), ratio_action.reshape(1)],
                dim=-1,
            )
            return action

    def get_action_samples(self, x, source_mask, num_samples=10, temperatures=None):
        """Sample (or exploit top-K) multiple candidate actions from one observation."""
        if temperatures is None:
            temperatures = {'source': 1.0, 'ratio': 1.0}
        with torch.no_grad():
            num_planets = self.num_planets
            if self.exploit:
                # Cannot take more unique pairs than valid (source, target) combinations
                num_samples = min(num_samples, num_planets * num_planets)

            if self.player_id == 1:
                planet_owners = x[:, :, 0]
                transporter_owners = x[:, :, 5]
            else:
                planet_owners = (x[:, :, 0] * 2) % 3
                transporter_owners = (x[:, :, 5] * 2) % 3
            zero_growth_rate = x[:, :, 2] == 0
            destination_planet = x[:, :, 6]

            x_proc = torch.cat(
                (
                    owner_one_hot_encoding(planet_owners, 1),
                    x[:, :, 1:5],
                    owner_one_hot_encoding(transporter_owners, 1),
                    # torch.nn.functional.one_hot(destination_planet.long(), num_classes=30),
                    x[:, :, 7:],
                ),
                dim=-1,
            )
            node_emb = self._get_node_emb(x_proc)                     # [1, N, H]

            pair_mask = self._build_pair_mask(source_mask, planet_owners, zero_growth_rate)
            joint_probs, _, _, _ = self._build_joint_dist(
                node_emb, pair_mask, planet_owners, zero_growth_rate,
                temperature=temperatures['source'],
            )

            if self.exploit:
                _, joint_actions = torch.topk(joint_probs.probs, num_samples, dim=-1)
                joint_actions = joint_actions.squeeze(0)               # [K]
            else:
                joint_actions = joint_probs.sample((num_samples,)).squeeze(-1)  # [K]

            source_actions, target_actions, is_noop = self._decode_joint(joint_actions, num_planets)
            ratio_actions = torch.zeros(num_samples, dtype=torch.float, device=x.device)

            valid_idx = ~is_noop
            if valid_idx.any():
                src_feats = node_emb[0, source_actions[valid_idx] - 1]    # [V, H]
                tgt_feats = node_emb[0, target_actions[valid_idx]]         # [V, H]
                ratio_input = torch.cat([src_feats, tgt_feats], dim=-1)   # [V, 2H]

                if self.args.discretized_ratio_bins == 0:
                    ratio_logits = self.ratio_actor_mean(ratio_input)
                    if self.exploit:
                        ratio_actions[valid_idx] = torch.sigmoid(ratio_logits).squeeze(-1)
                    else:
                        ratio_std = (
                            self.ratio_actor_logstd.exp().clamp(max=10.0)
                            * math.sqrt(temperatures['ratio'])
                        )
                        ratio_actions[valid_idx] = SigmoidTransformedDistribution(
                            ratio_logits, ratio_std
                        ).sample().squeeze(-1)
                else:
                    if self.exploit:
                        ratio_actions[valid_idx] = (
                            torch.argmax(self.ratio_actor(ratio_input), dim=-1)
                            + (0 if self.args.discretize_include_zero else 1)
                        ).float()
                    else:
                        ratio_actions[valid_idx] = (
                            Categorical(logits=self.ratio_actor(ratio_input)).sample()
                            + (0 if self.args.discretize_include_zero else 1)
                        ).float()
                    ratio_actions[valid_idx] = (
                        ratio_actions[valid_idx] / (self.args.discretized_ratio_bins - 1)
                    )

            return torch.stack(
                [source_actions.float(), target_actions.float(), ratio_actions],
                dim=-1,
            )

    def copy(self):
        new_agent = PlanetWarsAgentEdgeMLP(self.args, self.player_id)
        new_agent.load_state_dict(self.state_dict())
        return new_agent

    def copy_as_opponent(self):
        new_agent = PlanetWarsAgentEdgeMLP(self.args, self.player_id)
        new_agent.load_state_dict(self.state_dict())
        new_agent.player_id = 3 - self.player_id
        return new_agent
    

if __name__ == "__main__":
    from agents.ppo import Args
    import torch

    args = Args()
    args.node_feature_dim = 8
    args.hidden_dim = 64
    args.target_mask = 'not_self'
    args.discretized_ratio_bins = 0
    args.discretize_include_zero = False

    agent = PlanetWarsAgentEdgeMLP(args)
    print(f'Parameters: {sum(p.numel() for p in agent.parameters()):,}')

    B, N = 4, 30
    x = torch.zeros(B, N, args.node_feature_dim)
    x[:, :10, 0] = 1   # player 1 planets
    x[:, 10:20, 0] = 2 # player 2 planets
    x[:, :, 2] = 1.0   # growth rate
    x[:, :, 5] = 0     # no transporter
    x[:, :, 6] = 0     # destination = planet 0 (valid index)

    v = agent.get_value(x)
    print(f'get_value: {v.shape}')

    action, lp, ent, val = agent.get_action_and_value(x)
    print(f'action: {action.shape}, logprob: {lp.shape}, entropy: {ent.shape}, value: {val.shape}')
    print(f'sample actions:\n{action}')

    _, lp2, _, _ = agent.get_action_and_value(x, action)
    print(f'reeval logprob: {lp2.shape}')

    a = agent.get_action(x[:1])
    print(f'get_action: {a}')

    # Test discretized ratio
    args2 = Args()
    args2.node_feature_dim = 8
    args2.hidden_dim = 64
    args2.target_mask = 'not_self'
    args2.discretized_ratio_bins = 5
    args2.discretize_include_zero = True
    agent2 = PlanetWarsAgentEdgeMLP(args2)
    action2, lp2, ent2, val2 = agent2.get_action_and_value(x)
    print(f'discrete: action={action2.shape}, lp={lp2.shape}')
    _, lp2r, _, _ = agent2.get_action_and_value(x, action2)

    # Test get_action_samples
    source_mask = (x[0, :, 0] == 1) & (x[0, :, 5] == 0)
    source_mask = source_mask.unsqueeze(0)
    samples = agent.get_action_samples(x[:1], source_mask, num_samples=5)
    print(f'get_action_samples: {samples.shape}')

    print('All checks passed!')
