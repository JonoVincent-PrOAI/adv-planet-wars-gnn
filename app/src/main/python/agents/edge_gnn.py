import math

import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.nn import Sequential as PyGSequential
from torch_geometric.nn.norm import MeanSubtractionNorm
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch
from typing import Tuple, Union, List
from gym_utils.distributions import MaskedCategorical, SigmoidTransformedDistribution


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def layer_init_gat(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.lin_l.weight, std)
    torch.nn.init.constant_(layer.lin_l.bias, bias_const)
    torch.nn.init.orthogonal_(layer.lin_r.weight, std)
    torch.nn.init.constant_(layer.lin_r.bias, bias_const)
    if layer.lin_edge is not None:
        torch.nn.init.orthogonal_(layer.lin_edge.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    torch.nn.init.orthogonal_(layer.att, std)
    return layer


class PlanetWarsAgentEdgeGNN(nn.Module):
    """
    GNN agent that replaces hierarchical (source then target) action selection with a
    joint distribution over valid (source, target) planet pairs, identical in spirit to
    PlanetWarsAgentEdgeMLP but using GATv2Conv node embeddings.

    The critic is unchanged from PlanetWarsAgentGNN: global mean-pool -> MLP.
    Variable-size batches (different num_planets per observation) are handled via
    to_dense_batch padding, exactly as in the base GNN agent.
    """

    def __init__(self, args, player_id=1, exploit=True):
        super().__init__()
        self.args = args
        self.player_id = player_id
        self.exploit = exploit

        self.node_feature_dim = args.node_feature_dim
        self.hidden_dim = args.hidden_dim if hasattr(args, 'hidden_dim') else 128

        # GNN layers — identical to PlanetWarsAgentGNN
        self.a_gnn = PyGSequential('x, edge_index, edge_attr, batch', [
            (layer_init_gat(GATv2Conv(self.node_feature_dim, self.hidden_dim // 4, heads=4, concat=True, edge_dim=5, add_self_loops=False)), 'x, edge_index, edge_attr -> x'),
            (MeanSubtractionNorm(), 'x, batch -> x'),
            nn.ReLU(),
            (layer_init_gat(GATv2Conv(self.hidden_dim, self.hidden_dim // 4, heads=4, concat=True, edge_dim=5, add_self_loops=False)), 'x, edge_index, edge_attr -> x'),
            nn.ReLU(),
        ])
        if not args.shared_gnn:
            self.v_gnn = PyGSequential('x, edge_index, edge_attr, batch', [
                (layer_init_gat(GATv2Conv(self.node_feature_dim, self.hidden_dim // 4, heads=4, concat=True, edge_dim=5, add_self_loops=False)), 'x, edge_index, edge_attr -> x'),
                (MeanSubtractionNorm(), 'x, batch -> x'),
                nn.ReLU(),
                (layer_init_gat(GATv2Conv(self.hidden_dim, self.hidden_dim // 4, heads=4, concat=True, edge_dim=5, add_self_loops=False)), 'x, edge_index, edge_attr -> x'),
                nn.ReLU(),
            ])

        self.global_pool = global_mean_pool

        # Critic — unchanged from PlanetWarsAgentGNN
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.hidden_dim, self.hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(self.hidden_dim, 1), std=1.0),
        )

        # Edge actor: [src_emb || tgt_emb] -> scalar logit per (source, target) pair
        self.edge_actor = nn.Sequential(
            layer_init(nn.Linear(2 * self.hidden_dim, self.hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(self.hidden_dim, 1), std=0.01),
        )

        # No-op actor: global node embedding -> scalar logit
        self.noop_actor = nn.Sequential(
            layer_init(nn.Linear(self.hidden_dim, self.hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(self.hidden_dim, 1), std=0.01),
        )

        # Ratio actor: [src_emb || tgt_emb] -> ratio in (0, 1)
        if args.discretized_ratio_bins == 0:
            self.ratio_actor_mean = nn.Sequential(
                layer_init(nn.Linear(3 * self.hidden_dim, self.hidden_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(self.hidden_dim, 1), std=0.01),
            )
            self.ratio_actor_logstd = nn.Parameter(torch.zeros(1))
        else:
            self.ratio_actor = nn.Sequential(
                layer_init(nn.Linear(3 * self.hidden_dim, self.hidden_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(self.hidden_dim, args.discretized_ratio_bins - (0 if args.discretize_include_zero else 1)), std=0.01),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def forward_gnn(self, x, edge_index, edge_attr, batch=None, batch_size=None):
        h = self.a_gnn(x, edge_index, edge_attr, batch)
        if batch is None:
            global_features = torch.mean(h, dim=0, keepdim=True)
        else:
            global_features = self.global_pool(h, batch, batch_size)
        return h, global_features

    def forward_value_gnn(self, x, edge_index, edge_attr, batch=None, batch_size=None):
        if self.args.shared_gnn:
            h = self.a_gnn(x, edge_index, edge_attr, batch)
        else:
            h = self.v_gnn(x, edge_index, edge_attr, batch)
        if batch is None:
            global_features = torch.mean(h, dim=0, keepdim=True)
        else:
            global_features = self.global_pool(h, batch, batch_size)
        return self.critic(global_features)

    @staticmethod
    def _get_edge_features(dense_node_emb):
        """All-pairs concat: [B, N, H] -> [B, N, N, 2H]."""
        N = dense_node_emb.size(1)
        src = dense_node_emb.unsqueeze(2).expand(-1, -1, N, -1)
        tgt = dense_node_emb.unsqueeze(1).expand(-1, N, -1, -1)
        return torch.cat([src, tgt], dim=-1)

    def _build_pair_mask(self, dense_source_mask, dense_owners, node_padding_mask):
        """
        Build [B, N, N] valid-pair boolean mask.

        dense_source_mask  : [B, N] bool — which nodes may act as source
        dense_owners       : [B, N] long — owner class (0=neutral,1=own,2=enemy)
        node_padding_mask  : [B, N] bool — True for real nodes, False for padding
        """
        if self.args.target_mask == "all":
            target_mask = node_padding_mask
        elif self.args.target_mask == "enemy":
            target_mask = (dense_owners == 2) & node_padding_mask
        elif self.args.target_mask == "not_self":
            target_mask = (dense_owners != 1) & node_padding_mask
        elif self.args.target_mask == "not_neutral":
            target_mask = (dense_owners != 0) & node_padding_mask
        else:
            target_mask = node_padding_mask

        source_mask = dense_source_mask & node_padding_mask
        pair_mask = source_mask.unsqueeze(2) & target_mask.unsqueeze(1)  # [B, N, N]
        N = node_padding_mask.size(1)
        diag = torch.eye(N, dtype=torch.bool, device=node_padding_mask.device)
        return pair_mask & ~diag.unsqueeze(0)

    @staticmethod
    def _encode_joint(source_action, target_action, max_N):
        """(source [1-indexed, 0=noop], target [0-indexed]) -> joint index (0=noop)."""
        is_noop = source_action == 0
        edge_idx = (source_action - 1).clamp(min=0) * max_N + target_action
        return torch.where(is_noop, torch.zeros_like(edge_idx), edge_idx + 1)

    @staticmethod
    def _decode_joint(joint_action, max_N):
        """Joint index -> (source [1-indexed], target [0-indexed], is_noop)."""
        is_noop = joint_action == 0
        edge_idx = (joint_action - 1).clamp(min=0)
        source = torch.where(is_noop, torch.zeros_like(edge_idx), edge_idx // max_N + 1)
        target = torch.where(is_noop, torch.zeros_like(edge_idx), edge_idx % max_N)
        return source, target, is_noop

    def _ratio_forward(self, ratio_input, ratio_action_buf, ratio_logprob_buf, ratio_entropy_buf, valid_idx, action):
        """Compute ratio distribution and fill ratio_action/logprob/entropy buffers."""
        if self.args.discretized_ratio_bins == 0:
            ratio_mean = self.ratio_actor_mean(ratio_input)
            ratio_std = self.ratio_actor_logstd.exp().clamp(max=10.0)
            ratio_probs = SigmoidTransformedDistribution(ratio_mean, ratio_std)
            if action is None:
                ratio_sample = ratio_probs.sample()
                ratio_action_buf[valid_idx] = ratio_sample
                ratio_logprob_buf[valid_idx] = ratio_probs.log_prob(ratio_sample).squeeze(-1)
            else:
                ratio_logprob_buf[valid_idx] = ratio_probs.log_prob(ratio_action_buf[valid_idx]).squeeze(-1)
            ratio_entropy_buf[valid_idx] = ratio_probs.entropy().squeeze(-1)
        else:
            ratio_probs = Categorical(logits=self.ratio_actor(ratio_input))
            if action is None:
                ratio_bins = ratio_probs.sample().unsqueeze(-1)
                ratio_logprob_buf[valid_idx] = ratio_probs.log_prob(ratio_bins.squeeze(-1))
                ratio_action_buf[valid_idx] = (
                    ratio_bins + (0 if self.args.discretize_include_zero else 1)
                ).float() / (self.args.discretized_ratio_bins - 1)
            else:
                discrete_ratio = (
                    (ratio_action_buf[valid_idx] * (self.args.discretized_ratio_bins - 1))
                    .round().long()
                    - (0 if self.args.discretize_include_zero else 1)
                )
                ratio_logprob_buf[valid_idx] = ratio_probs.log_prob(discrete_ratio.squeeze(-1))
            ratio_entropy_buf[valid_idx] = ratio_probs.entropy()

    # ------------------------------------------------------------------
    # Public interface (mirrors PlanetWarsAgentGNN)
    # ------------------------------------------------------------------

    def get_value(self, obs, batch_size=None):
        data, batch = obs, obs.batch
        return self.forward_value_gnn(data.x, data.edge_index, data.edge_attr, batch, batch_size)

    def get_action_and_value(self, obs, source_mask=None, action=None):
        if isinstance(obs, (list, tuple)):
            obs = Batch.from_data_list(obs)
        elif isinstance(obs, Data) and not isinstance(obs, Batch):
            obs = Batch.from_data_list([obs])

        data, batch = obs, obs.batch
        batch_size = obs.batch_size

        # GNN forward pass
        node_features, global_features = self.forward_gnn(
            data.x, data.edge_index, data.edge_attr, batch, batch_size)
        if self.args.shared_gnn:
            value = self.critic(global_features)
        else:
            value = self.forward_value_gnn(
                data.x, data.edge_index, data.edge_attr, batch, batch_size)

        # Densify node embeddings to [B, max_N, H]
        dense_node_features, node_padding_mask = to_dense_batch(
            node_features, batch, fill_value=0.0)
        # Source mask (per-node, no no-op slot)
        dense_source_mask, _ = to_dense_batch(obs.source_mask, batch, fill_value=False)
        # Owner class for target mask
        flat_owners = data.x[:, :3].argmax(dim=-1)
        dense_owners, _ = to_dense_batch(flat_owners, batch, fill_value=0)

        max_N = dense_node_features.size(1)

        # Joint distribution over (noop + all pairs)
        pair_mask = self._build_pair_mask(dense_source_mask, dense_owners, node_padding_mask)
        edge_features = self._get_edge_features(dense_node_features)        # [B, max_N, max_N, 2H]
        raw_edge_logits = self.edge_actor(edge_features).squeeze(-1)           # [B, max_N, max_N]
        noop_logit = self.noop_actor(global_features)                          # [B, 1]

        full_logits = torch.cat([noop_logit, raw_edge_logits.reshape(batch_size, -1)], dim=1)
        full_mask = torch.cat([
            torch.ones(batch_size, 1, dtype=torch.bool, device=node_features.device),
            pair_mask.reshape(batch_size, -1),
        ], dim=1)
        joint_probs = MaskedCategorical(logits=full_logits, mask=full_mask)

        ratio_logprob = torch.zeros(batch_size, device=node_features.device)
        ratio_entropy = torch.zeros(batch_size, device=node_features.device)

        if action is None:
            joint_sample = joint_probs.sample()
            source_action, target_action, is_noop = self._decode_joint(joint_sample, max_N)
            ratio_action = torch.zeros(batch_size, 1, dtype=torch.float, device=node_features.device)
        else:
            source_action = action[:, 0].long()
            target_action = action[:, 1].long()
            ratio_action = action[:, 2].unsqueeze(-1)
            is_noop = source_action == 0
            joint_sample = self._encode_joint(source_action, target_action, max_N)

        joint_logprob = joint_probs.log_prob(joint_sample)
        joint_entropy = joint_probs.entropy()

        valid_idx = ~is_noop
        if valid_idx.any():
            valid_batch = torch.where(valid_idx)[0]
            src_idx = source_action[valid_idx] - 1   # 0-indexed
            tgt_idx = target_action[valid_idx]
            src_feats = dense_node_features[valid_batch, src_idx]   # [V, H]
            tgt_feats = dense_node_features[valid_batch, tgt_idx]   # [V, H]
            ratio_input = torch.cat([src_feats, tgt_feats, global_features[valid_batch]], dim=-1)  # [V, 2H]
            self._ratio_forward(ratio_input, ratio_action, ratio_logprob, ratio_entropy, valid_idx, action)

        action_out = torch.stack(
            [source_action.float(), target_action.float(), ratio_action.squeeze(-1)], dim=-1)
        return action_out, joint_logprob + ratio_logprob, joint_entropy + ratio_entropy, value

    def get_action(self, data):
        """Single Data object inference."""
        with torch.no_grad():
            N = data.x.size(0)
            node_features, global_features = self.forward_gnn(
                data.x, data.edge_index, data.edge_attr)

            node_emb = node_features.unsqueeze(0)                              # [1, N, H]
            source_mask = data.source_mask.unsqueeze(0)                        # [1, N]
            owners = data.x[:, :3].argmax(dim=-1).unsqueeze(0)                # [1, N]
            padding_mask = torch.ones(1, N, dtype=torch.bool, device=data.x.device)

            pair_mask = self._build_pair_mask(source_mask, owners, padding_mask)
            edge_features = self._get_edge_features(node_emb)
            raw_edge_logits = self.edge_actor(edge_features).squeeze(-1)       # [1, N, N]
            noop_logit = self.noop_actor(global_features)

            full_logits = torch.cat([noop_logit, raw_edge_logits.reshape(1, -1)], dim=1)
            full_mask = torch.cat([
                torch.ones(1, 1, dtype=torch.bool, device=data.x.device),
                pair_mask.reshape(1, -1),
            ], dim=1)
            joint_probs = MaskedCategorical(logits=full_logits, mask=full_mask)

            if self.exploit:
                joint_action = joint_probs.probs.argmax(dim=-1)
            else:
                joint_action = joint_probs.sample()

            source_action, target_action, is_noop = self._decode_joint(joint_action, N)
            ratio_action = torch.zeros(1, dtype=torch.float, device=data.x.device)

            if not is_noop.item():
                src_feats = node_features[source_action[0] - 1].unsqueeze(0)  # [1, H]
                tgt_feats = node_features[target_action[0]].unsqueeze(0)       # [1, H]
                ratio_input = torch.cat([src_feats, tgt_feats, global_features[0].unsqueeze(0)], dim=-1)  # [1, 2H]

                if self.args.discretized_ratio_bins == 0:
                    ratio_logits = self.ratio_actor_mean(ratio_input)
                    if self.exploit:
                        ratio_action = torch.sigmoid(ratio_logits).squeeze(-1)
                    else:
                        ratio_std = self.ratio_actor_logstd.exp().clamp(max=10.0)
                        ratio_action = SigmoidTransformedDistribution(
                            ratio_logits, ratio_std).sample().squeeze(-1)
                else:
                    ratio_logits_raw = self.ratio_actor(ratio_input)
                    if self.exploit:
                        ratio_action = (torch.argmax(ratio_logits_raw, dim=-1)
                                        + (0 if self.args.discretize_include_zero else 1)).float()
                    else:
                        ratio_action = (Categorical(logits=ratio_logits_raw).sample()
                                        + (0 if self.args.discretize_include_zero else 1)).float()
                    ratio_action = ratio_action / (self.args.discretized_ratio_bins - 1)

            return torch.cat(
                [source_action.float(), target_action.float(), ratio_action.reshape(1)], dim=-1)

    def get_action_samples(self, data, num_samples=10, temperatures=None):
        """Sample / exploit top-K actions from a single Data observation."""
        if temperatures is None:
            temperatures = {'source': 1.0, 'ratio': 1.0}
        with torch.no_grad():
            N = data.x.size(0)
            if self.exploit:
                num_samples = min(num_samples, N * N)

            node_features, global_features = self.forward_gnn(
                data.x, data.edge_index, data.edge_attr)

            node_emb = node_features.unsqueeze(0)
            source_mask = data.source_mask.unsqueeze(0)
            owners = data.x[:, :3].argmax(dim=-1).unsqueeze(0)
            padding_mask = torch.ones(1, N, dtype=torch.bool, device=data.x.device)

            pair_mask = self._build_pair_mask(source_mask, owners, padding_mask)
            edge_features = self._get_edge_features(node_emb)
            raw_edge_logits = (self.edge_actor(edge_features).squeeze(-1)
                               / temperatures['source'])                         # [1, N, N]
            noop_logit = self.noop_actor(global_features) / temperatures['source']

            full_logits = torch.cat([noop_logit, raw_edge_logits.reshape(1, -1)], dim=1)
            full_mask = torch.cat([
                torch.ones(1, 1, dtype=torch.bool, device=data.x.device),
                pair_mask.reshape(1, -1),
            ], dim=1)
            joint_probs = MaskedCategorical(logits=full_logits, mask=full_mask)

            if self.exploit:
                _, joint_actions = torch.topk(joint_probs.probs, num_samples, dim=-1)
                joint_actions = joint_actions.squeeze(0)                         # [K]
            else:
                joint_actions = joint_probs.sample((num_samples,)).squeeze(-1)   # [K]

            source_actions, target_actions, is_noop = self._decode_joint(joint_actions, N)
            ratio_actions = torch.zeros(num_samples, dtype=torch.float, device=data.x.device)

            valid_idx = ~is_noop
            if valid_idx.any():
                src_feats = node_features[source_actions[valid_idx] - 1]        # [V, H]
                tgt_feats = node_features[target_actions[valid_idx]]             # [V, H]
                ratio_input = torch.cat([src_feats, tgt_feats, global_features[0].unsqueeze(0)], dim=-1)  # [V, 2H]

                if self.args.discretized_ratio_bins == 0:
                    ratio_logits = self.ratio_actor_mean(ratio_input)
                    if self.exploit:
                        ratio_actions[valid_idx] = torch.sigmoid(ratio_logits).squeeze(-1)
                    else:
                        ratio_std = (self.ratio_actor_logstd.exp().clamp(max=10.0)
                                     * math.sqrt(temperatures['ratio']))
                        ratio_actions[valid_idx] = SigmoidTransformedDistribution(
                            ratio_logits, ratio_std).sample().squeeze(-1)
                else:
                    if self.exploit:
                        ratio_actions[valid_idx] = (
                            torch.argmax(self.ratio_actor(ratio_input), dim=-1)
                            + (0 if self.args.discretize_include_zero else 1)).float()
                    else:
                        ratio_actions[valid_idx] = (
                            Categorical(logits=self.ratio_actor(ratio_input)).sample()
                            + (0 if self.args.discretize_include_zero else 1)).float()
                    ratio_actions[valid_idx] /= (self.args.discretized_ratio_bins - 1)

            return torch.stack(
                [source_actions.float(), target_actions.float(), ratio_actions], dim=-1)

    def copy(self):
        new_agent = PlanetWarsAgentEdgeGNN(self.args, self.player_id)
        new_agent.load_state_dict(self.state_dict())
        return new_agent

    def copy_as_opponent(self):
        new_agent = PlanetWarsAgentEdgeGNN(self.args, self.player_id)
        new_agent.load_state_dict(self.state_dict())
        new_agent.player_id = 3 - self.player_id
        return new_agent


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from agents.ppo import Args
    from torch_geometric.data import Data, Batch
    from torch_geometric.utils import add_self_loops

    args = Args()
    args.node_feature_dim = 9   # 3 owner one-hot + 6 other
    args.hidden_dim = 64
    args.target_mask = 'not_self'
    args.discretized_ratio_bins = 0
    args.discretize_include_zero = False
    args.shared_gnn = True

    agent = PlanetWarsAgentEdgeGNN(args)
    print(f'Parameters: {sum(p.numel() for p in agent.parameters()):,}')

    def make_obs(num_planets):
        x = torch.zeros(num_planets, args.node_feature_dim)
        half = num_planets // 2
        x[:half, 1] = 1.0   # player 1 (one-hot class 1)
        x[half:, 2] = 1.0   # player 2 (one-hot class 2)
        x[:, 4] = 1.0        # growth rate
        source_mask = x[:, 1].bool()
        # Complete graph edges (no self-loops)
        rows, cols = zip(*[(i, j) for i in range(num_planets) for j in range(num_planets) if i != j])
        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        edge_attr = torch.zeros(edge_index.size(1), 5)
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr, fill_value='mean')
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, source_mask=source_mask)

    # Batch with different planet counts
    obs_list = [make_obs(10), make_obs(20), make_obs(15), make_obs(10)]
    batch = Batch.from_data_list(obs_list)

    v = agent.get_value(batch)
    print(f'get_value: {v.shape}')   # [4, 1]

    action, lp, ent, val = agent.get_action_and_value(batch)
    print(f'action: {action.shape}, logprob: {lp.shape}, entropy: {ent.shape}, value: {val.shape}')
    print(f'sample actions:\n{action}')

    _, lp2, _, _ = agent.get_action_and_value(batch, action)
    print(f'reeval logprob: {lp2.shape}')

    single = make_obs(20)
    a = agent.get_action(single)
    print(f'get_action (20 planets): {a}')

    samples = agent.get_action_samples(single, num_samples=5)
    print(f'get_action_samples: {samples.shape}')   # [5, 3]

    # Discretized ratio
    args2 = Args()
    args2.node_feature_dim = 9
    args2.hidden_dim = 64
    args2.target_mask = 'not_self'
    args2.discretized_ratio_bins = 5
    args2.discretize_include_zero = True
    args2.shared_gnn = True
    agent2 = PlanetWarsAgentEdgeGNN(args2)
    action2, lp2, ent2, val2 = agent2.get_action_and_value(batch)
    print(f'discrete: action={action2.shape}, lp={lp2.shape}')
    _, lp2r, _, _ = agent2.get_action_and_value(batch, action2)
    print(f'discrete reeval: {lp2r.shape}')

    print('All checks passed!')
