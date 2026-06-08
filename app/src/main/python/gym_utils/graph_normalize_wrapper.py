import gymnasium as gym
import numpy as np
import torch
from torch_geometric.data import Data
from gymnasium.wrappers.utils import RunningMeanStd


class NormalizeGraphObservation(gym.vector.VectorWrapper):
    """
    Gymnasium wrapper that normalizes node features and edge features of
    ``torch_geometric.data.Data`` observations using per-feature running
    mean and standard deviation.

    Each node (resp. edge) is treated as an independent data-point, so the
    statistics are accumulated across all nodes (resp. edges) seen over the
    lifetime of the wrapper regardless of how many planets the current map has.

    Only normalizes the last 2 features of node and edge feature vectors, which are the only continuous features.
    The other features are either binary or categorical (one-hot encoded) and do not require normalization.

    Parameters
    ----------
    env : gym.Env
        The wrapped environment.  Its ``step`` / ``reset`` must return
        ``torch_geometric.data.Data`` observations.
    node_feature_dim : int
        Dimensionality of node feature vectors (``data.x``).
    edge_feature_dim : int or None
        Dimensionality of edge feature vectors (``data.edge_attr``).
        Pass ``None`` to skip edge-feature normalisation.
    clip : float
        Clip normalised values to ``[-clip, clip]``.  Set to ``np.inf`` to
        disable clipping.
    epsilon : float
        Small constant added to std to avoid division by zero (already
        included inside ``RunningMeanStd.std``; this parameter is kept for
        explicit use in the clip path).
    training : bool
        When ``True`` the running statistics are updated on every observation.
        Set to ``False`` during evaluation to freeze the statistics.
    """

    def __init__(
        self,
        env: gym.Env,
        clip: float = 10.0,
        training: bool = True,
    ):
        super().__init__(env)
        self.node_rms = RunningMeanStd(shape=(2,))
        self.edge_rms = RunningMeanStd(shape=(2,))
        self.clip = clip
        self.training = training

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._normalize(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._normalize(obs), reward, terminated, truncated, info

    def _normalize(self, data: list[Data]) -> list[Data]:
        """Return the Data object with normalised node / edge features. Modified in-place."""
        # --- node features ---
        for d in data:
            x = d.x[:,-2:]  # (N, 2)
            
            x_np = x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
            if self.training:
                self.node_rms.update(x_np)
            x_norm = (x_np - self.node_rms.mean) / self.node_rms.var**0.5
            x_norm = np.clip(x_norm, -self.clip, self.clip)
            x_out = torch.tensor(x_norm, dtype=torch.float32)


            # --- edge features ---
            edge_attr = d.edge_attr[:,-2:]  # (E, 2) 
            if edge_attr is not None and self.edge_rms is not None:
                ea_np = edge_attr.numpy() if isinstance(edge_attr, torch.Tensor) else np.asarray(edge_attr)
                if self.training:
                    self.edge_rms.update(ea_np)
                ea_norm = (ea_np - self.edge_rms.mean) / self.edge_rms.var**0.5
                ea_norm = np.clip(ea_norm, -self.clip, self.clip)
                ea_out = torch.tensor(ea_norm, dtype=torch.float32)
            else:
                ea_out = edge_attr
            
            d.x[:,-2:]=x_out
            d.edge_attr[:,-2:]=ea_out
        return data

    def get_stats(self) -> dict:
        """Return a dict with the current running statistics."""
        stats = {
            'node_mean': self.node_rms.mean.copy(),
            'node_var': self.node_rms.var.copy(),
            'node_count': self.node_rms.count,
        }
        if self.edge_rms is not None:
            stats.update({
                'edge_mean': self.edge_rms.mean.copy(),
                'edge_var': self.edge_rms.var.copy(),
                'edge_count': self.edge_rms.count,
            })
        return stats

    def load_stats(self, stats: dict):
        """Restore running statistics from a previously saved dict."""
        self.node_rms.mean = stats['node_mean'].copy()
        self.node_rms.var = stats['node_var'].copy()
        self.node_rms.count = stats['node_count']
        if self.edge_rms is not None and 'edge_mean' in stats:
            self.edge_rms.mean = stats['edge_mean'].copy()
            self.edge_rms.var = stats['edge_var'].copy()
            self.edge_rms.count = stats['edge_count']

class NormalizeMLPObservation(gym.vector.VectorWrapper):
    """
    Gymnasium wrapper that normalizes MLP observations using per-feature running
    mean and standard deviation.
    Each row is treated as an independent data-point, so the statistics are accumulated across all
    rows seen over the lifetime of the wrapper regardless of how many planets the current map has.

    Only normalizes the continuous features of the observation.

    Parameters
    ----------
    env : gym.Env
        The wrapped environment.  Its ``step`` / ``reset`` must return
        1D numpy array observations.
    clip : float
        Clip normalised values to ``[-clip, clip]``.  Set to ``np.inf`` to
        disable clipping.
    epsilon : float
        Small constant added to std to avoid division by zero (already
        included inside ``RunningMeanStd.std``; this parameter is kept for
        explicit use in the clip path).
    training : bool
        When ``True`` the running statistics are updated on every observation.
        Set to ``False`` during evaluation to freeze the statistics.
    """

    def __init__(
        self,
        env: gym.Env,
        clip: float = 10.0,
        training: bool = True,
    ):
        super().__init__(env)
        self.planet_rms = RunningMeanStd(shape=(4,))
        self.transporter_rms = RunningMeanStd(shape=(2,))
        self.clip = clip
        self.training = training

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._normalize(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._normalize(obs), reward, terminated, truncated, info

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        """Return the normalised observation."""
        planet_features = obs[:,:,1:5]
        transporter_features = obs[:,:,7:]
        if self.training:
            self.planet_rms.update(planet_features)
            self.transporter_rms.update(transporter_features)
        planet_norm = (planet_features - self.planet_rms.mean) / self.planet_rms.var**0.5
        transporter_norm = (transporter_features - self.transporter_rms.mean) / self.transporter_rms.var**0.5
        planet_norm = np.clip(planet_norm, -self.clip, self.clip)
        transporter_norm = np.clip(transporter_norm, -self.clip, self.clip)
        obs[:,:,1:5] = planet_norm
        obs[:,:,7:] = transporter_norm
        return obs

    def get_stats(self) -> dict:
        """Return a dict with the current running statistics."""
        stats = {
            'planet_mean': self.planet_rms.mean.copy(),
            'planet_var': self.planet_rms.var.copy(),
            'planet_count': self.planet_rms.count,
            'transporter_mean': self.transporter_rms.mean.copy(),
            'transporter_var': self.transporter_rms.var.copy(),
            'transporter_count': self.transporter_rms.count
        }
        return stats
    
    def load_stats(self, stats: dict):
        """Restore running statistics from a previously saved dict."""
        self.planet_rms.mean = stats['planet_mean'].copy()
        self.planet_rms.var = stats['planet_var'].copy()
        self.planet_rms.count = stats['planet_count']
        self.transporter_rms.mean = stats['transporter_mean'].copy()
        self.transporter_rms.var = stats['transporter_var'].copy()
        self.transporter_rms.count = stats['transporter_count']

        
                                      
