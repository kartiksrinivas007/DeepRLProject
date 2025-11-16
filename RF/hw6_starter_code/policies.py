# policies.py
import torch
import torch.nn as nn
from typing import Sequence, Tuple
from torch.distributions import Categorical, Normal


# --------------------------- helpers ---------------------------


def mlp(sizes: Sequence[int], activation=nn.ReLU, out_activation=None) -> nn.Sequential:
    """
    Simple MLP: sizes like [in, h1, h2, ..., out].
    """
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
        elif out_activation is not None:
            layers.append(out_activation())
    seq = nn.Sequential(*layers)
    for m in seq.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, 1.0)
            nn.init.zeros_(m.bias)
    return seq


# -------------------- shared tanh-squashed distribution --------------------


class _TanhDiagGaussian:
    """
    Lightweight distribution wrapper for a diagonal Gaussian in pre-tanh space.

    Exposes:
      - sample()    : non-reparameterized sample (good for PPO env stepping)
      - rsample()   : reparameterized sample (required for SAC actor update)
      - log_prob(a) : log π(a|s) with tanh + scale correction; returns [B]
      - entropy()   : entropy of base Gaussian (sum over dims); returns [B]
      - mean_action : deterministic action tanh(μ) mapped to env bounds
    """

    def __init__(
        self,
        mu: torch.Tensor,
        log_std: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor,
    ):
        """
        Args:
            mu:      [B, A]
            log_std: [B, A] or [A]
            scale:   [A]  (maps [-1,1] to env bounds)
            bias:    [A]
        """
        self.mu = mu
        self.log_std = log_std
        self.std = log_std.exp()
        self.scale = scale
        self.bias = bias
        self._base = torch.distributions.Normal(
            self.mu, self.std
        )  # broadcast over [B, A]

    # --- sampling ---

    def sample(self) -> torch.Tensor:
        """Non-reparameterized sample (no grads through noise). Returns [B, A]."""
        u = self._base.sample()
        return torch.tanh(u) * self.scale + self.bias

    def rsample(self) -> torch.Tensor:
        """Reparameterized sample (pathwise grads). Returns [B, A]."""
        u = self._base.rsample()
        return torch.tanh(u) * self.scale + self.bias

    # --- densities ---

    def log_prob(self, a: torch.Tensor) -> torch.Tensor:
        """
        Log π(a|s) with tanh change-of-variables and affine logdet.
        Accepts actions in env bounds; returns [B].
        """
        if a.ndim == 1:  # [A] -> [1, A]
            a = a.unsqueeze(0)

        # Map action back to pre-tanh space: y in (-1,1), u = atanh(y)
        y = (a - self.bias) / (self.scale + 1e-8)
        y = torch.clamp(y, -0.999999, 0.999999)
        u = 0.5 * (torch.log1p(y) - torch.log1p(-y))  # atanh(y)

        logp_u = self._base.log_prob(u).sum(dim=-1)  # sum over action dims

        # Jacobian of tanh: diag(1 - tanh(u)^2)
        correction = torch.log(1 - torch.tanh(u).pow(2) + 1e-8).sum(dim=-1)

        # Affine scale (bias adds no volume)
        scale_logdet = torch.log(self.scale + 1e-8).sum(dim=-1)

        return logp_u - correction - scale_logdet

    def entropy(self) -> torch.Tensor:
        """
        Proxy entropy: entropy of the base Gaussian (sum over dims). Returns [B].
        (True tanh entropy has no simple closed form; this proxy is standard in PPO.)
        """
        return self._base.entropy().sum(dim=-1)

    # --- eval convenience ---

    @property
    def mean_action(self) -> torch.Tensor:
        """Deterministic action for evaluation, [B, A]."""
        return torch.tanh(self.mu) * self.scale + self.bias


# ------------------------------- Actor -------------------------------


class Actor(nn.Module):
    """
    Unified tanh-squashed Gaussian policy for continuous control.

    Works for both PPO and SAC:
      - PPO:    use dist.sample(), dist.log_prob(a), dist.entropy(), dist.mean_action
      - SAC:    use dist.rsample(), dist.log_prob(a)

    Args:
      obs_dim, act_dim: dimensions
      act_low, act_high: per-dim bounds (array-like)
      hidden: MLP hidden sizes
      state_independent_std: if True, use a learned global log_std vector (PPO-friendly);
                             if False, predict log_std with a head (SAC-friendly)
      log_std_bounds: clamp range for numerical stability
      body_activation: nonlinearity for MLP trunk
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        act_low,
        act_high,
        hidden: Tuple[int, ...] = (256, 256),
        state_independent_std: bool = False,
        log_std_bounds: Tuple[float, float] = (-20.0, 2.0),
        body_activation=nn.ReLU,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.log_std_min, self.log_std_max = log_std_bounds
        self.state_independent_std = state_independent_std

        # shared trunk
        self.body = mlp([obs_dim, *hidden], activation=body_activation)

        # mean head
        self.mu = nn.Linear(hidden[-1], act_dim)

        # std parameterization
        if state_independent_std:
            self.log_std_param = nn.Parameter(torch.zeros(act_dim))
        else:
            self.log_std_head = nn.Linear(hidden[-1], act_dim)

        # action bounds
        low_t = torch.as_tensor(act_low, dtype=torch.float32)
        high_t = torch.as_tensor(act_high, dtype=torch.float32)
        self.register_buffer("scale", (high_t - low_t) / 2.0)
        self.register_buffer("bias", (high_t + low_t) / 2.0)

        # init heads
        nn.init.orthogonal_(self.mu.weight, 1.0)
        nn.init.zeros_(self.mu.bias)
        if not state_independent_std:
            nn.init.orthogonal_(self.log_std_head.weight, 1.0)
            nn.init.zeros_(self.log_std_head.bias)

    def forward(self, obs: torch.Tensor) -> _TanhDiagGaussian:
        """
        Returns a distribution-like object: _TanhDiagGaussian
        """
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        feats = self.body(obs)
        mu = self.mu(feats)
        if self.state_independent_std:
            log_std = self.log_std_param.expand_as(mu)
        else:
            log_std = self.log_std_head(feats)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return _TanhDiagGaussian(mu, log_std, self.scale, self.bias)


# ------------------------------- Critic -------------------------------


class Critic(nn.Module):
    """
    Q-network: predicts Q(s, a). Used by off-policy algorithms like SAC/TD3.
    """

    def __init__(
        self, obs_dim: int, act_dim: int, hidden: Tuple[int, ...] = (256, 256)
    ):
        super().__init__()
        self.q = mlp([obs_dim + act_dim, *hidden, 1], activation=nn.ReLU)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        if act.ndim == 1:
            act = act.unsqueeze(0)
        x = torch.cat([obs, act], dim=-1)
        return self.q(x).squeeze(-1)


# ----------------------------- ActorCritic -----------------------------


class ActorCritic(nn.Module):
    """
    PPO-style module that reuses the **same actor pass** and adds a value head V(s).

    Forward:
      obs -> (dist, value)

    Notes:
      - This mirrors your preference: the actor pass (tanh-Gaussian) is primary;
        the value is an additional head on the same trunk.
      - For PPO, you typically set state_independent_std=True for stability.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        act_low,
        act_high,
        hidden: Tuple[int, ...] = (64, 64),
        state_independent_std: bool = True,
        body_activation=nn.Tanh,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)

        # shared trunk
        self.body = mlp([obs_dim, *hidden], activation=body_activation)

        # actor heads (same as `Actor`)
        self.mu = nn.Linear(hidden[-1], act_dim)
        self.state_independent_std = state_independent_std
        if state_independent_std:
            # self.log_std_param = nn.Parameter(torch.zeros(act_dim))
            self.log_std_param = nn.Parameter(
                torch.full((act_dim,), -0.5)
            )  # std ≈ 0.61
        else:
            self.log_std_head = nn.Linear(hidden[-1], act_dim)
        self.log_std_min, self.log_std_max = (-20.0, 2.0)

        # value head
        self.v = nn.Linear(hidden[-1], 1)

        # action bounds
        low_t = torch.as_tensor(act_low, dtype=torch.float32)
        high_t = torch.as_tensor(act_high, dtype=torch.float32)
        self.register_buffer("scale", (high_t - low_t) / 2.0)
        self.register_buffer("bias", (high_t + low_t) / 2.0)

        # # init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, 1.0)
                nn.init.zeros_(m.bias)

        # for name, m in self.named_modules():
        #     if isinstance(m, nn.Linear):
        #         if name.endswith(".mu"):
        #             nn.init.orthogonal_(m.weight, 0.01)
        #             nn.init.zeros_(m.bias)
        #         elif name.endswith(".v"):
        #             nn.init.orthogonal_(m.weight, 1.0)
        #             nn.init.zeros_(m.bias)
        #         else:
        #             nn.init.orthogonal_(m.weight, 1.414)  # sqrt(2) for ReLU/Tanh
        #             nn.init.zeros_(m.bias)

        # After creating your ActorCritic
        # for name, module in self.named_modules():
        #     if isinstance(module, nn.Linear):
        #         if 'mu' in name:  # action head
        #             nn.init.orthogonal_(module.weight, 0.01)
        #         else:  # body and value head
        #             nn.init.orthogonal_(module.weight, 1.414)  # sqrt(2)
        #         nn.init.zeros_(module.bias)

    def forward(self, obs: torch.Tensor):
        """
        Returns:
            dist: _TanhDiagGaussian
            value: Tensor[B]
        """
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        feats = self.body(obs)

        # actor bits
        mu = self.mu(feats)
        if self.state_independent_std:
            log_std = self.log_std_param.expand_as(mu)
        else:
            log_std = self.log_std_head(feats)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        dist = _TanhDiagGaussian(mu, log_std, self.scale, self.bias)

        # value
        value = self.v(feats).squeeze(-1)
        return dist, value


class ActorCriticOld(nn.Module):
    """
    Diagonal Gaussian + tanh squashing + affine to action bounds.
    """

    def __init__(self, obs_dim, act_dim, act_low, act_high, hidden=(64, 64)):
        super().__init__()
        self.body = mlp([obs_dim, *hidden], activation=nn.Tanh, out_activation=nn.Tanh)
        self.mu = nn.Linear(hidden[-1], act_dim)
        self.v = nn.Linear(hidden[-1], 1)

        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)

        # Initialize log_std to smaller values for better exploration
        self.log_std = nn.Parameter(-0.5 * torch.ones(act_dim))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, 1.0)
                nn.init.zeros_(m.bias)

        act_low = torch.as_tensor(act_low, dtype=torch.float32)
        act_high = torch.as_tensor(act_high, dtype=torch.float32)
        self.register_buffer("scale", (act_high - act_low) / 2.0)
        self.register_buffer("bias", (act_high + act_low) / 2.0)

    def forward(self, obs):
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        x = self.body(obs)
        mu = self.mu(x)
        std = torch.clamp(self.log_std.exp(), min=1e-3, max=1.0)  # Clamp std
        value = self.v(x).squeeze(-1)

        class TanhDiagGaussian:
            def __init__(self, mu, std, scale, bias):
                self.base = Normal(mu, std)
                self.scale, self.bias = scale, bias

            def sample(self):
                u = self.base.rsample()
                a = torch.tanh(u) * self.scale + self.bias
                return a, u

            def log_prob(self, a, u=None):
                if a.ndim == 1:  # [A] -> [1, A]
                    a = a.unsqueeze(0)

                # # Map action back to pre-tanh space: y in (-1,1), u = atanh(y)
                # y = (a - self.bias) / (self.scale + 1e-8)
                # y = torch.clamp(y, -0.999999, 0.999999)
                # u = 0.5 * (torch.log1p(y) - torch.log1p(-y))  # atanh(y)
                if u is None:
                    # Inverse transform: a -> u
                    y = (a - self.bias) / self.scale
                    y = torch.clamp(y, -0.999999, 0.999999)
                    u = torch.atanh(y)  # Correct inverse tanh

                # Base log prob
                lp = self.base.log_prob(u).sum(-1)

                # Jacobian correction for tanh transform
                log_det_jacobian = torch.log(1 - torch.tanh(u).pow(2) + 1e-8).sum(-1)

                # Scale correction
                log_scale = torch.log(self.scale + 1e-8).sum()

                return lp - log_det_jacobian - log_scale

            def entropy(self):
                # Approximate entropy (exact computation is complex for tanh-Normal)
                return self.base.entropy().sum(-1)

            @property
            def mean_action(self):
                return torch.tanh(self.base.loc) * self.scale + self.bias

        dist = TanhDiagGaussian(mu, std, self.scale, self.bias)
        return dist, value
