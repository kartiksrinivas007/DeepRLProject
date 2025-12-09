import torch
import torch.nn.functional as F
from model import ReinFormer
from lamb import Lamb



class ReinFormerTrainer:
    def __init__(
        self, 
        state_dim,
        act_dim,
        device,
        variant
    ):
        super().__init__()
                
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.device = device
        self.grad_norm = variant["grad_norm"]

        self.model = ReinFormer(
            state_dim=state_dim,
            act_dim=act_dim,
            n_blocks=variant["n_blocks"],
            h_dim=variant["embed_dim"],
            context_len=variant["context_len"],
            n_heads=variant["n_heads"],
            drop_p=variant["dropout_p"],
            init_temperature=variant["init_temperature"],
            target_entropy=variant.get("target_entropy", -self.act_dim),
        ).to(self.device)

        self.optimizer = Lamb(
            self.model.parameters(),
            lr=variant["lr"],
            weight_decay=variant["wd"],
            eps=1e-8,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda steps: min((steps+1)/variant["warmup_steps"], 1)
        )

        self.tau = variant["tau"]
        self.context_len=variant["context_len"]


        self.log_temperature_optimizer = torch.optim.Adam(
            [self.model.log_temperature],
            lr=1e-4,
            betas=[0.9, 0.999],
        )

    
    def train_step(
        self,
        timesteps,
        states,
        actions,
        returns_to_go,
        rewards,
        traj_mask,
        beta_rl: float = 0.0,
        adv_scale: float = 1.0,
    ):
        self.model.train()
        # data to gpu ------------------------------------------------
        timesteps = timesteps.to(self.device)      # B x T
        states = states.to(self.device)            # B x T x state_dim
        actions = actions.to(self.device)          # B x T x act_dim
        returns_to_go = returns_to_go.to(self.device).unsqueeze(
            dim=-1
        )                                          # B x T x 1
        
        rewards = rewards.to(self.device).unsqueeze(
            dim=-1
        )                                          # B x T x 1
        traj_mask = traj_mask.to(self.device)      # B x T

        # model forward ----------------------------------------------
        (
            returns_to_go_preds,
            actions_dist_preds,
            _,
        ) = self.model.forward(
            timesteps=timesteps,
            states=states,
            actions=actions,
            returns_to_go=returns_to_go,
        )

        returns_to_go_target = torch.clone(returns_to_go).view(
            -1, 1
        )[
            traj_mask.view(-1,) > 0
        ]
        returns_to_go_preds = returns_to_go_preds.view(-1, 1)[
            traj_mask.view(-1,) > 0
        ]

        # returns_to_go_loss -----------------------------------------
        norm = returns_to_go_target.abs().mean()
        u = (returns_to_go_target - returns_to_go_preds) / norm
        returns_to_go_loss = torch.mean(
            torch.abs(
                self.tau - (u < 0).float()
            ) * u ** 2
        )
        # action_loss ------------------------------------------------
        actions_target = torch.clone(actions)
        log_prob_all = actions_dist_preds.log_prob(actions_target).sum(axis=2)
        log_likelihood = log_prob_all[
            traj_mask > 0
        ].mean()
        entropy = actions_dist_preds.entropy().sum(axis=2).mean()
        action_loss = -(log_likelihood + self.model.temperature().detach() * entropy)

        # Advantage-weighted RL term (AWAC-style) -------------------
        adv_norm = None
        weights = None
        rl_loss = torch.tensor(0.0, device=self.device)
        if beta_rl > 0.0:
            # (a) raw advantage from RTG predictions
            adv_raw = returns_to_go_target - returns_to_go_preds.detach()
            # (b) normalize advantage
            adv_norm = (adv_raw - adv_raw.mean()) / (adv_raw.std() + 1e-8)
            # (c) scale by a temperature λ (adv_scale) and clip
            scaled = (adv_norm / max(adv_scale, 1e-6)).clamp(min=-10, max=10)
            # (d) exponentiate and clip importance weights
            weights = torch.exp(scaled).clamp(max=20.0)
            rl_loss = -(
                weights.view(-1) * log_prob_all.view(-1)[traj_mask.view(-1,) > 0]
            ).mean()

        loss = returns_to_go_loss + action_loss + beta_rl * rl_loss

        # optimization -----------------------------------------------
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients and then measure post-clipping norm.
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            self.grad_norm
        )
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        self.optimizer.step()

        self.log_temperature_optimizer.zero_grad()
        temperature_loss = (
            self.model.temperature() * (entropy - self.model.target_entropy).detach()
        )
        temperature_loss.backward()
        self.log_temperature_optimizer.step()

        self.scheduler.step()

        # Diagnostics for logging/analysis.
        adv_mean = adv_std = adv_max = adv_min = None
        weights_mean = weights_max = None
        if adv_norm is not None and adv_norm.numel() > 0:
            flat_adv = adv_norm.view(-1)
            adv_mean = flat_adv.mean().detach().cpu().item()
            adv_std = flat_adv.std(unbiased=False).detach().cpu().item()
            adv_max = flat_adv.max().detach().cpu().item()
            adv_min = flat_adv.min().detach().cpu().item()
        if weights is not None and weights.numel() > 0:
            flat_w = weights.view(-1)
            weights_mean = flat_w.mean().detach().cpu().item()
            weights_max = flat_w.max().detach().cpu().item()

        return (
            loss.detach().cpu().item(),
            {
                "returns_to_go_loss": returns_to_go_loss.detach().cpu().item(),
                "action_loss": action_loss.detach().cpu().item(),
                "temperature_loss": temperature_loss.detach().cpu().item(),
                "entropy": entropy.detach().cpu().item(),
                "temperature": self.model.temperature().detach().cpu().item(),
                "grad_norm": float(total_norm),
                "rl_loss": rl_loss.detach().cpu().item(),
                "adv_mean": adv_mean,
                "adv_std": adv_std,
                "adv_max": adv_max,
                "adv_min": adv_min,
                "weights_mean": weights_mean,
                "weights_max": weights_max,
            },
        )
