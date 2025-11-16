# td3_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict
from buffer import Buffer
from policies import Actor, Critic


class TD3Agent:
    """
    Twin Delayed DDPG (TD3) agent that matches PPO's interface pattern.
    Simplified to remove unnecessary abstraction layers.
    """

    def __init__(
        self,
        env_info,
        offline_buffer: Buffer,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        batch_size=128,
        update_every=1,
        policy_noise=0.2,
        noise_clip=0.5,
        delay=2,
        bc_regularization_weight: float = 0.0,
        cql_alpha: float = 0.0,
        cql_n_actions: int = 4,
        cql_temp: float = 1.0,
        device="cpu",
    ):
        self.device = torch.device(device)

        # Environment info
        self.obs_dim = env_info["obs_dim"]
        self.act_dim = env_info["act_dim"]
        self.act_low = torch.as_tensor(
            env_info["act_low"], dtype=torch.float32, device=self.device
        )
        self.act_high = torch.as_tensor(
            env_info["act_high"], dtype=torch.float32, device=self.device
        )

        # TD3 hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.update_every = update_every
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = delay  # Standard TD3 delay
        self.bc_regularization_weight = bc_regularization_weight
        self.cql_alpha = cql_alpha
        self.cql_n_actions = cql_n_actions
        self.cql_temp = cql_temp

        self.actor = Actor(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            act_low=self.act_low,
            act_high=self.act_high,
            hidden=(128, 128),
            state_independent_std=True,  # Deterministic actor
        ).to(self.device)

        self.target_actor = Actor(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            act_low=self.act_low,
            act_high=self.act_high,
            hidden=(128, 128),
            state_independent_std=True,
        ).to(self.device)

        # Twin critics with targets
        self.critic1 = Critic(self.obs_dim, self.act_dim, hidden=(128, 128)).to(
            self.device
        )
        self.critic2 = Critic(self.obs_dim, self.act_dim, hidden=(128, 128)).to(
            self.device
        )
        self.target_critic1 = Critic(self.obs_dim, self.act_dim, hidden=(128, 128)).to(
            self.device
        )
        self.target_critic2 = Critic(self.obs_dim, self.act_dim, hidden=(128, 128)).to(
            self.device
        )

        # Initialize target networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # Optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr
        )

        self._offline_buffer = offline_buffer

        # Training state
        self._update_count = 0

    def act(self, obs, deterministic=True):
        """Return action info dict matching PPO's interface"""
        with torch.no_grad():
            obs_t = torch.as_tensor(
                obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            dist = self.actor(obs_t)
            action = dist.mean_action  # TD3 uses deterministic policy

            # Clamp to environment bounds
            action = torch.clamp(action, self.act_low, self.act_high)

            return {"action": action.squeeze(0).cpu().numpy()}

    def step(self) -> Dict[str, float]:
        """
        Add transition to buffer and perform updates when ready.
        Matches PPO's step interface.
        """

        # Perform TD3 updates
        return self._perform_update()

    def _perform_update(self) -> Dict[str, float]:
        """Perform TD3 updates and return stats"""
        all_stats = []

        # Perform updates based on update_every
        num_updates = max(1, self.update_every)

        for _ in range(num_updates):
            # Sample batch from buffer
            batch = self._offline_buffer.sample(self.batch_size)

            # Perform one TD3 update step
            self._update_count += 1
            do_actor_update = (self._update_count % self.policy_delay) == 0
            stats = self._td3_update_step(batch, do_actor_update)
            all_stats.append(stats)

        # Average stats across updates
        if all_stats:
            return {k: np.mean([s[k] for s in all_stats]) for k in all_stats[0].keys()}
        else:
            return {}

    def _td3_update_step(self, batch, do_actor_update: bool) -> Dict[str, float]:
        """Single TD3 update step"""
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]

        # Compute target Q-values with target policy smoothing
        with torch.no_grad():
            next_dist = self.target_actor(next_obs)
            next_actions = next_dist.mean_action

            # Target policy smoothing
            if self.policy_noise > 0.0:
                noise = torch.randn_like(next_actions) * self.policy_noise
                noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
                next_actions = next_actions + noise

            next_actions = torch.clamp(next_actions, self.act_low, self.act_high)

            target_q1 = self.target_critic1(next_obs, next_actions)
            target_q2 = self.target_critic2(next_obs, next_actions)
            target_q = torch.min(target_q1, target_q2)

            target_q = rewards + (1.0 - dones) * self.gamma * target_q

        # Update critics
        current_q1 = self.critic1(obs, actions)
        current_q2 = self.critic2(obs, actions)

        critic_loss = nn.functional.mse_loss(
            current_q1, target_q
        ) + nn.functional.mse_loss(current_q2, target_q)

        if self.cql_alpha > 0.0:
            cql_loss, cql_stats = self._get_cql_loss(batch)
            critic_loss += self.cql_alpha * cql_loss
        else:
            cql_stats = {}

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Delayed policy updates
        actor_loss = torch.zeros((), device=self.device)
        ddpg_policy_loss = torch.zeros((), device=self.device)
        bc_loss = torch.zeros((), device=self.device)
        if do_actor_update:
            # Update actor
            actions_pi = self.actor(obs).mean_action
            ddpg_policy_loss = -self.critic1(obs, actions_pi).mean()
            actor_loss = ddpg_policy_loss

            if self.bc_regularization_weight > 0:
                # ------ Problem 1.2: BC regularization loss ------
                # If the python interpreter reaches this line, it will open PDB.
                # You can then inspect variables and step through the code.
                bc_loss = torch.mean(torch.pow(actions_pi - actions, 2).sum(dim=-1))
                actor_loss += bc_loss * self.bc_regularization_weight
                ### BEGIN STUDENT SOLUTION - 1.2 ###
                ### END STUDENT SOLUTION - 1.2 ###

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            # Soft update all target networks
            self._soft_update(self.actor, self.target_actor)
            self._soft_update(self.critic1, self.target_critic1)
            self._soft_update(self.critic2, self.target_critic2)

        # Return stats in format expected by runner
        stats = {
            "actor_loss": float(actor_loss.item()),
            "ddpg_policy_loss": float(ddpg_policy_loss.item()),
            "bc_regularization_loss": float(bc_loss.item()),
            "bc_regularization_weight": self.bc_regularization_weight,
            "critic1_loss": float(nn.functional.mse_loss(current_q1, target_q).item()),
            "critic2_loss": float(nn.functional.mse_loss(current_q2, target_q).item()),
            "alpha": 0.0,  # TD3 doesn't use temperature
            "q1": float(current_q1.mean().item()),
            "q2": float(current_q2.mean().item()),
        }
        stats.update(cql_stats)
        return stats

    def _get_cql_loss(self, batch):
        """Returns the CQL loss and stats for logging"""
        obs = batch["obs"]
        next_obs = batch["next_obs"]
        batch_size = obs.shape[0]
        
        actions = batch["actions"] # this is the in-dist Q

        # Use these variable names so that the plots pick up on the values.
        cql_loss = torch.zeros((), device=self.device)
        current_action_q1_values = torch.zeros(
            (batch_size, self.cql_n_actions), device=self.device
        )
        random_action_q1_values = torch.zeros(
            (batch_size, self.cql_n_actions), device=self.device
        )
        q_pred = torch.zeros((batch_size,), device=self.device)

        # ------ Problem 2.1: CQL loss ------

        # Steps to compute CQL loss:
        # 1. Compute Q-values for the in-distribution actions (i.e., actions from the offline dataset).
        # 2. Sample random actions (uniformly from the action space).
        # 3. Also sample actions using the current policy from both the current and next observations.
        # 4. Compute Q-values for the sampled actions (in-distribution, and (random, current, next)).
        # 5. Compute the CQL loss using torch.logsumexp.
        # Note: elf.cql_n_actions indicates how many random actions to sample for each element in the batch.

        ### BEGIN STUDENT SOLUTION - 2.1 ###
        n_rand = self.cql_n_actions
        temp = self.cql_temp

        q1_in = self.critic1(obs, actions) # (batch,)
        q2_in = self.critic2(obs, actions) # (batch,)
        q_in_min = torch.min(q1_in, q2_in) 

        # Sample random actions uniformly in action space
        # shape: (batch, n_rand, act_dim)
        rand_actions = torch.rand((batch_size, self.cql_n_actions, self.act_dim), device=self.device)
        rand_actions = rand_actions * (self.act_high - self.act_low) + self.act_low

        # Get current policy actions (for obs and next_obs)
        # Detach them so the critic update does not update the actor.
        with torch.no_grad():
            current_policy_actions = self.actor(obs).mean_action  # (batch, act_dim)
            next_policy_actions = self.actor(next_obs).mean_action  # (batch, act_dim)

        # Build the full sampled action pool per state:
        # concatenate random actions + current policy action + next policy action
        # result shape: (batch, n_rand + 2, act_dim)
        current_policy_actions_exp = current_policy_actions.unsqueeze(1)  # (batch,1,act_dim)
        next_policy_actions_exp = next_policy_actions.unsqueeze(1)  # (batch,1,act_dim)
        sampled_actions = torch.cat(
            [rand_actions, current_policy_actions_exp, next_policy_actions_exp], dim=1
        )
        n_total = sampled_actions.shape[1]  # n_rand + 2

        # Evaluate Q for all sampled actions (vectorized)
        # Expand obs to match sampled actions
        obs_expanded = obs.unsqueeze(1).expand(-1, n_total, -1)  # (batch, n_total, obs_dim)
        # Flatten to feed through critic: (batch * n_total, act_dim)
        obs_flat = obs_expanded.reshape(batch_size * n_total, -1)
        acts_flat = sampled_actions.reshape(batch_size * n_total, -1)

        # Critic evaluations (flattened) then reshape back to (batch, n_total)
        q1_all = self.critic1(obs_flat, acts_flat).view(batch_size, n_total)
        q2_all = self.critic2(obs_flat, acts_flat).view(batch_size, n_total)

        # split q1_all / q2_all into random vs current vs next for logging if desired
        q1_rand = q1_all[:, :n_rand]
        q1_current = q1_all[:, n_rand] 

        # For the conservative loss, compute logsumexp over the full pool per state.
        # Use temperature as: L = (logsumexp(Q / temp) * temp) - Q(data)
        logsumexp_q1 = torch.logsumexp(q1_all / temp, dim=1) * temp
        logsumexp_q2 = torch.logsumexp(q2_all / temp, dim=1) * temp      

        cql1 = (logsumexp_q1 - q1_in).mean()
        cql2 = (logsumexp_q2 - q2_in).mean()

        cql_loss = cql1 + cql2

        # For logging, compute useful scalars
        current_action_q1_values = q1_current.detach() if q1_current is not None else torch.zeros((batch_size,), device=self.device)
        random_action_q1_values = q1_rand.detach() if q1_rand is not None else torch.zeros((batch_size, n_rand), device=self.device)
        ### END STUDENT SOLUTION - 2.1 ###

        return cql_loss, {
            "cql_loss": cql_loss.item(),
            "cql_current_actions": current_action_q1_values.mean().item(),
            "cql_random_actions": random_action_q1_values.mean().item(),
            "in_distribution_q_pred": q_in_min.mean().item(),
        }

    def _soft_update(self, local_model, target_model):
        """Soft update target network parameters using Polyak averaging"""
        with torch.no_grad():
            for target_param, local_param in zip(
                target_model.parameters(), local_model.parameters()
            ):
                target_param.data.copy_(
                    self.tau * local_param.data + (1.0 - self.tau) * target_param.data
                )
