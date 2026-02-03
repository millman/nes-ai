# Current Loss + Gradient Flow Overview

This note reflects the defaults in `jepa_world_model_trainer.py` (`LossWeights`, `TrainConfig`) and `jepa_world_model/model_config.py`.
It is intended as a quick "what is enabled" and "where gradients go" snapshot, aligned with the algebra notes (see `action_algebra_v5.md`, `overview_v3.md`).

## Enabled losses (defaults in `LossWeights`)

Core:
- jepa = 1.0

Recon + pixel:
- recon_multi_box = 1.0
- pixel_delta_multi_box = 1.0

Hidden state alignment:
- h2z = 1.0
- z2h = 1.0

Action-algebra (h):
- action_delta_h = 1.0
- rollout_kstep_h = 1.0
- rollout_kstep_delta_h = 1.0

Action-algebra / odometry (p):
- action_delta_dp = 1.0
- additivity_dp = 1.0
- rollout_kstep_p = 1.0
- scale_dp = 1.0
- geometry_rank_p = 1.0

Inverse dynamics:
- inverse_dynamics_h = 1.0

Disabled by default (weight = 0.0):
- sigreg, recon, recon_patch, recon_multi_gauss
- pixel_delta
- inverse_dynamics_z, inverse_dynamics_p, inverse_dynamics_dp
- action_delta_z
- additivity_h
- rollout_kstep_z, rollout_recon_z, rollout_recon_multi_box_z, rollout_recon_delta_z
- rollout_recon_multi_box_delta_z, rollout_project_z

## Key detach / gate settings (defaults)

- `TrainConfig.detach_z_from_h_and_p = True`
  - z is stop-grad when feeding predictor and h/p rollout losses.
- `TrainConfig.detach_decoder = False`
  - recon losses backprop into encoder + decoder.
- `ModelConfig.pose_delta_detach_h = True`
  - p/pose losses do not backprop into h (predictor).

## High-level flow (default detach points)

```
x_t ----> Encoder ----> z_t ---------------------> Decoder ----> x_recon
                      \                         (recon_multi_box, pixel_delta_multi_box)
                       \-- stop-grad? --------> Predictor(h_t, a_t) ---> h_{t+1}
                                                |                         |
                                                |                         +--> inverse_dynamics_h
                                                |
                                                +--> h_to_z ----> z_hat_{t+1} (JEPA target is stop-grad z_{t+1})
                                                |
                                                +--> h_action_delta_projector (action_delta_h, rollout_kstep_h)
                                                |
                                                +--> p_action_delta_projector ---> pose_t
                                                     (pose_delta_detach_h=True)
                                                     (action_delta_dp, additivity_dp, rollout_kstep_p, scale_dp)
                                                     +--> geometry_rank_p
```

Legend:
- `stop-grad?` is controlled by `detach_z_from_h_and_p`. Default: yes (z is detached).
- Pose rollout stops gradient into h by default (`pose_delta_detach_h=True`).

## Gradient flow by loss (default config)

| Loss | Target (stop-grad?) | Primary parameters updated | Notes |
| --- | --- | --- | --- |
| jepa | target `z_{t+1}` is detached | predictor, h_to_z | Encoder gets grads only if `detach_z_from_h_and_p=False`. |
| recon_multi_box | target `x_t` | encoder + decoder | `detach_decoder=True` would isolate decoder only. |
| pixel_delta_multi_box | target `x_{t+1}-x_t` | encoder + decoder | Same detach rules as recon. |
| h2z | target `z_t` detached | h_to_z + predictor (via h states) | No encoder grads. |
| z2h | z_t detached (t >= start_frame) | z_to_h only (h_to_z frozen for aux) | z_t -> z_to_h -> h_to_z -> compare to z_t. |
| inverse_dynamics_h | uses `h_t, h_{t+1}` | inverse_dynamics_h head + predictor | No encoder grads (z detached). |
| action_delta_h | uses `h_{t+1}-h_t` | h_action_delta_projector + predictor | No encoder grads (z detached). |
| rollout_kstep_h | compares predicted h rollouts | predictor + h_to_z | z inputs detached by default. |
| rollout_kstep_delta_h | predicted delta rollouts | predictor + h_to_z | z inputs detached by default. |
| action_delta_dp | ΔP (pose delta) vs action prototype | dp_action_delta_projector + p_action_delta_projector | `pose_delta_detach_h=True` blocks grads into h. |
| additivity_dp | ΔP composition | p_action_delta_projector | Same as above. |
| rollout_kstep_p | k-step integration (pose) | p_action_delta_projector | Same as above. |
| scale_dp | ΔP magnitude stats vs target | p_action_delta_projector | Same as above. |
| geometry_rank_p | ranking on pose | p_action_delta_projector | No grads into h or z by default. |

## Relation to algebra notes

Aligned with `action_algebra_v5.md` and `overview_v3.md`:

- **z (appearance / place descriptor)**: no action algebra losses enabled; action-conditioned z losses are off by default.
  - Goal: keep z for loop-closure / appearance similarity, not action composition.
  - Enabled losses on z are reconstruction-style + pixel delta, which preserve observation fidelity.
- **h (dynamics state)**: local action algebra is active.
  - `action_delta_h`, `rollout_kstep_h`, and `rollout_kstep_delta_h` enforce short-horizon composability.
  - `inverse_dynamics_h` encourages action-identifiable transitions without forcing geometry.
- **p (planning pose)**: odometry algebra + planning geometry are active.
  - `action_delta_dp`, `additivity_dp`, `rollout_kstep_p` are the algebraic constraints.
  - `scale_dp` anchors magnitude; `geometry_rank_p` adds goal-conditioned ordering.
  - `pose_delta_detach_h=True` keeps p losses from shaping h directly.

## Quick checks when editing weights

- If you enable any z-action algebra loss (`action_delta_z`, rollout_z variants), re-evaluate loop-closure invariance for z.
- If you set `detach_z_from_h_and_p=False`, h/p losses will backprop into the encoder; consider whether that reintroduces action algebra into z.
- If you set `pose_delta_detach_h=False`, p losses will shape h and may conflict with predictive dynamics.
