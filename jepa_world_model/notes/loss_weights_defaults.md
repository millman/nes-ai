# LossWeights Defaults and Rationale

Suggested default values (balanced auxiliaries):

- jepa = 1.0
- sigreg = 0.01
- recon = 0.0
- recon_patch = 0.0
- recon_multi_gauss = 0.0
- recon_multi_box = 0.15
- pixel_delta = 0.05
- h2z = 0.1
- h2z_2hop = 0.02
- delta_z = 0.05
- inverse_dynamics_z = 0.05
- inverse_dynamics_h = 0.05
- geometry_rank_p = 0.02

Rationale by parameter:

- jepa (1.0): anchor loss; all other terms are auxiliary and should be scaled below the main transition objective.
- sigreg (0.01): regularizer kept small to stabilize embeddings without dominating gradients.
- recon (0.0): avoid stacking multiple pixel losses by default; use one reconstruction path at a time.
- recon_patch (0.0): redundant with multi-scale hardness unless explicitly testing patch overlap behavior.
- recon_multi_gauss (0.0): keep gaussian and box hardness mutually exclusive for clearer attribution.
- recon_multi_box (0.15): enough to improve decoder stability while staying well below the main JEPA objective.
- pixel_delta (0.05): modest temporal consistency signal that should not overpower absolute reconstruction or JEPA.
- h2z (0.1): helpful alignment of hidden state to z, but smaller than JEPA because it is an auxiliary constraint.
- h2z_2hop (0.02): two-hop composability is noisier; keep small to avoid destabilizing 1-step dynamics.
- delta_z (0.05): light auxiliary for smooth latent transitions; similar scale to other minor terms.
- inverse_dynamics_z (0.05): action inference from z is secondary; keep in the same small band.
- inverse_dynamics_h (0.05): symmetric with z inverse dynamics to avoid overemphasis on either latent.
- geometry_rank_p (0.02): ranking losses can be brittle; keep small to prevent training collapse early on.

Notes:

- The defaults assume no loss normalization. If loss normalization is enabled, the auxiliary weights can be raised.
- If you want a "pure JEPA" baseline, set all weights except jepa and sigreg to 0.
- Prefer enabling exactly one of recon/recon_patch/recon_multi_gauss/recon_multi_box at a time.
