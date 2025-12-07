# ConvNeXt-inspired encoder/decoder ideas for JEPA (224x224 inputs)

## Goals and constraints
- Keep roughly the same parameter budget as the current CNN encoder/decoder schedule.
- Improve sensitivity to small (7–10 px) objects and tiny (1–2 px) motions between frames.
- Remain convolutional/few-attention so training speed stays close to current implementation.

## Encoder suggestions (ConvNeXt / ConvNeXtV2 flavor)
1. **Patch-embedding front-end tuned for small motion**
   - Replace the initial stride-2 conv with a *two-layer patch stem*: `3x3, stride 1` followed by `3x3, stride 2` using depthwise + pointwise (DW+PW). This keeps early spatial resolution and preserves 1–2 px movements before downsampling.
   - Use **anti-aliased downsampling** (blur-pool or average-pool before stride) to avoid washing out tiny displacements.

2. **ConvNeXt blocks per stage**
   - Each stage: DW conv (kernel 7x7, padding 3) → LayerNorm (channels-last) → pointwise MLP (1x1, GELU, 4x expansion) → optional **stochastic depth**.
   - Replace GroupNorm with LayerNorm to match ConvNeXt behavior; add **learnable scaling (γ)** on the residual branch to stabilize with small channel counts.
   - Keep stage widths similar to current schedule; adjust block counts (e.g., 2–3 blocks early, 3–4 mid, 2 late) to keep parameters steady.

3. **ConvNeXtV2 refinements for motion sensitivity**
   - Swap GELU for **SiLU** and include **Global Response Normalization (GRN)** in the PW-MLP to boost local contrast for tiny objects.
   - Add **dilated DW conv** in the highest-resolution stage (dilation=2) to enlarge context without extra downsampling.

4. **Multi-scale high-frequency pathway**
   - Keep a shallow **"detail skip"**: a lightweight DW+PW branch at 112x112 feeding into mid-stage via concatenation or gated add. This preserves sub-4px cues lost after stride-2.

## Decoder suggestions
1. **Mirror stages with ConvNeXt-style up blocks**
   - Use ConvTranspose or nearest-neighbor upsample followed by DW 7x7 + LN + PW-MLP (with GRN). Maintain the γ-scaling and stochastic depth mirrors.
   - Inject the **detail skip** from the encoder into corresponding resolution for sharper small-object reconstructions.

2. **Anti-aliased upsampling**
   - After each upsample, apply a small blur (e.g., 3x3 separable) before DW conv to reduce checkerboard artifacts that obscure 1–2 px motion.

3. **Output sharpening head**
   - Final stage: DW 5x5 + PW conv with **residual high-pass branch** (e.g., Laplacian filter) to emphasize fine edges in the prediction target.

## Training/regularization tweaks
- Use **stronger weight decay on depthwise kernels** and milder decay on PW to keep DW filters responsive to micro-motion.
- Apply **Mixup/CutMix at low ratios** (lambda≈0.1) to avoid blurring small objects; favor random shifts/rotations of ≤2 px for augmentation.
- Monitor **feature map std/mean at 112x112 and 56x56** to ensure small-motion signals survive early stages.

## Outline to implement
1. **Define ConvNeXt-style block** with DW 7x7, LN (channels-last), PW-MLP (4x hidden), optional GRN and γ scaling.
2. **Refit encoder schedule**: patch stem (1x stride1 + 1x stride2 DW+PW), then stages with similar widths; add a detail skip at 112x112.
3. **Refit decoder** mirroring stages; integrate skip connections and anti-aliased upsampling; add edge-sharpen head.
4. **Adjust predictor inputs** if latent dim changes; keep latent width close to existing to match parameter budget.
5. **Add monitoring utilities** for feature stats at early stages to validate sensitivity to 1–2 px motion.
