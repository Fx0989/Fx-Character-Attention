# Fx Character Attention

Attention-couple regional prompting for one background + 2-4 character
conditionings, built for SDXL / Illustrious checkpoints.

## Install

1. Copy this folder into `ComfyUI/custom_nodes/` (keep the folder name, e.g.
   `ComfyUI/custom_nodes/character_couple_attention/`).
2. Restart ComfyUI.
3. The node appears as **"Fx Character Attention"** under
   `conditioning/attention_couple`.

## Wiring it up

- `model` → your checkpoint's MODEL output.
- `width` / `height` → your target image resolution (must match the
  `EmptyLatentImage` you sample into).
- `background` → a plain `CLIPTextEncode` (or SDXL's
  `CLIPTextEncodeSDXL`) output for the overall scene/style prompt. No mask
  needed — it's treated as covering the whole canvas.
- `character_1` / `character_2` (required) and `character_3` /
  `character_4` (optional) → each must already have a mask attached via
  **ConditioningSetMask** upstream (the node reads the mask and strength
  straight from there — don't set strength again on this node). Leave the
  3rd/4th inputs unconnected for a 2-character scene.
- `background_bleed` (0–1, default 0.2) → how much background weight
  persists *inside* each character's own mask. At 0, a character's own
  area is driven entirely by their prompt (background only fills empty
  canvas). Higher values let the background compete more with each
  character, which can lower their prompt details — if your characters
  look too similar to each other or too generic, lower this.

Outputs:
- `model` → plug into KSampler's `model` input.
- `positive` → plug into KSampler's `positive` input (this is just your
  background conditioning passed through — the character prompts are
  applied inside the patched model, not as a separate conditioning list).
- Your regular negative conditioning goes into KSampler's `negative` as
  normal — it is *not* run through this node.


## Known caveats

- **Internal API**: this uses ComfyUI's undocumented
  `ModelPatcher.set_model_attn2_replace` mechanism (the same one IPAdapter/
  ControlNet-style nodes rely on). It's stable in practice but could shift
  on a future ComfyUI update — if the node throws on load, that's the
  first place to check.
- **Batch size**: tested conceptually for batch size 1 (one image per
  generation). Larger batches should work via the row-tiling logic but
  haven't been verified — start with batch size 1.
- **SDXL pooled output**: only the background conditioning's pooled output
  is used (this matches how the reference attention-couple implementations
  handle SDXL).
- **LoRA regional targeting**: not implemented — LoRAs apply globally as
  usual, they aren't restricted to a character's mask.
- **Character count**: supports 2-4 characters (the first two inputs are
  required, the 3rd/4th are optional). Beyond 4 would need the node
  extended further with dynamic inputs.
