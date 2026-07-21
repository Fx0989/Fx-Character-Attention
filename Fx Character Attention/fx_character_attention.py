"""
Character Couple Attention — ComfyUI custom node
--------------------------------------------------
Attention-couple style regional prompting for SDXL / Illustrious models.

Takes a background conditioning (covers the whole canvas) and 2-4 character
conditionings (each expected to already carry a mask baked in via
ConditioningSetMask). Patches every cross-attention (attn2) layer of the
UNet so that:

  - each spatial location computes attention separately against each
    region's own text embeddings
  - the per-region outputs are blended by *normalized* mask weight at
    that location

Because the blend is normalized per-pixel (weights always sum to 1), areas
where two character masks overlap (e.g. a hug) automatically get a smooth
proportional mix of both prompts instead of either one fighting the other,
and without needing to manually merge prompt text.

This relies on ComfyUI's internal (undocumented but widely used by
IPAdapter / ControlNet / GLIGEN style nodes) attn2 replacement API:
  ModelPatcher.set_model_attn2_replace(patch_fn, block_name, number, transformer_index)
and the `transformer_options["cond_or_uncond"]` convention passed into the
patch function. If this breaks on a future ComfyUI version, the two spots
most likely to need adjustment are marked with "# COMFY-INTERNAL".
"""

import torch
import torch.nn.functional as F

from comfy.ldm.modules.attention import optimized_attention


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _extract_cond_and_mask(conditioning, target_h, target_w, device):
    """
    Pull the text-embedding tensor and (optionally) the mask out of a
    ComfyUI CONDITIONING list entry. ConditioningSetMask stores the mask
    under cond[1]['mask'] at whatever resolution it was authored at, so we
    resize it once here to the working (height, width) the node was given.

    Returns (context_tensor[1, seq, dim], mask_tensor[1, target_h, target_w] or None)
    """
    cond_tensor, cond_dict = conditioning[0]
    context = cond_tensor.to(device)

    mask = cond_dict.get("mask", None)
    if mask is None:
        return context, None

    mask = mask.to(device).float()
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)  # [1, H, W]
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)  # [1, 1, H, W]

    mask = F.interpolate(mask, size=(target_h, target_w), mode="bilinear", align_corners=False)
    mask = mask.squeeze(1)  # [1, target_h, target_w]

    strength = cond_dict.get("mask_strength", 1.0)
    if strength != 1.0:
        mask = mask * strength

    return context, mask


def _resize_mask_to_tokens(mask, seq_len, latent_h, latent_w, device, dtype):
    """
    Downsample a [1, H, W] mask (at full working resolution) to whatever
    spatial resolution the current attention layer operates at, inferred
    from seq_len vs. the known latent aspect ratio, then flatten to tokens.

    Returns [1, seq_len, 1]
    """
    scale = (seq_len / (latent_h * latent_w)) ** 0.5
    rh = max(1, round(latent_h * scale))
    rw = max(1, round(latent_w * scale))

    m = F.interpolate(mask.unsqueeze(1), size=(rh, rw), mode="nearest-exact")
    m = m.reshape(1, rh * rw, 1)

    if m.shape[1] != seq_len:
        # Fallback for odd aspect-ratio rounding mismatches: interpolate the
        # flattened token dimension directly rather than crash.
        m = F.interpolate(m.permute(0, 2, 1), size=seq_len, mode="nearest-exact").permute(0, 2, 1)

    return m.to(device=device, dtype=dtype)


def _make_attn2_patch(module, regions, latent_h, latent_w):
    """
    regions: list of dicts, each {"context": [1, seq, dim], "mask": [1, H, W] or None}
    module:  the specific attn2 CrossAttention submodule this patch is bound to
             (captured via closure so we can reuse its to_k / to_v projections)
    """
    def patch(q, k, v, extra_options):
        # COMFY-INTERNAL: cond_or_uncond is a list of 0 (cond) / 1 (uncond)
        # flags, one per "group" the batch dimension was tiled from.
        cond_or_uncond = extra_options.get("cond_or_uncond", [0])
        n_heads = extra_options["n_heads"]

        total_rows = q.shape[0]
        num_groups = max(1, len(cond_or_uncond))
        rows_per_group = max(1, total_rows // num_groups)

        seq_len = q.shape[1]
        # zeros_like (not empty_like): if row-group math ever leaves a row
        # unassigned, it stays a harmless zero instead of propagating
        # uninitialized memory / NaNs through the rest of the UNet.
        out = torch.zeros_like(q)

        for gi in range(num_groups):
            flag = cond_or_uncond[gi] if gi < len(cond_or_uncond) else 1
            row_slice = slice(gi * rows_per_group, (gi + 1) * rows_per_group)

            if flag != 0:
                # Uncond / unrelated pass — leave untouched, standard attention.
                out[row_slice] = optimized_attention(
                    q[row_slice], k[row_slice], v[row_slice], n_heads
                )
                continue

            rows_here = q[row_slice].shape[0]
            region_outputs = []
            region_weights = []

            for region in regions:
                r_context = region["context"].to(device=q.device, dtype=q.dtype)
                if r_context.shape[0] == 1 and rows_here > 1:
                    r_context = r_context.expand(rows_here, -1, -1)

                r_k = module.to_k(r_context)
                r_v = module.to_v(r_context)
                r_out = optimized_attention(q[row_slice], r_k, r_v, n_heads)
                region_outputs.append(r_out)

                if region["mask"] is not None:
                    w = _resize_mask_to_tokens(
                        region["mask"], seq_len, latent_h, latent_w, q.device, q.dtype
                    )
                else:
                    w = torch.ones((1, seq_len, 1), device=q.device, dtype=q.dtype)
                if w.shape[0] == 1 and rows_here > 1:
                    w = w.expand(rows_here, -1, -1)
                region_weights.append(w)

            weight_stack = torch.stack(region_weights, dim=0)          # [n_regions, rows, seq, 1]
            norm = weight_stack.sum(dim=0).clamp(min=1e-6)             # [rows, seq, 1]

            blended = torch.zeros_like(q[row_slice])
            for r_out, w in zip(region_outputs, region_weights):
                blended = blended + r_out * (w / norm)

            out[row_slice] = blended

        return out

    return patch


def _enumerate_attn2_modules(unet):
    """
    Walk the UNet and yield (block_name, block_id, transformer_index, module)
    for every cross-attention (attn2) submodule, matching the key format
    ComfyUI's ModelPatcher.set_model_attn2_replace expects.
    """
    results = []
    for name, module in unet.named_modules():
        if not name.endswith("attn2"):
            continue
        parts = name.split(".")
        raw_block_name = parts[0]  # "input_blocks" / "middle_block" / "output_blocks"
        # ComfyUI's internal transformer_options use short names ("input",
        # "middle", "output") when constructing the lookup key during the
        # forward pass — not the raw nn.Module attribute names.
        block_name_map = {
            "input_blocks": "input",
            "middle_block": "middle",
            "output_blocks": "output",
        }
        block_name = block_name_map.get(raw_block_name, raw_block_name)
        block_id = int(parts[1]) if raw_block_name != "middle_block" else 0

        transformer_index = 0
        for i, p in enumerate(parts):
            if p == "transformer_blocks":
                transformer_index = int(parts[i + 1])
                break

        results.append((block_name, block_id, transformer_index, module))

    return results


# --------------------------------------------------------------------------
# Node
# --------------------------------------------------------------------------

class FxCharacterAttention:
    """
    Regional prompting via attention-couple for one background conditioning
    plus 2-4 character conditionings. Character conditionings should already
    have a mask attached via ConditioningSetMask.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "background": ("CONDITIONING",),
                "character_1": ("CONDITIONING",),
                "character_2": ("CONDITIONING",),
                "background_bleed": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "How much background influence persists inside "
                               "character masks. 0 = characters fully own their "
                               "area, background only fills empty canvas. "
                               "Higher values let the background compete more "
                               "with each character, which can lower their "
                               "prompt details."
                }),
            },
            "optional": {
                "character_3": ("CONDITIONING",),
                "character_4": ("CONDITIONING",),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING")
    RETURN_NAMES = ("model", "positive")
    FUNCTION = "apply"
    CATEGORY = "conditioning/attention_couple"

    def apply(
        self,
        model,
        width,
        height,
        background,
        character_1,
        character_2,
        background_bleed=0.2,
        character_3=None,
        character_4=None,
    ):
        device = model.load_device if hasattr(model, "load_device") else "cpu"
        latent_h = height // 8
        latent_w = width // 8

        bg_context, bg_mask_explicit = _extract_cond_and_mask(background, latent_h, latent_w, device)

        character_conditionings = [
            ("character_1", character_1),
            ("character_2", character_2),
            ("character_3", character_3),
            ("character_4", character_4),
        ]

        character_regions = []
        character_coverage = None
        for input_name, conditioning in character_conditionings:
            if conditioning is None:
                continue
            context, mask = _extract_cond_and_mask(conditioning, latent_h, latent_w, device)
            if mask is None:
                raise ValueError(
                    f"{input_name} has no mask attached. "
                    "Run it through ConditioningSetMask before plugging it into this node."
                )
            character_regions.append({"context": context, "mask": mask})
            character_coverage = mask if character_coverage is None else character_coverage + mask

        if len(character_regions) < 2:
            raise ValueError(
                "At least character_1 and character_2 are required."
            )

        character_coverage = torch.clamp(character_coverage, min=0.0, max=1.0)

        # Background gets full weight on empty canvas, and only
        # `background_bleed` fraction of weight inside character masks —
        # otherwise it competes 50/50 with each character everywhere and
        # drowns out their distinguishing prompt details.
        bg_mask = torch.clamp(1.0 - character_coverage, min=0.0, max=1.0)
        bg_mask = bg_mask + background_bleed * character_coverage
        if bg_mask_explicit is not None:
            # Respect an explicit background mask if one was attached, on
            # top of the empty-canvas/bleed logic above.
            bg_mask = bg_mask * bg_mask_explicit

        regions = [{"context": bg_context, "mask": bg_mask}] + character_regions

        model_patched = model.clone()
        unet = model_patched.model.diffusion_model

        for block_name, block_id, transformer_index, module in _enumerate_attn2_modules(unet):
            patch_fn = _make_attn2_patch(module, regions, latent_h, latent_w)
            # COMFY-INTERNAL: signature is (patch, block_name, number, transformer_index)
            model_patched.set_model_attn2_replace(patch_fn, block_name, block_id, transformer_index)

        # The background conditioning is what actually goes into KSampler's
        # positive input — it carries the base text embedding (and, for
        # SDXL, the pooled_output) that the sampler needs outside of the
        # attention patch itself.
        return (model_patched, background)


NODE_CLASS_MAPPINGS = {
    "FxCharacterAttention": FxCharacterAttention,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FxCharacterAttention": "Fx Character Attention",
}
