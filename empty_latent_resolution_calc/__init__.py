"""
Empty Latent + Resolution Calc
--------------------------------
A ComfyUI custom node that merges the behavior of "Empty Latent Image"
with a Flux-Resolution-Calc style aspect-ratio / megapixel calculator.

Given an aspect ratio (or custom ratio) and a target megapixel count, it:
  1. Computes width/height that hit the target pixel count while
     preserving the aspect ratio.
  2. Snaps both dimensions to a multiple of 8 (or 16, selectable) so the
     result is safe for the VAE.
  3. Builds the empty latent tensor with the correct channel count for
     the selected model family (SD1.5/SDXL = 4ch, SD3/Flux = 16ch, offered
     as one combined option since they share the same latent shape),
     downscaled by 8x (standard VAE stride).
  4. Returns LATENT, width (INT), and height (INT) so you can wire the
     resolution directly into other nodes without a separate node.
  5. Displays the resolved "width x height (MP)" directly on the node
     itself via a read-only widget, computed entirely client-side in
     JavaScript (see web/display.js) — no text is sent back through the
     execution/API response, so external tools that talk to ComfyUI's
     API (e.g. the Krita plugin) won't see or receive it as an output.

Install: drop this whole folder into ComfyUI/custom_nodes/ and restart
ComfyUI (full restart, not just browser refresh). The node will show up
as "Empty Latent (Resolution Calc)" under "latent/resolution".
"""

import torch


# Common aspect ratio presets. Add/remove as you like.
ASPECT_RATIOS = {
    "1:1 (Perfect Square)": (1, 1),
    "2:3 (Classic Portrait)": (2, 3),
    "3:4 (Golden Ratio)": (3, 4),
    "3:5 (Elegant Vertical)": (3, 5),
    "4:5 (Artistic Frame)": (4, 5),
    "5:7 (Balanced Portrait)": (5, 7),
    "5:8 (Tall Portrait)": (5, 8),
    "7:9 (Modern Portrait)": (7, 9),
    "9:16 (Slim Vertical)": (9, 16),
    "9:19 (Tall Slim)": (9, 19),
    "9:21 (Ultra Tall)": (9, 21),
    "9:32 (Skyline)": (9, 32),
    "3:2 (Golden Landscape)": (3, 2),
    "4:3 (Classic Landscape)": (4, 3),
    "5:3 (Wide Horizon)": (5, 3),
    "5:4 (Balanced Frame)": (5, 4),
    "7:5 (Elegant Landscape)": (7, 5),
    "8:5 (Cinematic View)": (8, 5),
    "9:7 (Artful Horizon)": (9, 7),
    "16:9 (Panorama)": (16, 9),
    "19:9 (Cinematic Ultrawide)": (19, 9),
    "21:9 (Epic Ultrawide)": (21, 9),
    "32:9 (Extreme Ultrawide)": (32, 9),
    "custom": None,  # use custom_ratio_w / custom_ratio_h instead
}

# Megapixel dropdown options, 0.1 -> 2.0 in steps of 0.1
MEGAPIXEL_OPTIONS = [f"{round(0.1 * i, 1)}" for i in range(1, 21)]

# channel count + latent downscale factor per model family.
# SD3 and Flux share the same 16-channel / 8x-downscale VAE, so they're
# offered as a single combined option.
MODEL_LATENT_SPECS = {
    "SD1.5 / SDXL": {"channels": 4, "downscale": 8},
    "SD3 / Flux": {"channels": 16, "downscale": 8},
}


def _round_to_multiple(value: float, multiple: int) -> int:
    return max(multiple, int(round(value / multiple) * multiple))


def _resolve_dimensions(ratio_w, ratio_h, megapixels, snap):
    """Given a ratio and a target megapixel count, return (width, height)
    snapped to a multiple of `snap`, as close as possible to the target
    pixel count while preserving the ratio."""
    target_pixels = megapixels * 1_000_000
    # area = w * h, w/h = ratio_w/ratio_h  ->  w = sqrt(area * ratio_w/ratio_h)
    raw_w = (target_pixels * (ratio_w / ratio_h)) ** 0.5
    raw_h = raw_w * (ratio_h / ratio_w)

    width = _round_to_multiple(raw_w, snap)
    height = _round_to_multiple(raw_h, snap)
    return width, height


def _parse_custom_ratio(ratio_str):
    """Parse a 'W:H' string into (w, h) floats. Falls back to 1:1 on
    anything malformed (empty, missing colon, zero/negative, non-numeric)."""
    try:
        w_str, h_str = ratio_str.split(":")
        w, h = float(w_str.strip()), float(h_str.strip())
        if w <= 0 or h <= 0:
            raise ValueError
        return w, h
    except Exception:
        return 1.0, 1.0


class EmptyLatentResolutionCalc:
    """Empty Latent Image + Flux-style resolution calculator, in one node.
    Displays the resolved width x height directly on the node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_type": (list(MODEL_LATENT_SPECS.keys()), {"default": "SD3 / Flux"}),
                "aspect_ratio": (list(ASPECT_RATIOS.keys()), {"default": "1:1 (Perfect Square)"}),
                "megapixels": (MEGAPIXEL_OPTIONS, {"default": "1.0"}),
                "snap_to_multiple_of": (["8", "16", "32", "64"], {"default": "16"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            },
            "optional": {
                "custom_ratio": ("STRING", {"default": "1:1"}),
            },
        }

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("latent", "width", "height")
    FUNCTION = "generate"
    CATEGORY = "latent/resolution"

    def generate(
        self,
        model_type,
        aspect_ratio,
        megapixels,
        snap_to_multiple_of,
        batch_size,
        custom_ratio="1:1",
    ):
        ratio = ASPECT_RATIOS[aspect_ratio]
        if ratio is None:  # "custom"
            ratio_w, ratio_h = _parse_custom_ratio(custom_ratio)
        else:
            ratio_w, ratio_h = ratio

        width, height = _resolve_dimensions(
            ratio_w, ratio_h, float(megapixels), int(snap_to_multiple_of)
        )

        spec = MODEL_LATENT_SPECS[model_type]
        channels = spec["channels"]
        downscale = spec["downscale"]

        latent = torch.zeros(
            [batch_size, channels, height // downscale, width // downscale]
        )

        return ({"samples": latent}, width, height)


NODE_CLASS_MAPPINGS = {
    "EmptyLatentResolutionCalc": EmptyLatentResolutionCalc,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EmptyLatentResolutionCalc": "Fx Smart Latent Image",
}

WEB_DIRECTORY = "web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
