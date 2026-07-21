# Fx Smart Latent Image

A ComfyUI custom node that merges "Empty Latent Image" with a Flux-style
aspect-ratio / megapixel resolution calculator — in a single node.

## Features
- Pick an aspect ratio preset (or a custom ratio) and a target megapixel count.
- Automatically computes width/height, snapped to a multiple of 8/16/etc.
- Builds the empty latent tensor with the correct channel count for
  SD1.5/SDXL (4ch) or SD3/Flux (16ch).
- Outputs `LATENT`, `width` (INT), and `height` (INT).
- Displays the resolved `width x height (MP)` directly on the node itself.

## Installation
1. Copy this entire folder into your `ComfyUI/custom_nodes/` directory, so
   the path looks like:
   ```
   ComfyUI/custom_nodes/empty_latent_resolution_calc/__init__.py
   ComfyUI/custom_nodes/empty_latent_resolution_calc/web/display.js
   ```
2. Fully restart ComfyUI (not just a browser refresh — it needs to pick up
   the new node folder and the web directory).
3. In the node search/add menu, look for **"Fx Smart Latent Image"**
   under the category **latent/resolution**.

## Remembering your last-used settings
New copies of this node start from whatever `aspect_ratio`, `megapixels`,
and `snap_to_multiple_of` you last used — not from a fixed hardcoded
default. This is remembered per-browser (via localStorage), so it's local
to the machine/browser you're using, not saved inside the workflow file
itself. Loading an existing saved workflow always restores that
workflow's own saved values, regardless of what you last used elsewhere.

## Usage
- `model_type`: choose SD1.5/SDXL, or SD3 / Flux (sets latent channel count — SD3 and Flux share the same 16-channel latent, so they're one combined option).
- `aspect_ratio`: pick a named preset (23 options, spanning square,
  portrait, landscape, and ultrawide ratios), or choose "custom" and set
  `custom_ratio` (a single `W:H` text field, e.g. `13:7`, default `1:1`).
  This field is always visible/editable, but only affects the result when
  `aspect_ratio` is set to "custom".
- `megapixels`: dropdown from 0.1 to 2.0 in steps of 0.1.
- `snap_to_multiple_of`: dropdown (8, 16, 32, or 64) — rounds width/height
  to this multiple. 16 is a safe default for Flux; use 32 or 64 for extra
  safety margin against edge artifacts on unusual aspect ratios.
- `batch_size`: number of latents to generate in the batch.

### Aspect ratio presets
- Portrait: 2:3, 3:4, 3:5, 4:5, 5:7, 5:8, 7:9, 9:16, 9:19, 9:21, 9:32
- Landscape: 3:2, 4:3, 5:3, 5:4, 7:5, 8:5, 9:7, 16:9, 19:9, 21:9, 32:9
- Square: 1:1
- Custom: set your own ratio via the single `custom_ratio` field (format `W:H`)

Outputs:
- `latent`: an empty latent tensor, ready to feed into a KSampler.
- `width` / `height`: INT outputs you can wire into other nodes.
- A read-only text field on the node shows the resolved resolution, e.g.
  `1024 x 1024  (1.05 MP)`. This preview updates live as you change any
  setting (aspect ratio, megapixels, snap, or custom ratio) — you don't
  need to run the workflow first to see the resulting size. It's computed
  entirely client-side in the browser and is never included in the node's
  execution/API response, so external tools that talk to ComfyUI's API
  won't pick it up as a text output.
