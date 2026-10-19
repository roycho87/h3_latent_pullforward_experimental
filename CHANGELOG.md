# Changelog

## v0.1.0 — first public release (experimental)

First release. Everything here should be treated as a prototype.

* **Motion Mask** — finds what moves in the decoded clip, outputs a motion mask and a stable mask.
* **H3 Context Motion Mask** — the same, carried across clips so the mask knows what the previous clip was doing.
* **H3 Stable Mask To Latent** — converts the pixel stable mask to the H3 16×16 latent grid.
* **H3 Latent Pull-Forward** — pastes each still cell's anchor forward in the latent. Block-aware (`block_latents` 5),
  with a cumulative clip-1 reference that adds cells as they settle and revokes them if they move.
* **H3 Moving Mask**, **H3 Latent Tail / Tail Merge**, **H3 Mask To Video Image** — helpers for the resample pass and
  for debugging.
* **Sequential Wildcard Processor** (in `extras/`) — walks a wildcard list in order, one line per clip.
* Workflows with the exact settings that were run, plus a no-Pull-Forward baseline for an A/B.

Known issues:
* `ignore_first_latent_frames` ships at 7 (22 frames) while the shipped workflow uses a 39-frame overlap. 12 is the
  matching value.
* Locked-off camera only. `camera_compensation` exists but has had almost no testing.
* The resample pass needs a second model in VRAM.
