"""H3 Context Motion Mask  --  stand-alone node, separate from the Pull-Forward pack and from the original Motion Mask.

A drop-in replacement for Motion Mask (same settings, same four outputs) that carries MOTION INFORMATION from one clip to the next
through Motion Context. It does not make a mask of its own. The last `context_frames` frames of clip N are the first
`context_frames` frames of clip N+1, but clip N+1 has no history of what happened BEFORE them, so `linger` and `hold` start from
zero there. This node saves, at the end of every clip:
  * the raw motion of its last `context_frames` frames (looked at with clip N's own neighbours), and
  * for every pixel, how long it had been still (for linger) and how long its stable run was (for hold) just before those frames.
Clip N+1 loads that and (1) adds the saved motion to its first `context_frames` frames and (2) starts linger and hold from the saved
counters instead of from zero, so a pixel that moved at the end of clip N is treated as having moved when the next clip decides
what is stable. Pure numpy. Drop this file into ComfyUI/custom_nodes/ and restart ComfyUI. GPL-3.0.
"""
import math
import os

import numpy as np

# ------------------------------------------------------------------ small image filters (numpy only)


def _box_1d(x, r, axis):
    """Mean filter of radius r along `axis`, edge-replicated (cumulative-sum trick, O(1) per pixel)."""
    if r <= 0:
        return x
    x = np.moveaxis(x, axis, -1)
    xp = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(r + 1, r)], mode='edge')
    c = np.cumsum(xp, axis=-1, dtype=np.float64)
    n = x.shape[-1]
    out = ((c[..., 2 * r + 1: 2 * r + 1 + n] - c[..., :n]) / (2 * r + 1)).astype(np.float32)
    return np.moveaxis(out, -1, axis)


def blur(x, sigma):
    """Approximate Gaussian blur (3 box passes) over the last two axes. sigma in pixels."""
    if sigma <= 0:
        return x
    w = math.sqrt(4.0 * sigma * sigma + 1.0)
    r = max(1, int(round((w - 1) / 2)))
    for _ in range(3):
        x = _box_1d(x, r, -1)
        x = _box_1d(x, r, -2)
    return x


def _max_1d(x, r, axis, fill):
    """Sliding-window maximum, window 2r+1, centred, O(log r) passes (sparse-table trick)."""
    k = 2 * r + 1
    x = np.moveaxis(x, axis, -1)
    n = x.shape[-1]
    pad = np.full(x.shape[:-1] + (r,), fill, x.dtype)
    xp = np.concatenate([pad, x, pad], axis=-1)
    J = int(math.floor(math.log2(k)))
    M, s = xp, 1
    for _ in range(J):
        L = M.shape[-1] - s
        M = np.maximum(M[..., :L], M[..., s:s + L])
        s *= 2
    off = k - (1 << J)
    out = np.maximum(M[..., :n], M[..., off:off + n])
    return np.moveaxis(out, -1, axis)


def dilate(m, r):
    """Grow white areas by r pixels (square footprint)."""
    if r <= 0:
        return m
    return _max_1d(_max_1d(m, r, -1, 0), r, -2, 0)


def erode(m, r):
    """Shrink white areas by r pixels. Image borders do not erode."""
    if r <= 0:
        return m
    return 1 - dilate(1 - m, r)


def close_gaps(m, r):
    """Bridge gaps up to ~2r px between white strokes along rows, then along columns (fills the inside of flat moving shapes)."""
    if r <= 0:
        return m
    for ax in (-1, -2):
        m = 1 - _max_1d(1 - _max_1d(m, r, ax, 0), r, ax, 0)       # dilate then erode along one axis; borders do not erode
    return m


def temporal_max(m, linger, lookahead):
    """Keep a pixel white for `linger` frames after motion and switch it on `lookahead` frames before."""
    out = m.copy()
    for s in range(1, min(linger, len(m) - 1) + 1):
        out[s:] = np.maximum(out[s:], m[:-s])
    for s in range(1, min(lookahead, len(m) - 1) + 1):
        out[:-s] = np.maximum(out[:-s], m[s:])
    return out


# ------------------------------------------------------------------ camera translation estimate (optional)


def estimate_shift(a, b, max_frac=0.2, min_peak=0.03, weight=None):
    """Integer (dy, dx) roll to apply to `b` so that it lines up with `a` (phase correlation)."""
    H, W = a.shape
    f = max(1, max(H, W) // 192)
    h, w = (H // f) * f, (W // f) * f
    ds = lambda z: z[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3))
    a2, b2 = ds(a), ds(b)
    win = np.outer(np.hanning(a2.shape[0]), np.hanning(a2.shape[1]))
    if weight is not None:                                   # only still pixels vote (a big moving subject must not drag the estimate)
        w2 = ds(np.asarray(weight, np.float32))
        if w2.sum() < 0.05 * w2.size:
            return 0, 0
        win = win * w2
        a2 = a2 - (a2 * w2).sum() / w2.sum()
        b2 = b2 - (b2 * w2).sum() / w2.sum()
    A, B = np.fft.fft2((a2 - a2.mean()) * win), np.fft.fft2((b2 - b2.mean()) * win)
    R = A * np.conj(B)
    r = np.fft.ifft2(R / (np.abs(R) + 1e-9)).real
    if r.max() < min_peak:
        return 0, 0
    py, px = np.unravel_index(int(r.argmax()), r.shape)
    py = py - r.shape[0] if py > r.shape[0] // 2 else py
    px = px - r.shape[1] if px > r.shape[1] // 2 else px
    dy, dx = int(py) * f, int(px) * f          # peak sits at -(shift of b), i.e. exactly the roll that re-aligns b
    if abs(dy) > max_frac * H or abs(dx) > max_frac * W:
        return 0, 0
    return dy, dx


def _pair_changed(a, b, threshold, compensate):
    """bool [H,W]: True where frame b differs from frame a by more than `threshold`."""
    if compensate:
        dy, dx = estimate_shift(a, b)
        if dy or dx:
            b = np.roll(b, (dy, dx), axis=(0, 1))
            d = np.abs(a - b) > threshold
            if dy > 0: d[:dy] = True          # strips that wrapped around = newly revealed = treat as moving
            elif dy < 0: d[dy:] = True
            if dx > 0: d[:, :dx] = True
            elif dx < 0: d[:, dx:] = True
            return d
    return np.abs(a - b) > threshold


# ------------------------------------------------------------------ the algorithm

COMPARE_MODES = ["previous and next frame", "previous frame", "first frame (change since start)"]


def motion_masks(frames, threshold=0.04, compare=COMPARE_MODES[0], frame_gap=1, blur_sigma=2.0, despeckle=2,
                 grow=8, linger=2, lookahead=2, feather=4.0, camera="off", invert=False, fill_holes=40):
    """frames: float array [B,H,W,3(+)] in 0..1.   returns mask [B,H,W] float32 in 0..1"""
    frames = np.asarray(frames, dtype=np.float32)
    B, H, W = frames.shape[:3]
    if B < 2:
        z = np.zeros((B, H, W), np.float32)
        return 1 - z if invert else z
    rgb = frames[..., :3] if frames.shape[-1] >= 3 else np.repeat(frames[..., :1], 3, -1)
    luma = np.empty((B, H, W), np.float32)
    for i in range(B):                                   # one frame at a time keeps memory low
        luma[i] = blur(rgb[i] @ np.array([0.299, 0.587, 0.114], np.float32), blur_sigma)
    comp = camera == "translation"
    g = max(1, min(int(frame_gap), B - 1))

    if compare.startswith("first"):
        ch = np.stack([_pair_changed(luma[0], luma[t], threshold, comp) if t else np.zeros((H, W), bool) for t in range(B)])
        ch[0] = ch[1]
    else:
        P = [_pair_changed(luma[i], luma[i + g], threshold, comp) for i in range(B - g)]     # change between i and i+g
        prev = [P[t - g] if t >= g else P[min(t, B - g - 1)] for t in range(B)]
        nxt = [P[t] if t < B - g else P[B - g - 1] for t in range(B)]
        ch = np.stack([prev[t] if compare == "previous frame" else (prev[t] | nxt[t]) for t in range(B)])
    m = ch.astype(np.uint8)

    CH = 16                                              # spatial clean-up in chunks to bound memory
    if despeckle > 0:
        for i in range(0, B, CH):
            m[i:i + CH] = dilate(erode(m[i:i + CH], despeckle), despeckle)
    if linger or lookahead:
        m = temporal_max(m, int(linger), int(lookahead))
    if fill_holes > 0:
        for i in range(0, B, CH):
            m[i:i + CH] = close_gaps(m[i:i + CH], int(fill_holes) // 2)
    out = np.empty((B, H, W), np.float32)
    for i in range(0, B, CH):
        c = dilate(m[i:i + CH], grow).astype(np.float32) if grow > 0 else m[i:i + CH].astype(np.float32)
        out[i:i + CH] = np.clip(blur(c, feather), 0, 1) if feather > 0 else c
    return 1 - out if invert else out


def overlay_frames(frames, mask, strength=0.55):
    rgb = np.asarray(frames, np.float32)[..., :3]
    t = (strength * mask)[..., None]
    return rgb * (1 - t) + np.array([1.0, 0.0, 1.0], np.float32) * t


def make_report(mask, fps_hint=None):
    cov = mask.reshape(len(mask), -1).mean(1) * 100
    k = int(cov.argmax())
    return (f"{len(mask)} frames | motion coverage: mean {cov.mean():.1f}% | max {cov.max():.1f}% (frame {k}) | "
            f"frames with <0.5% motion: {int((cov < 0.5).sum())}")


def stability_mask(moving, hold):
    """moving: bool [B,H,W] (True = moving).  Returns float32 [B,H,W]: 1 = stable, 0 = not.
    Everything starts stable (white). A pixel turns black the moment it moves, stays black while it moves, and turns white again
    only after it has been still for `hold` frames in a row; it then stays white until it moves again."""
    B, H, W = moving.shape
    hold = max(1, int(hold))
    count = np.full((H, W), hold, np.int32)
    out = np.empty((B, H, W), np.float32)
    for t in range(B):
        count = np.where(moving[t], 0, np.minimum(count + 1, hold))
        out[t] = count >= hold
    return out




def MotionMask_INPUTS():
    return {"required": {
        "images": ("IMAGE", {"tooltip": "Video frames (a batch of images)."}),
        "threshold": ("FLOAT", {"default": 0.01, "min": 0.002, "max": 1.0, "step": 0.002,
                                 "tooltip": "How much a pixel must change (0-1 brightness) to count as motion. "
                                            "Lower = more sensitive. 0.01 is the default for locked-off AI video (it catches slow motion); "
                                            "raise it if the background flickers white."}),
        "compare": (COMPARE_MODES, {"default": COMPARE_MODES[0],
                                    "tooltip": "What each frame is compared with. 'previous and next' finds what is moving "
                                               "at this instant; 'first frame' flags anything that differs from the start."}),
        "frame_gap": ("INT", {"default": 1, "min": 1, "max": 16,
                              "tooltip": "Compare with the frame this many steps away. Use 2-4 to catch slow motion."}),
        "blur": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 20.0, "step": 0.5,
                           "tooltip": "Pre-blur (px) that hides compression noise before comparing."}),
        "despeckle": ("INT", {"default": 2, "min": 0, "max": 32,
                              "tooltip": "Removes isolated white specks smaller than ~2x this many pixels."}),
        "fill_holes": ("INT", {"default": 80, "min": 0, "max": 200, "step": 4,
                              "tooltip": "Fills the inside of moving shapes: flat areas (a plain shirt) barely change between frames, so "
                                         "only their edges light up. Bridges white gaps up to this many px wide. 40 is a good start; "
                                         "above ~100 it starts painting rectangular chunks of background."}),
        "grow": ("INT", {"default": 16, "min": 0, "max": 128,
                         "tooltip": "Expands the white areas by this many pixels (safety margin around moving parts)."}),
        "linger": ("INT", {"default": 22, "min": 0, "max": 4096,
                           "tooltip": "Frames the mask stays white AFTER motion stops (areas that just moved). Set it to the clip length to hold every pixel that ever moved for the rest of the clip."}),
        "lookahead": ("INT", {"default": 3, "min": 0, "max": 48,
                              "tooltip": "Frames the mask turns white BEFORE motion starts (areas about to move/uncover)."}),
        "feather": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 64.0, "step": 0.5,
                              "tooltip": "Softens the mask edge (px). 0 = hard black/white mask."}),
        "camera_compensation": (["off", "translation"], {"default": "off",
                                "tooltip": "'translation' cancels a camera pan/shake before comparing, so only things moving "
                                           "relative to the scene turn white. Does not handle zoom or rotation."}),
        "invert": ("BOOLEAN", {"default": False, "tooltip": "White = static, black = motion."}),
        "hold": ("INT", {"default": 22, "min": 1, "max": 4096,
                         "tooltip": "For the stable_mask output: how many frames in a row a pixel must be still (in motion_mask) before it counts as stable. "
                                    "stable_mask starts white, goes black the moment a pixel moves, and returns to white after `hold` still frames, staying white until it moves again."}),
    }}



# ------------------------------------------------------------------ context carry-over

NEVER = 100000          # "this pixel has not moved for as long as we know"


def _state_path(name, clip_index):
    import re
    import folder_paths
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(name or "motion_ctx")) or "motion_ctx"
    d = os.path.join(folder_paths.get_output_directory(), "h3_pullforward")
    return d, os.path.join(d, f"{safe}_motion_state_{int(clip_index)}.npz")


def motion_masks_ctx(frames, threshold, compare, frame_gap, blur_sigma, despeckle, grow, linger, lookahead, feather, camera,
                     invert, fill_holes, tail_raw=None, prior_since=None):
    """Same as motion_masks, plus: tail_raw bool [K,H,W] is OR-ed into the raw motion of the first K frames, and prior_since int [H,W]
    (frames since the pixel last moved, measured at the frame just before frame 0) makes the linger carry on across the clip boundary.
    Returns (mask, raw_motion bool [B,H,W])."""
    frames = np.asarray(frames, dtype=np.float32)
    B, H, W = frames.shape[:3]
    if B < 2:
        z = np.zeros((B, H, W), np.float32)
        return (1 - z if invert else z), np.zeros((B, H, W), bool)
    rgb = frames[..., :3] if frames.shape[-1] >= 3 else np.repeat(frames[..., :1], 3, -1)
    luma = np.empty((B, H, W), np.float32)
    for i in range(B):
        luma[i] = blur(rgb[i] @ np.array([0.299, 0.587, 0.114], np.float32), blur_sigma)
    comp = camera == "translation"
    g = max(1, min(int(frame_gap), B - 1))
    if compare.startswith("first"):
        ch = np.stack([_pair_changed(luma[0], luma[t], threshold, comp) if t else np.zeros((H, W), bool) for t in range(B)])
        ch[0] = ch[1]
    else:
        P = [_pair_changed(luma[i], luma[i + g], threshold, comp) for i in range(B - g)]
        prev = [P[t - g] if t >= g else P[min(t, B - g - 1)] for t in range(B)]
        nxt = [P[t] if t < B - g else P[B - g - 1] for t in range(B)]
        ch = np.stack([prev[t] if compare == "previous frame" else (prev[t] | nxt[t]) for t in range(B)])
    m = ch.astype(np.uint8)
    CH = 16
    if despeckle > 0:
        for i in range(0, B, CH):
            m[i:i + CH] = dilate(erode(m[i:i + CH], despeckle), despeckle)
    if tail_raw is not None:                                           # motion the previous clip saw in these same frames
        k = min(len(tail_raw), B)
        m[:k] = np.maximum(m[:k], tail_raw[:k].astype(np.uint8))
    raw = m.astype(bool)
    if linger or lookahead:
        m = temporal_max(m, int(linger), int(lookahead))
    if prior_since is not None and linger:                             # linger carried over from before frame 0
        t = np.arange(B, dtype=np.int64)[:, None, None]
        carry = ((prior_since[None].astype(np.int64) + t + 1) <= int(linger)) & (prior_since[None] < NEVER)
        m = np.maximum(m, carry.astype(np.uint8))
    if fill_holes > 0:
        for i in range(0, B, CH):
            m[i:i + CH] = close_gaps(m[i:i + CH], int(fill_holes) // 2)
    out = np.empty((B, H, W), np.float32)
    for i in range(0, B, CH):
        c = dilate(m[i:i + CH], grow).astype(np.float32) if grow > 0 else m[i:i + CH].astype(np.float32)
        out[i:i + CH] = np.clip(blur(c, feather), 0, 1) if feather > 0 else c
    return (1 - out if invert else out), raw


def stability_mask_ctx(moving, hold, start_count=None):
    """Like stability_mask, but the still-run counter starts from start_count [H,W] (frames in a row the pixel was still just
    before frame 0) instead of from `hold`."""
    B, H, W = moving.shape
    hold = max(1, int(hold))
    count = np.full((H, W), hold, np.int32) if start_count is None else np.minimum(np.asarray(start_count, np.int32), hold)
    out = np.empty((B, H, W), np.float32)
    for t in range(B):
        count = np.where(moving[t], 0, np.minimum(count + 1, hold))
        out[t] = count >= hold
    return out


class H3ContextMotionMask:
    CATEGORY = "mask/motion"
    FUNCTION = "run"
    RETURN_TYPES = ("MASK", "IMAGE", "STRING", "MASK")
    RETURN_NAMES = ("motion_mask", "overlay", "report", "stable_mask")

    @classmethod
    def INPUT_TYPES(cls):
        base = MotionMask_INPUTS()
        base["required"]["hold"] = ("INT", {"default": 22, "min": 1, "max": 4096,
                                            "tooltip": "For the stable_mask output: how many frames in a row a pixel must be still before it counts as stable. "
                                                       "With context on, the count carries over from the previous clip."})
        base["optional"] = {
            "clip_index": ("INT", {"default": 1, "min": 1, "max": 100000, "tooltip": "Shot number. Shot 1 starts fresh; shot N loads what shot N-1 saved. Every shot saves its own state."}),
            "context_frames": ("INT", {"default": 22, "min": 0, "max": 512, "tooltip": "How many frames the clips share through Motion Context (22). 0 = turn the carry-over off."}),
            "context_name": ("STRING", {"default": "motion_ctx", "tooltip": "Name of the saved state files (output/h3_pullforward). Use a different name for a different scene."}),
        }
        return base

    def run(self, images, threshold, compare, frame_gap, blur, despeckle, fill_holes, grow, linger, lookahead, feather,
            camera_compensation, invert, hold=22, clip_index=1, context_frames=22, context_name="motion_ctx"):
        import torch
        frames = images.detach().cpu().float().numpy()
        B, H, W = frames.shape[:3]
        K = max(0, int(context_frames))
        notes = []
        tail_raw = prior_since = prior_still = None
        if K > 0 and int(clip_index) > 1:
            try:
                _, p = _state_path(context_name, int(clip_index) - 1)
                if os.path.exists(p):
                    z = np.load(p)
                    if tuple(z["since"].shape) == (H, W) and int(z["ctx"]) == K:
                        tail_raw, prior_since, prior_still = z["tail"], z["since"], z["still"]
                        notes.append(f"context: loaded clip {int(clip_index) - 1}'s motion for the shared {K} frames "
                                     f"({float(tail_raw.mean()) * 100:.1f}% moving) and its linger/hold counters.")
                    else:
                        notes.append("context: the saved state has a different size or frame count, so it was ignored.")
                else:
                    notes.append(f"context: no saved state for clip {int(clip_index) - 1} yet, so this clip started fresh.")
            except Exception as e:
                notes.append(f"context: could not read the saved state ({e}).")
        mask, raw = motion_masks_ctx(frames, threshold, compare, frame_gap, blur, despeckle, grow, linger, lookahead, feather,
                                     camera_compensation, invert, fill_holes, tail_raw, prior_since)
        ov = overlay_frames(frames, mask)
        moving = (mask < 0.5) if invert else (mask > 0.5)
        stable = stability_mask_ctx(moving, hold, prior_still)
        if K > 0 and B > K + 1:
            try:
                j = B - K - 1                                                  # the frame just before the shared frames start
                since = np.full((H, W), NEVER, np.int32)
                for t in range(j + 1):
                    since = np.where(raw[t], 0, np.minimum(since + 1, NEVER))
                still = np.zeros((H, W), np.int32)
                for t in range(j + 1):
                    still = np.where(moving[t], 0, np.minimum(still + 1, NEVER))
                d, p = _state_path(context_name, int(clip_index))
                os.makedirs(d, exist_ok=True)
                np.savez(p, tail=raw[B - K:], since=since, still=still, ctx=np.int32(K))
                notes.append(f"context: saved this clip's last {K} frames of motion and counters for clip {int(clip_index) + 1}.")
            except Exception as e:
                notes.append(f"context: could not save the state ({e}).")
        report = make_report(mask) + ("\n" + "\n".join(notes) if notes else "")
        return (torch.from_numpy(np.ascontiguousarray(mask)), torch.from_numpy(np.ascontiguousarray(ov)),
                report, torch.from_numpy(np.ascontiguousarray(stable)))


NODE_CLASS_MAPPINGS = {"H3ContextMotionMask": H3ContextMotionMask}
NODE_DISPLAY_NAME_MAPPINGS = {"H3ContextMotionMask": "H3 Context Motion Mask (Motion Mask that carries motion between clips)"}
