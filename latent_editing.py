"""H3 Pull-Forward LATENT EDITING  --  stand-alone node, separate from the Pull-Forward pack.

Builds a latent block that is genuinely a STILL picture and hands it to Pull-Forward's `anchor_latent`, so the still parts of the
background are replaced by a clean still instead of by whatever the sampler produced. Shot 1 builds the block (the clip's own first
latent frame is decoded to one picture, repeated, VAE-encoded) and saves it; later shots load it. Drop this file into
ComfyUI/custom_nodes/ and restart ComfyUI. GPL-3.0.
"""
import os
import re

import numpy as np


def _unpack(latent):
    s = latent["samples"]
    if hasattr(s, "is_nested") and s.is_nested:
        video, audio = s.unbind()
        return video, audio, True
    if isinstance(s, (list, tuple)):
        return s[0], s[1], True
    return s, None, False


RAW_MASK_CUTOFF = 0.5


def _starts(n_src, n_dst):
    return np.minimum((np.arange(n_dst) * n_src) // n_dst, n_src - 1).astype(np.intp)


def _stable_to_latent(mask, F, H, W):
    """stable mask (1 = stable) -> bool [F,H,W]. Latent-sized masks are used as they are; a pixel mask is reduced so a latent cell is
    stable only if every pixel it covers is stable in every frame of its group (same rule as Pull-Forward)."""
    m = np.asarray(mask, np.float32)
    if m.ndim == 2:
        m = m[None]
    if m.shape == (F, H, W):
        return m > RAW_MASK_CUTOFF
    b = (m > RAW_MASK_CUTOFF).astype(np.uint8)
    Bp, Hp, Wp = b.shape
    if Hp < H or Wp < W:
        raise ValueError(f"the stable mask ({Hp}x{Wp}) is smaller than the latent grid ({H}x{W})")
    b = np.minimum.reduceat(b, _starts(Hp, H), axis=1)
    b = np.minimum.reduceat(b, _starts(Wp, W), axis=2)
    b = np.minimum.reduceat(b, _starts(Bp, F), axis=0)
    return b.astype(bool)


def _reference_path(name):
    import folder_paths
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(name or "latent_editing_still_block")) or "latent_editing_still_block"
    d = os.path.join(folder_paths.get_output_directory(), "h3_pullforward")
    return d, os.path.join(d, safe + ".npz")


class H3LatentEditingStillBlock:
    """Makes a latent block that is genuinely a STILL picture: takes one image (or, if none is connected, a picture assembled from the clip's own first STABLE cells),
    repeats it to fill one H3 block (block_latents latent frames = 4*(block_latents-1)+1 video frames) and runs it through the VAE encoder.
    The result (slot 0 = the picture, slots 1-4 = four identical-picture chunks) is what the decoder expects for a static scene.
    Shot 1 builds the block and SAVES it; every later shot just LOADS it, so the decode/encode round trip happens exactly once.
    Feed it to the diagnostic node's anchor_latent. 'size_like' is the clip's latent (e.g. the sampler output); the picture is resized to match it."""
    CATEGORY = "mask/motion"
    FUNCTION = "run"
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("still_block", "report")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "vae": ("VAE",),
            "size_like": ("LATENT", {"tooltip": "The clip's latent. Its width/height set the encode size, and (when no image is connected) its first latent frame is the picture that gets held still."}),
            "block_latents": ("INT", {"default": 5, "min": 1, "max": 64}),
            "clip_index": ("INT", {"default": 1, "min": 1, "max": 100000, "tooltip": "Shot number. Shot 1 builds the still block and saves it; later shots load the saved block (nothing is decoded or encoded again)."}),
            "cache_name": ("STRING", {"default": "latent_editing_still_block"}),
        }, "optional": {
            "image": ("IMAGE", {"tooltip": "Optional. A picture to hold still. Leave it EMPTY to build the picture from the clip itself (see stable_mask)."}),
            "stable_mask": ("MASK", {"tooltip": "The same stable_mask Pull-Forward uses (white = stable). With it, the still picture is assembled cell by cell from each cell's FIRST STABLE value (so something that is only in the first frames, like a hand, never ends up in it). Without it, the clip's first latent frame is used."}),
            "min_stable_run": ("INT", {"default": 10, "min": 1, "max": 64, "tooltip": "A cell's source frame is the first block-start where it becomes stable AND STAYS stable for at least this many latent frames (or until the end of the clip, if less is left). A cell that is only still for a moment and then moves again is skipped. 10 latent frames is about 34 video frames."}),
            "skip_first_latent_frames": ("INT", {"default": 0, "min": 0, "max": 64, "tooltip": "Latent frames at the start that are never used as a source (e.g. 7 for the Motion Context overlap if you ever build this on shot 2 or later)."}),
        }}

    def run(self, vae, size_like, block_latents, clip_index, cache_name, image=None, stable_mask=None, min_stable_run=10, skip_first_latent_frames=0):
        import torch
        import torch.nn.functional as TF
        bl = max(1, int(block_latents))
        n = 4 * (bl - 1) + 1
        ref, _, _ = _unpack(size_like)
        h, w = int(ref.shape[-2]), int(ref.shape[-1])
        d, p = _reference_path(cache_name or "latent_editing_still_block")
        if int(clip_index) > 1 and os.path.exists(p):
            try:
                lat = torch.from_numpy(np.load(p)["block"])
                if lat.ndim == 5 and int(lat.shape[2]) == bl and tuple(lat.shape[3:]) == (h, w):
                    lat = lat.to(device=ref.device, dtype=ref.dtype)
                    report = f"STILL block: loaded the saved still block from shot 1 {tuple(lat.shape)}. Nothing was decoded or encoded."
                    print("[H3 Latent Editing: Still Block] " + report)
                    return ({"samples": lat}, report)
                notes_bad = f"saved block {tuple(lat.shape)} does not match this clip ({bl} frames, {h}x{w}); rebuilding it. "
            except Exception as e:
                notes_bad = f"could not read the saved block ({e}); rebuilding it. "
        else:
            notes_bad = ""
        notes = [notes_bad] if notes_bad else []
        if image is None:
            blk = ref.unsqueeze(0) if ref.ndim == 4 else ref
            if stable_mask is not None:
                vid = blk[0].detach().float().cpu().numpy()                       # [C,F,H,W]
                C_, F_, H_, W_ = vid.shape
                st = _stable_to_latent(stable_mask.detach().float().cpu().numpy(), F_, H_, W_)
                starts = [g for g in range(int(skip_first_latent_frames), F_) if g % bl == 0]   # only a block's first latent frame decodes as a single picture
                if not starts:
                    raise ValueError("no block-start latent frame is available as a source")
                comp = np.zeros((C_, H_, W_), np.float32)
                run = np.zeros((F_ + 1, H_, W_), np.int32)                       # run[g] = how many latent frames in a row this cell stays stable from g on
                for g in range(F_ - 1, -1, -1):
                    run[g] = np.where(st[g], run[g + 1] + 1, 0)
                req = int(min_stable_run)
                got = np.zeros((H_, W_), bool)
                src = np.full((H_, W_), -1, np.int32)
                for g in starts:                                                    # 1st choice: becomes stable AND stays stable
                    m = st[g] & (run[g] >= min(req, F_ - g)) & ~got
                    if m.any():
                        comp[:, m] = vid[:, g][:, m]
                        src[m] = g
                        got |= m
                strict = float(got.mean())
                for g in starts:                                                    # 2nd choice: first block-start where it is stable at all
                    m = st[g] & ~got
                    if m.any():
                        comp[:, m] = vid[:, g][:, m]
                        src[m] = g
                        got |= m
                miss = ~got
                if miss.any():
                    comp[:, miss] = vid[:, starts[0]][:, miss]
                used = src[got]
                notes.append(f"assembled the still cell by cell from each cell's first stable block-start frame: {strict * 100:.1f}% of cells became stable and stayed stable ({req} latent frames or to the end), "
                             f"{(float(got.mean()) - strict) * 100:.1f}% were only briefly stable (used their first stable frame); source latent frames {int(used.min()) if used.size else '-'}..{int(used.max()) if used.size else '-'}; "
                             f"the other {float(miss.mean()) * 100:.1f}% never stable at a block start, so they use latent frame {starts[0]}.")
                blk = torch.from_numpy(comp)[None, :, None].to(device=ref.device, dtype=ref.dtype)   # [1,C,1,H,W]
            else:
                blk = blk[:, :, :1]                      # latent frame 0 alone: it decodes to exactly ONE video frame
                notes.append("no stable_mask connected: used the clip's first latent frame.")
            dec = vae.decode(blk)
            dec = dec.reshape(-1, dec.shape[-3], dec.shape[-2], dec.shape[-1])
            image = dec[0:1]
            notes.append("decoded that latent frame (a block's first latent frame decodes to a single video frame) and used that picture.")
        img = image[0:1].movedim(-1, 1).float()                       # [1,3,H,W]
        try:
            f = int(vae.spacial_compression_encode())
        except Exception:
            f = 16
            notes.append("could not read the VAE's spatial compression, assumed 16.")
        th, tw = h * f, w * f
        if tuple(img.shape[-2:]) != (th, tw):
            img = TF.interpolate(img, size=(th, tw), mode="bicubic", align_corners=False).clamp(0, 1)
            notes.append(f"resized the picture to {tw}x{th}.")
        tried = []
        raw = None
        for nn in (n, 33, 65, 97, 129):
            nn = max(int(nn), n)
            frames = img.movedim(1, -1).expand(nn, -1, -1, -1).contiguous()   # [nn,H,W,3], every frame identical
            raw = vae.encode(frames[:, :, :, :3])
            raw = raw.unsqueeze(0) if raw.ndim == 4 else raw
            if raw.ndim == 5 and int(raw.shape[0]) != 1 and int(raw.shape[2]) == 1:
                raw = raw.movedim(0, 2)
            tried.append(f"{nn} frames -> {int(raw.shape[2])} latent frames")
            if int(raw.shape[2]) >= bl:
                break
        T = int(raw.shape[2])
        spread = (raw[:, :, 1:].float() - raw[:, :, 1:2].float()).abs().max().item() if T > 2 else 0.0
        if T >= bl:
            lat = raw[:, :, :bl]
        else:
            lat = torch.cat([raw, raw[:, :, -1:].expand(-1, -1, bl - T, -1, -1)], dim=2)
            notes.append(f"the encoder returned only {T} latent frames, so the last one was repeated to fill {bl} slots.")
        try:
            os.makedirs(d, exist_ok=True)
            np.savez(p, block=lat.detach().float().cpu().numpy())
            notes.append("saved the still block for the later shots.")
        except Exception as e:
            notes.append(f"could not save the still block ({e}).")
        report = (f"STILL block: encoder results [{'; '.join(tried)}]. Final block {tuple(lat.shape)}. "
                  f"Largest difference between the encoded static slots (1..): {spread:.5f} (near 0 = a true still; only measurable when 3+ latent frames come back). " + " ".join(notes))
        print("[H3 Latent Editing: Still Block] " + report)
        return ({"samples": lat}, report)


NODE_CLASS_MAPPINGS = {"H3LatentEditingStillBlock": H3LatentEditingStillBlock}
NODE_DISPLAY_NAME_MAPPINGS = {"H3LatentEditingStillBlock": "H3 Latent Editing: Still Block (encode once, paste via Pull-Forward)"}
