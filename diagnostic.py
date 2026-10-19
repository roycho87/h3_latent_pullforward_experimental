"""H3 Latent Pull-Forward DIAGNOSTIC  --  stand-alone node, separate from the Pull-Forward pack.

Ignores motion completely and pastes the FIRST block of a MiniMax H3 latent (5 latent frames = 17 video frames) over every block of the clip.
If pasting works, every block decodes to the same picture, so the clip must not degrade. Chained through Motion Context it also shows
whether the clip-to-clip hand-off changes anything. Drop this file into ComfyUI/custom_nodes/ and restart ComfyUI.
GPL-3.0.
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


def _reference_path(name):
    import folder_paths
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(name or "diag_clip1_block")) or "diag_clip1_block"
    d = os.path.join(folder_paths.get_output_directory(), "h3_pullforward")
    return d, os.path.join(d, safe + ".npz")


class H3LatentPullForwardDiagnostic:
    """DIAGNOSTIC. Ignores motion completely: pastes the FIRST block of the latent (or just its first latent frame) over the whole clip.
    If pasting works, every block decodes to the same picture, so the clip must not degrade. If it still degrades, the cause is not the latent."""
    CATEGORY = "mask/motion"
    FUNCTION = "run"
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "report")
    SRC_OWN = "this clip's first block"
    SRC_SAVED = "clip 1's first block (saved)"
    PAT_BLOCK = "repeat first block (block-aware)"
    PAT_FRAME = "repeat one latent frame only"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "latent": ("LATENT",),
            "source": ([cls.SRC_OWN, cls.SRC_SAVED], {"default": cls.SRC_OWN,
                       "tooltip": "this clip's first block = each clip repeats ITS OWN first block, so any change that crosses from one clip to the next shows up (the real test). "
                                  "clip 1's first block (saved) = every clip repeats the SAME saved block from clip 1 (control: nothing can differ in the latent)."}),
            "pattern": ([cls.PAT_BLOCK, cls.PAT_FRAME], {"default": cls.PAT_BLOCK,
                         "tooltip": "repeat first block = latent frame g gets the first block's frame (g mod block_latents), which keeps H3's 5-frame block structure. "
                                    "repeat one latent frame only = the latent frame picked by frame_index (0-based, inside the first block) is copied into EVERY position of the clip. Done entirely in latent space. Slot 0 of a block decodes to 1 video frame, slots 1-4 to 4 each, so try 0 and 1 and compare."}),
            "block_latents": ("INT", {"default": 5, "min": 1, "max": 64}),
            "clip_index": ("INT", {"default": 1, "min": 1, "max": 100000, "tooltip": "Shot number. Shot 1 saves its first block; later shots can use it and report how far their own first block has drifted from it."}),
            "reference_name": ("STRING", {"default": "diag_clip1_block"}),
            "frame_index": ("INT", {"default": 0, "min": 0, "max": 63, "tooltip": "Only used by 'repeat one latent frame only': which latent frame of the first block (0 = first) to copy everywhere."}),
        }, "optional": {
            "anchor_latent": ("LATENT", {"tooltip": "Optional. A ready-made block to paste instead of the clip's own first block, e.g. the output of 'H3 Still Block Encode' (one picture, VAE-encoded, so it really is a still). Overrides 'source'. Must have the same shape as the clip's first block."}),
        }}

    def run(self, latent, source, pattern, block_latents, clip_index, reference_name, frame_index=0, anchor_latent=None):
        import torch
        import comfy.nested_tensor
        video, audio, nested = _unpack(latent)
        v = video.unsqueeze(0) if video.ndim == 4 else video            # [B,C,F,H,W]
        F = int(v.shape[2])
        bl = max(1, min(int(block_latents), F))
        fi = max(0, min(int(frame_index), bl - 1))
        own = v[:, :, :bl].clone()
        notes = []
        saved = None
        try:
            _, p = _reference_path(reference_name)
            if os.path.exists(p):
                saved = torch.from_numpy(np.load(p)["block"]).to(device=v.device, dtype=v.dtype)
        except Exception as e:
            notes.append(f"could not read the saved block ({e}).")
        anchor = own
        if int(clip_index) == 1:
            try:
                d, p = _reference_path(reference_name)
                os.makedirs(d, exist_ok=True)
                np.savez(p, block=own.detach().float().cpu().numpy())
                notes.append("clip 1: saved its first block.")
            except Exception as e:
                notes.append(f"could not save the first block ({e}).")
        elif source == self.SRC_SAVED and anchor_latent is None:
            if saved is not None and tuple(saved.shape) == tuple(own.shape):
                anchor = saved
                notes.append("using clip 1's saved first block.")
            else:
                notes.append("no matching saved clip 1 block, so this clip used its own first block.")
        if anchor_latent is not None:
            ext, _, _ = _unpack(anchor_latent)
            ext = ext.unsqueeze(0) if ext.ndim == 4 else ext
            ext = ext[:, :, :bl].to(device=v.device, dtype=v.dtype)
            if tuple(ext.shape[1:2] + ext.shape[3:]) != tuple(own.shape[1:2] + own.shape[3:]) or int(ext.shape[2]) != bl:
                raise ValueError(f"anchor_latent has shape {tuple(ext.shape)} but the clip's first block is {tuple(own.shape)}; they must match in channels, size and block length.")
            anchor = ext.expand(own.shape[0], -1, -1, -1, -1) if ext.shape[0] != own.shape[0] else ext
            source = "external anchor_latent"
            notes.append("pasting the anchor_latent block instead of the clip's own first block.")
        if int(clip_index) > 1 and saved is not None and tuple(saved.shape) == tuple(own.shape):
            d = (own.float() - saved.float()).abs().mean().item()
            m = saved.float().abs().mean().item() + 1e-12
            notes.append(f"this clip's own first block differs from clip 1's by mean abs {d:.5f} ({d / m * 100:.2f}% of clip 1's mean magnitude); 0 = identical.")
        if pattern == self.PAT_BLOCK:
            idx = torch.arange(F, device=v.device) % bl
            out = anchor[:, :, idx].clone()
        else:
            out = anchor[:, :, fi:fi + 1].expand(-1, -1, F, -1, -1).clone()
        # self-check: every pasted frame must equal its source frame exactly
        ref = anchor[:, :, (torch.arange(F, device=v.device) % bl)] if pattern == self.PAT_BLOCK else anchor[:, :, fi:fi + 1].expand(-1, -1, F, -1, -1)
        worst = (out.float() - ref.float()).abs().max().item()
        report = (f"DIAGNOSTIC paste: {F} latent frames all set from {'the first block of ' + str(bl) + ' latent frames' if pattern == self.PAT_BLOCK else f'latent frame {fi} of the first block'} "
                  f"({source}). Largest difference between a pasted frame and its source: {worst}. " + " ".join(notes))
        print("[H3 Pull-Forward Diagnostic] " + report)
        out = out.to(device=v.device, dtype=v.dtype)
        if video.ndim == 4:
            out = out.squeeze(0)
        new = dict(latent)
        new["samples"] = comfy.nested_tensor.NestedTensor((out, audio)) if nested else out
        new.pop("noise_mask", None)
        return (new, report)


class H3StillBlockEncode:
    """Makes a latent block that is genuinely a STILL picture: takes one image (or, if none is connected, the clip's own first latent frame),
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
            "cache_name": ("STRING", {"default": "diag_still_block"}),
        }, "optional": {
            "image": ("IMAGE", {"tooltip": "Optional. A picture to hold still. Leave it EMPTY to use the clip's own first latent frame (which decodes to exactly one video frame)."}),
        }}

    def run(self, vae, size_like, block_latents, clip_index, cache_name, image=None):
        import torch
        import torch.nn.functional as TF
        bl = max(1, int(block_latents))
        n = 4 * (bl - 1) + 1
        ref, _, _ = _unpack(size_like)
        h, w = int(ref.shape[-2]), int(ref.shape[-1])
        d, p = _reference_path(cache_name or "diag_still_block")
        if int(clip_index) > 1 and os.path.exists(p):
            try:
                lat = torch.from_numpy(np.load(p)["block"])
                if lat.ndim == 5 and int(lat.shape[2]) == bl and tuple(lat.shape[3:]) == (h, w):
                    lat = lat.to(device=ref.device, dtype=ref.dtype)
                    report = f"STILL block: loaded the saved still block from shot 1 {tuple(lat.shape)}. Nothing was decoded or encoded."
                    print("[H3 Still Block Encode] " + report)
                    return ({"samples": lat}, report)
                notes_bad = f"saved block {tuple(lat.shape)} does not match this clip ({bl} frames, {h}x{w}); rebuilding it. "
            except Exception as e:
                notes_bad = f"could not read the saved block ({e}); rebuilding it. "
        else:
            notes_bad = ""
        notes = [notes_bad] if notes_bad else []
        if image is None:
            blk = ref.unsqueeze(0) if ref.ndim == 4 else ref
            blk = blk[:, :, :1]                      # latent frame 0 alone: it decodes to exactly ONE video frame
            dec = vae.decode(blk)
            dec = dec.reshape(-1, dec.shape[-3], dec.shape[-2], dec.shape[-1])
            image = dec[0:1]
            notes.append("decoded the clip's first latent frame (it decodes to a single video frame) and used that picture.")
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
        print("[H3 Still Block Encode] " + report)
        return ({"samples": lat}, report)


NODE_CLASS_MAPPINGS = {"H3LatentPullForwardDiagnostic": H3LatentPullForwardDiagnostic, "H3StillBlockEncode": H3StillBlockEncode}
NODE_DISPLAY_NAME_MAPPINGS = {"H3LatentPullForwardDiagnostic": "H3 Latent Pull-Forward DIAGNOSTIC (paste first block everywhere)", "H3StillBlockEncode": "H3 Still Block Encode (one picture -> still latent block)"}
