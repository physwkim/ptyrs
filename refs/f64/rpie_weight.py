"""The rPIE object step weight of the engine's Ptycho::rpie, shared by the numpy models and the
pty-chi runs it is checked against. With p_km frame k's probe mode m in the batch:

    weight_km = conj(p_km) / ((1 - a) d_km + a max_{k, r} sum_m d_km)
    d_km = max(0, Re E_k(D_m)),  D_m = sum_k P_k(|p_km|^2)

P_k places frame k's window as the object step places it and E_k reads it back: integer
windows under pty-chi's PATCH, its Fourier sampling and that sampling's adjoint under
FULL_OBJECT, D_m complex until read back. One frame per batch has d_km = |p_km|^2, pty-chi's
RPIEReconstructor.calculate_object_step_weight; at more, that weight's batch sum overshoots
where the frames overlap.
"""


def weight(p, a, overlap):
    """The weight for the (frames, modes, h, w) probes p (numpy or torch), `overlap(p2)` the
    read-back D_m of the per-frame p2 = |p|^2 (complex), the same shape."""
    d = overlap(abs(p) ** 2 + 0j).real.clip(min=0)
    return p.conj() / ((1 - a) * d + a * d.sum(1).max())


def scatter_overlap(corners, shape):
    """numpy `overlap` of integer windows: frame k's at corners[k] of a `shape` plane."""
    import numpy as np

    def overlap(p2):
        h, w = p2.shape[-2:]
        d = np.zeros((p2.shape[1], *shape), p2.dtype)
        for (y, x), pk in zip(corners, p2, strict=True):
            d[:, y:y + h, x:x + w] += pk
        return np.stack([d[:, y:y + h, x:x + w] for y, x in corners])

    return overlap


def patch_ptychi(rec):
    """Replaces the pty-chi RPIEReconstructor `rec`'s object step weight by `weight`, D_m placed
    and read back as rec's object step places (place_object_patches_on_probe_grid with
    integer_mode=True, which FULL_OBJECT samples by Fourier) at the batch's positions."""
    import torch

    fm = rec.forward_model
    forward = fm.forward
    batch = {}

    def stash(indices, *args, **kwargs):
        batch["indices"] = indices
        return forward(indices, *args, **kwargs)

    def overlap(p2):
        pos = rec.parameter_group.probe_positions.tensor[batch["indices"].cpu()]
        return torch.stack([
            fm.extract_probe_grid_patches(
                fm.place_object_patches_on_probe_grid(pos, p2[:, m], integer_mode=True),
                pos, p2.shape[-2:], integer_mode=True)
            for m in range(p2.shape[1])], 1)

    fm.forward = stash
    rec.calculate_object_step_weight = lambda p: weight(
        p, rec.parameter_group.object.options.alpha, overlap)
