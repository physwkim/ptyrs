"""Float64 reference of pty-chi's free-space propagators (propagate.py) on fixed inputs.

    micromamba run -n ptyrs-ref python -I refs/f64/propagators.py <dir>

Writes <dir>/propagator_select_f64.txt, one row per geometry,

    ny nx wavelength py px distance kind fresnel_number scale

with kind forward_models.PlanarPtychographyForwardModel.build_free_space_propagator's choice
(0 FourierPropagator, 1 AngularSpectrumPropagator, 2 FresnelTransformPropagator),
fresnel_number WavefieldPropagatorParameters.fresnel_number (0 for an infinite distance) and
scale get_intensity_scaling_factor of the chosen propagator. The geometries straddle the
selection threshold on each axis, for both distance signs and anisotropic pixels.

Writes <dir>/propagator_cases_f64.txt, one row per field case,

    case direct ny nx wavelength py px distance kind scale

with direct 1 for an AngularSpectrumPropagator built whatever the selection (the in-object
propagator), else 0 for the free-space choice; and <dir>/propagator_fields_f64.txt, one row
per case and pixel in natural (centred) row-major order,

    case iy ix x_re x_im y_re y_im fwd_re fwd_im bwd_re bwd_im

where x and y are complex uniform in [-1, 1) per part (numpy default_rng(case)), fwd =
propagate_forward(x) and bwd = propagate_backward(y).
"""
import sys

import numpy as np
import torch

import ptychi.utils as utils
from ptychi.propagate import (AngularSpectrumPropagator, FourierPropagator,
                              FresnelTransformPropagator, WavefieldPropagatorParameters,
                              get_intensity_scaling_factor)

FMT = "%.17e"
SHAPE = (12, 16)
# (direct, wavelength, py, px, distance): Fourier, angular spectrum with and without an
# evanescent band and at a negative distance with anisotropic pixels, Fresnel at both signs,
# and angular spectrum built where the free-space rule picks Fresnel.
CASES = [
    (0, 1e-10, 50e-9, 50e-9, np.inf),
    (0, 1e-10, 50e-9, 50e-9, 1e-4),
    (0, 5e-7, 3e-7, 3e-7, 1.5e-6),
    (0, 1e-10, 60e-9, 45e-9, -2e-4),
    (0, 1e-10, 50e-9, 50e-9, 1.25e-3),
    (0, 1e-10, 40e-9, 50e-9, -1.6e-3),
    (1, 1e-10, 50e-9, 50e-9, 0.05),
]


def params(shape, wavelength, py, px, distance):
    return WavefieldPropagatorParameters.create_simple(
        wavelength_m=wavelength, width_px=shape[1], height_px=shape[0],
        pixel_width_m=px, pixel_height_m=py, propagation_distance_m=distance)


def free_space(shape, wavelength, py, px, distance):
    """build_free_space_propagator, from the probe shape and pixel."""
    if distance == np.inf:
        return FourierPropagator()
    p = params(shape, wavelength, py, px, distance)
    if p.is_fresnel_transform_preferrable():
        return FresnelTransformPropagator(p)
    return AngularSpectrumPropagator(p)


def kind(prop):
    return [FourierPropagator, AngularSpectrumPropagator,
            FresnelTransformPropagator].index(type(prop))


def select_rows():
    rows = []
    for shape in [(12, 16), (16, 12), (32, 32)]:
        for py, px in [(50e-9, 50e-9), (40e-9, 50e-9), (60e-9, 45e-9)]:
            wavelength = 1e-10
            w, h = px / wavelength, py / wavelength
            # Distances putting w²/|z| or h²/|z| just either side of 1/nx or 1/ny.
            zs = [np.inf]
            for side, n in ((w, shape[1]), (h, shape[0])):
                for f in (0.5, 0.999, 1.001, 2.0):
                    zs.append(side * side * n * f * wavelength)
            zs += [1e-6, 1e-2]
            for z in zs:
                for sign in ((1,) if z == np.inf else (1, -1)):
                    d = sign * z
                    prop = free_space(shape, wavelength, py, px, d)
                    fr = params(shape, wavelength, py, px, d).fresnel_number
                    rows.append([shape[0], shape[1], wavelength, py, px, d, kind(prop), fr,
                                 get_intensity_scaling_factor(prop, shape)])
    return np.array(rows)


def main(d):
    torch.set_default_dtype(torch.float64)
    utils.set_default_complex_dtype(torch.complex128)
    np.savetxt(f"{d}/propagator_select_f64.txt", select_rows(), fmt=FMT)

    cases, fields = [], []
    ny, nx = SHAPE
    iy, ix = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    for c, (direct, wavelength, py, px, distance) in enumerate(CASES):
        prop = (AngularSpectrumPropagator(params(SHAPE, wavelength, py, px, distance)) if direct
                else free_space(SHAPE, wavelength, py, px, distance))
        rng = np.random.default_rng(c)
        x, y = (rng.uniform(-1, 1, SHAPE) + 1j * rng.uniform(-1, 1, SHAPE) for _ in range(2))
        fwd = prop.propagate_forward(torch.from_numpy(x)).numpy()
        bwd = prop.propagate_backward(torch.from_numpy(y)).numpy()
        cases.append([c, direct, ny, nx, wavelength, py, px, distance, kind(prop),
                      get_intensity_scaling_factor(prop, SHAPE)])
        fields.append(np.stack([np.full(SHAPE, c), iy, ix, x.real, x.imag, y.real, y.imag,
                                fwd.real, fwd.imag, bwd.real, bwd.imag], -1).reshape(-1, 11))
    np.savetxt(f"{d}/propagator_cases_f64.txt", np.array(cases), fmt=FMT)
    np.savetxt(f"{d}/propagator_fields_f64.txt", np.concatenate(fields), fmt=FMT)


if __name__ == "__main__":
    main(sys.argv[1])
