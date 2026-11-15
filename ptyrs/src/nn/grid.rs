//! The launch grid of every nn kernel. CUDA launches at most 65535 cubes along a grid's Y and Z, so
//! a kernel that laid a dimension of its own on Y or Z failed at launch once that dimension passed
//! 65535. An nn kernel names its cube by the linear `CUBE_POS` alone and is launched on [`grid`],
//! which may hold more cubes than asked; a cube past those asked for leaves at once.

use burn_cubecl::cubecl::client::Client;
use burn_cubecl::cubecl::prelude::CubeCount;

/// A grid of at least `cubes` cubes within the limits of `client`'s device, filling X first.
pub(super) fn grid(client: &Client, cubes: usize) -> CubeCount {
    let (most_x, most_y, _) = client.properties().hardware.max_cube_count;
    let cubes = u32::try_from(cubes).expect("a launch of fewer than 2³² cubes");
    // CUDA's X takes 2³¹ - 1 cubes, so `rows` is at most 3 and the grid one deep.
    let rows = cubes.div_ceil(most_x);
    CubeCount::Static(cubes.min(most_x), rows.min(most_y), rows.div_ceil(most_y))
}

/// Whether a tensor of `shape` holds fewer than 2³² elements: the kernels index tensors with 32
/// bits, and cubecl's launch panics with no message past that.
pub(super) fn fits(shape: &[usize]) -> bool {
    let len = shape.iter().try_fold(1usize, |p, &d| p.checked_mul(d));
    len.is_some_and(|len| u32::try_from(len).is_ok())
}

/// Panics unless each tensor of `shapes`, the shapes of every tensor a launch reads or writes,
/// [`fits`]. A launch site calls it before it packs or allocates anything.
pub(super) fn check_elements(shapes: &[&[usize]]) {
    assert!(
        shapes.iter().all(|s| fits(s)),
        "a launch of fewer than 2³² elements per tensor"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn just_under_2_32_elements_fit() {
        check_elements(&[&[65535, 65537]]);
    }

    #[test]
    #[should_panic(expected = "a launch of fewer than 2³² elements per tensor")]
    fn exactly_2_32_elements_do_not() {
        check_elements(&[&[65535, 65537], &[1 << 16, 1 << 16]]);
    }

    #[test]
    #[should_panic(expected = "a launch of fewer than 2³² elements per tensor")]
    fn an_element_count_past_usize_does_not_fit() {
        check_elements(&[&[1 << 32, 1 << 32]]);
    }
}
