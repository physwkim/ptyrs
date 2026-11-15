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
