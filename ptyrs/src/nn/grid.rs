//! The launch grid of every nn kernel. CUDA launches at most 65535 cubes along a grid's Y and Z, so
//! a kernel that laid a dimension of its own on Y or Z failed at launch once that dimension passed
//! 65535. An nn kernel names its cube by the linear `CUBE_POS` alone and is launched on [`grid`],
//! which may hold a few more cubes than asked but none past 2³², so the 32-bit `CUBE_POS` names
//! each cube once and a cube past those asked for leaves at once.

use burn::prelude::Tensor;
use burn::tensor::DType;
use burn_cubecl::cubecl::client::Client;
use burn_cubecl::cubecl::prelude::CubeCount;
use burn_cubecl::tensor::CubeTensor;

/// A grid of at least `cubes` cubes within the limits of `client`'s device, spread evenly over the
/// fewest rows along X.
pub(super) fn grid(client: &Client, cubes: usize) -> CubeCount {
    let (most_x, most_y, _) = client.properties().hardware.max_cube_count;
    let cubes = u32::try_from(cubes).expect("a launch of fewer than 2³² cubes");
    // CUDA's X takes 2³¹ - 1 cubes, so `rows` is at most 3 and the grid one deep. Spread evenly
    // over the rows, the cubes pass `cubes` by fewer than `rows`, so none is past 2³²: 3 rows
    // come only of 2³² - 1 cubes, which 3 divides.
    let rows = cubes.div_ceil(most_x).max(1);
    CubeCount::Static(
        cubes.div_ceil(rows),
        rows.min(most_y),
        rows.div_ceil(most_y),
    )
}

/// Whether a tensor of `shape` holds fewer than 2³² elements: the kernels index tensors with 32
/// bits, and cubecl's launch panics with no message past that.
pub(super) fn fits(shape: &[usize]) -> bool {
    let len = shape.iter().try_fold(1usize, |p, &d| p.checked_mul(d));
    len.is_some_and(|len| u32::try_from(len).is_ok())
}

/// A tensor as [`check_operands`] reads it.
pub(super) trait Operand {
    fn shape(&self) -> Vec<usize>;
    fn float(&self) -> DType;
}

impl Operand for CubeTensor {
    fn shape(&self) -> Vec<usize> {
        self.meta.shape().to_vec()
    }

    fn float(&self) -> DType {
        self.dtype
    }
}

impl<const D: usize> Operand for Tensor<D> {
    fn shape(&self) -> Vec<usize> {
        self.dims().to_vec()
    }

    fn float(&self) -> DType {
        self.dtype()
    }
}

/// Panics unless each of `operands`, every tensor read with `x` that `x` does not make, has the
/// shape paired with it and `x`'s float type. The kernels index an operand as of that shape and
/// read it as `x`'s float type; burn's elementwise ops and convolutions launch in their first
/// operand's float type without checking the others', and its elementwise ops broadcast a size-1
/// axis. A module's operands are checked at its entry, before it picks a kernel or burn:
/// [`Conv::forward`], [`Conv::fold`], [`norm_relu`] and the predictor's standardisation.
///
/// [`Conv::forward`]: crate::nn::conv::Conv::forward
/// [`Conv::fold`]: crate::nn::conv::Conv::fold
/// [`norm_relu`]: crate::nn::norm_relu::norm_relu
pub(super) fn check_operands(x: &dyn Operand, operands: &[(&dyn Operand, &[usize])]) {
    for &(t, shape) in operands {
        assert_eq!(
            t.shape(),
            shape,
            "an operand of the shape its kernels index"
        );
        assert_eq!(t.float(), x.float(), "an operand in the input's float type");
    }
}

/// Panics unless each tensor of `shapes`, the shapes of every tensor a launch reads or writes,
/// [`fits`]. A launch site calls it before it packs or allocates anything.
pub(super) fn check_elements(shapes: &[&[usize]]) {
    assert!(
        shapes.iter().all(|s| fits(s)),
        "a launch of fewer than 2³² elements per tensor"
    );
}

/// The client of the GPU the tests run on.
#[cfg(test)]
pub(super) fn client() -> Client {
    Tensor::<1>::empty([1], &crate::nn::cuda(0))
        .try_into_primitive::<burn_cubecl::CubeBackend>()
        .unwrap()
        .client
}

/// Runs `launch`, a test's launch of a module's probe of its kernel helpers, with an output
/// buffer of `len` values on [`client`]'s GPU, and reads them back.
#[cfg(test)]
pub(super) fn run_probe(
    len: usize,
    launch: impl FnOnce(&Client, burn_cubecl::cubecl::prelude::BufferArg),
) -> Vec<u32> {
    use burn_cubecl::cubecl::prelude::{BufferArg, CubeElement};
    let client = client();
    let out = client.empty(len * size_of::<u32>());
    // SAFETY: the buffer holds `len` values.
    launch(&client, unsafe {
        BufferArg::from_raw_parts(out.clone(), len)
    });
    u32::from_bytes(&client.read_one(out).unwrap()).to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Up to 2³² - 1 cubes the grid holds fewer more than asked than it has rows, and none past
    /// 2³², so the 32-bit `CUBE_POS` names each cube once. Three rows of 2³¹ - 1 cubes for
    /// 2³² - 1 named cubes 0 .. 2³¹ - 3 twice.
    #[test]
    fn each_cube_position_names_one_cube() {
        let client = client();
        let most_x = u64::from(client.properties().hardware.max_cube_count.0);
        let tops = [
            1,
            most_x - 1,
            most_x,
            most_x + 1,
            2 * most_x,
            2 * most_x + 1,
        ];
        for cubes in tops.into_iter().filter(|&c| c <= u64::from(u32::MAX)) {
            let CubeCount::Static(x, y, z) = grid(&client, cubes as usize) else {
                unreachable!("a static grid");
            };
            let held = u64::from(x) * u64::from(y) * u64::from(z);
            let past = held.checked_sub(cubes);
            assert!(
                past.is_some_and(|p| p < u64::from(y * z)) && held <= 1 << 32,
                "{cubes} cubes on {x}×{y}×{z}"
            );
        }
    }

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
