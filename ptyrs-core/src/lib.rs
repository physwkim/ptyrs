//! ptyrs data model (DESIGN §4, §5).
//!
//! Layout conventions shared by every crate:
//! - 2D arrays are row-major `[y, x]`; stacks are `[n, y, x]`.
//! - Positions are sample translations in metres, x horizontal (columns), y vertical (rows),
//!   as in PyNX and the CXI `translation` dataset.
//! - Measured counts are `u16` in detector order (zero frequency at `[ny/2, nx/2]`), with
//!   [`MASKED`] as the mask sentinel (invariant 3); there is no separate mask array.

mod addr;
mod array;
mod counts;
mod error;
pub mod frc;
mod geometry;
mod positions;
mod scan;
pub mod sim;

pub use addr::{AddrTable, View, required_obj_shape, view_corner};
pub use array::{C64, Object, Probe, Stack};
pub use counts::{Counts, MASKED, MAX_COUNT};
pub use error::CoreError;
pub use geometry::Geometry;
pub use positions::Positions;
pub use scan::Scan;
