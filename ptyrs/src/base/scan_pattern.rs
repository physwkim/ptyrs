use std::num::NonZeroUsize;

/// The order in which a scan visits its positions, as live batching needs it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScanPattern {
    /// Row-major raster, every row `row` points long and scanned in the same direction.
    Raster { row: NonZeroUsize },
    /// Fermat spiral, or any scan without rows.
    Spiral,
}
