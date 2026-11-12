use crate::{CoreError, Counts, Geometry, Positions};

/// One ptychography scan: geometry, raw positions (as stored in CXI) and measured counts.
#[derive(Clone, Debug, PartialEq)]
pub struct Scan {
    pub geometry: Geometry,
    pub positions: Positions,
    pub counts: Counts,
}

impl Scan {
    pub fn new(
        geometry: Geometry,
        positions: Positions,
        counts: Counts,
    ) -> Result<Self, CoreError> {
        if counts.nframe() != positions.len() {
            return Err(CoreError::Shape(format!(
                "{} frames vs {} positions",
                counts.nframe(),
                positions.len()
            )));
        }
        if counts.frame_shape() != geometry.frame {
            return Err(CoreError::Shape(format!(
                "frame shape {:?} vs geometry {:?}",
                counts.frame_shape(),
                geometry.frame
            )));
        }
        Ok(Self {
            geometry,
            positions,
            counts,
        })
    }
}
