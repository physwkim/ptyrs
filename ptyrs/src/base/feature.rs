/// A modelled quantity's values and whether a reconstruction refines them. Held values feed
/// the model wherever refined ones do; only a refined quantity's sums, derivatives and step
/// run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Feature<T> {
    pub values: T,
    pub refine: bool,
}

impl<T> Feature<T> {
    /// `values`, refined.
    pub fn refined(values: T) -> Self {
        Self {
            values,
            refine: true,
        }
    }

    /// `values`, held.
    pub fn held(values: T) -> Self {
        Self {
            values,
            refine: false,
        }
    }

    /// The values to step: some while refined.
    pub fn stepped(&mut self) -> Option<&mut T> {
        self.refine.then_some(&mut self.values)
    }

    /// The same refine over `f` of the values.
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Feature<U> {
        Feature {
            values: f(self.values),
            refine: self.refine,
        }
    }

    /// The same refine over a borrow of the values.
    pub fn as_ref(&self) -> Feature<&T> {
        Feature {
            values: &self.values,
            refine: self.refine,
        }
    }
}
