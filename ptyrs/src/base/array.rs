use std::ops::{Deref, DerefMut};

use crate::base::CoreError;

pub use num_complex::Complex64 as C64;

/// Dense row-major stack of 2D planes with shape `[n, ny, nx]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Stack<T> {
    dims: [usize; 3],
    data: Vec<T>,
}

/// Element count of `dims`, the plane first: with `n = 0` an overflowing plane still fails, so
/// [`Stack::plane_len`] is in range for every stack.
fn element_count(dims: [usize; 3]) -> Option<usize> {
    dims[1].checked_mul(dims[2])?.checked_mul(dims[0])
}

impl<T: Clone> Stack<T> {
    /// # Panics
    ///
    /// If the element count passes `usize`, as `vec!` does when the allocation size does.
    pub fn filled(dims: [usize; 3], value: T) -> Self {
        let len = element_count(dims)
            .unwrap_or_else(|| panic!("stack {dims:?} has more than usize elements"));
        Self {
            dims,
            data: vec![value; len],
        }
    }
}

impl<T> Stack<T> {
    pub fn from_vec(dims: [usize; 3], data: Vec<T>) -> Result<Self, CoreError> {
        match element_count(dims) {
            Some(want) if want == data.len() => Ok(Self { dims, data }),
            Some(want) => Err(CoreError::Shape(format!(
                "stack {dims:?} needs {want} elements, got {}",
                data.len()
            ))),
            None => Err(CoreError::Shape(format!(
                "stack {dims:?} needs more than usize elements"
            ))),
        }
    }

    pub fn dims(&self) -> [usize; 3] {
        self.dims
    }

    /// Number of planes.
    pub fn len(&self) -> usize {
        self.dims[0]
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// `[ny, nx]` of one plane.
    pub fn plane_shape(&self) -> [usize; 2] {
        [self.dims[1], self.dims[2]]
    }

    pub fn plane_len(&self) -> usize {
        self.dims[1] * self.dims[2]
    }

    pub fn plane(&self, k: usize) -> &[T] {
        let n = self.plane_len();
        &self.data[k * n..(k + 1) * n]
    }

    pub fn plane_mut(&mut self, k: usize) -> &mut [T] {
        let n = self.plane_len();
        &mut self.data[k * n..(k + 1) * n]
    }

    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    pub fn into_vec(self) -> Vec<T> {
        self.data
    }
}

macro_rules! complex_stack {
    ($(#[$doc:meta])* $name:ident) => {
        $(#[$doc])*
        #[derive(Clone, Debug, PartialEq)]
        pub struct $name(pub Stack<C64>);

        impl $name {
            pub fn new(stack: Stack<C64>) -> Self {
                Self(stack)
            }

            /// Number of modes (planes).
            pub fn nmode(&self) -> usize {
                self.0.len()
            }
        }

        impl Deref for $name {
            type Target = Stack<C64>;
            fn deref(&self) -> &Stack<C64> {
                &self.0
            }
        }

        impl DerefMut for $name {
            fn deref_mut(&mut self) -> &mut Stack<C64> {
                &mut self.0
            }
        }
    };
}

complex_stack!(
    /// Probe modes `[nprobe, ny, nx]` in natural (centred) real-space order; plane shape equals
    /// the detector frame shape.
    Probe
);
complex_stack!(
    /// Object modes `[nobj, nyo, nxo]`; the scan is centred on `[nyo/2, nxo/2]` (see
    /// [`crate::base::view_corner`]).
    Object
);

#[cfg(test)]
mod tests {
    use super::*;

    /// A wrapped product would match the empty buffer: `[0, 2^32, 2^32]` and `[2^62, 2, 2]`.
    #[test]
    fn from_vec_rejects_element_count_past_usize() {
        let side = 1 << (usize::BITS / 2);
        for dims in [[0, side, side], [1 << (usize::BITS - 2), 2, 2]] {
            let e = Stack::<u8>::from_vec(dims, Vec::new()).unwrap_err();
            assert!(e.to_string().contains("more than usize"), "{dims:?}: {e}");
        }
        assert!(Stack::<u8>::from_vec([0, 3, 5], Vec::new()).is_ok());
    }

    #[test]
    #[should_panic(expected = "more than usize elements")]
    fn filled_panics_on_element_count_past_usize() {
        let side = 1 << (usize::BITS / 2);
        Stack::filled([0, side, side], 0u8);
    }
}
