use std::ops::{Deref, DerefMut};

use crate::CoreError;

pub use num_complex::Complex64 as C64;

/// Dense row-major stack of 2D planes with shape `[n, ny, nx]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Stack<T> {
    dims: [usize; 3],
    data: Vec<T>,
}

impl<T: Clone> Stack<T> {
    pub fn filled(dims: [usize; 3], value: T) -> Self {
        Self {
            dims,
            data: vec![value; dims.iter().product()],
        }
    }
}

impl<T> Stack<T> {
    pub fn from_vec(dims: [usize; 3], data: Vec<T>) -> Result<Self, CoreError> {
        let want: usize = dims.iter().product();
        if data.len() != want {
            return Err(CoreError::Shape(format!(
                "stack {dims:?} needs {want} elements, got {}",
                data.len()
            )));
        }
        Ok(Self { dims, data })
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
    /// [`crate::view_corner`]).
    Object
);
