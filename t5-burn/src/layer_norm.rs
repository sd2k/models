use burn::{
    config::Config,
    nn::{RmsNorm, RmsNormConfig},
    prelude::*,
};

/// Configuration for T5LayerNorm
#[derive(Config, Debug)]
pub struct T5LayerNormConfig {
    /// Model dimension
    pub d_model: usize,
    /// Layer norm epsilon
    pub layer_norm_epsilon: f64,
}

impl T5LayerNormConfig {
    /// Initialize the T5LayerNorm module
    pub fn init<B: Backend>(&self, device: &B::Device) -> T5LayerNorm<B> {
        T5LayerNorm {
            layer_norm: RmsNormConfig::new(self.d_model)
                .with_epsilon(self.layer_norm_epsilon)
                .init(device),
        }
    }
}
/// T5LayerNorm implementation using RMS normalization
#[derive(Module, Debug)]
pub struct T5LayerNorm<B: Backend> {
    layer_norm: RmsNorm<B>,
}

impl<B: Backend> T5LayerNorm<B> {
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        self.layer_norm.forward(input)
    }
}

#[cfg(test)]
mod test_t5_layer_norm {
    use burn::prelude::*;
    use burn_ndarray::NdArrayTensor;
    use ndarray::Array2;
    use numpy::ToPyArray;
    use pyo3::{
        Python,
        ffi::c_str,
        types::{PyDict, PyDictMethods},
    };

    type Backend = burn_ndarray::NdArray<f64>;

    use super::T5LayerNormConfig;

    #[test]
    fn t5_layer_norm() {
        let device = Default::default();
        let raw = (0..9).map(f64::from).collect::<Vec<_>>();
        let ndarray_input = Array2::from_shape_vec((3, 3), raw.clone()).unwrap();
        let burn_input = Tensor::<_, 2>::from_data(TensorData::new(raw, [3, 3]), &device);

        let layer_norm = T5LayerNormConfig {
            d_model: 3,
            layer_norm_epsilon: 1e-5,
        }
        .init::<Backend>(&device);
        let output = layer_norm.forward(burn_input);
        let output = NdArrayTensor::<f64>::from_data(output.into_data());

        Python::with_gil(|py| {
            let pyarray = ndarray_input.to_pyarray(py);
            let locals = PyDict::new(py);
            locals.set_item("input", pyarray).unwrap();
            locals
                .set_item("output", output.array.to_pyarray(py))
                .unwrap();
            py.run(
                c_str!(
                    r#"
from transformers.models.t5.modeling_t5 import T5LayerNorm
import torch

layer_norm = T5LayerNorm(3, 1e-5)
input = torch.from_numpy(locals()["input"])
output = torch.from_numpy(locals()["output"])
expected = layer_norm(input)
assert torch.allclose(expected, output, atol=1e-4)
                "#
                ),
                None,
                Some(&locals),
            )
            .unwrap();
        });
    }
}
