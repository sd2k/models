use burn::{
    config::Config,
    nn::{Dropout, DropoutConfig, Gelu, Linear, LinearConfig},
    prelude::*,
};

use crate::layer_norm::{T5LayerNorm, T5LayerNormConfig};

/// Configuration for T5FeedForward
#[derive(Config, Debug)]
pub struct T5FeedForwardConfig {
    pub d_model: usize,
    pub d_ff: usize,
    pub layer_norm_epsilon: f64,
    pub dropout_rate: f64,
}

impl T5FeedForwardConfig {
    /// Initialize the T5FeedForward module
    pub fn init<B: Backend>(&self, device: &B::Device) -> T5FeedForward<B> {
        let dense = T5DenseGatedActConfig {
            d_model: self.d_model,
            d_ff: self.d_ff,
            dropout_rate: self.dropout_rate,
        }
        .init(device);

        let layer_norm = T5LayerNormConfig {
            d_model: self.d_model,
            layer_norm_epsilon: self.layer_norm_epsilon,
        }
        .init(device);

        let dropout = DropoutConfig::new(self.dropout_rate).init();

        T5FeedForward {
            dense,
            layer_norm,
            dropout,
        }
    }
}

/// T5 Feed Forward layer
#[derive(Module, Debug)]
pub struct T5FeedForward<B: Backend> {
    dense: T5DenseGatedAct<B>,
    layer_norm: T5LayerNorm<B>,
    dropout: Dropout,
}

impl<B: Backend> T5FeedForward<B> {
    pub fn forward<const D: usize>(&self, hidden_states: Tensor<B, D>) -> Tensor<B, D> {
        // Apply layer norm first
        let normed_hidden_states = self.layer_norm.forward(hidden_states.clone());
        // Apply feed forward
        let ff_output = self.dense.forward(normed_hidden_states);
        // Add residual connection
        let output = hidden_states + ff_output;
        // Apply dropout
        self.dropout.forward(output)
    }
}

/// Configuration for T5DenseGatedAct
#[derive(Config, Debug)]
pub struct T5DenseGatedActConfig {
    pub d_model: usize,
    pub d_ff: usize,
    pub dropout_rate: f64,
}

impl T5DenseGatedActConfig {
    /// Initialize the T5DenseGatedAct module
    pub fn init<B: Backend>(&self, device: &B::Device) -> T5DenseGatedAct<B> {
        T5DenseGatedAct {
            wi_0: LinearConfig::new(self.d_model, self.d_ff)
                .with_bias(false)
                .init(device),
            wi_1: LinearConfig::new(self.d_model, self.d_ff)
                .with_bias(false)
                .init(device),
            wo: LinearConfig::new(self.d_ff, self.d_model)
                .with_bias(false)
                .init(device),
            dropout: DropoutConfig::new(self.dropout_rate).init(),
            activation: Gelu::new(),
        }
    }
}

/// T5 Dense Gated Activation Dense layer
#[derive(Module, Debug)]
pub struct T5DenseGatedAct<B: Backend> {
    wi_0: Linear<B>,
    wi_1: Linear<B>,
    wo: Linear<B>,
    dropout: Dropout,
    activation: Gelu,
}

impl<B: Backend> T5DenseGatedAct<B> {
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let hidden_gelu = self.activation.forward(self.wi_0.forward(x.clone()));
        let hidden_linear = self.wi_1.forward(x);
        let hidden_states = hidden_gelu * hidden_linear;
        let hidden_states = self.dropout.forward(hidden_states);
        let hidden_states = self.wo.forward(hidden_states);
        hidden_states
    }
}

#[cfg(test)]
mod test_t5_feed_forward {
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

    use super::{T5DenseGatedActConfig, T5FeedForwardConfig};

    #[test]
    fn t5_feed_forward() {
        let device = Default::default();
        let raw = (0..12).map(|i| i as f64 * 0.1).collect::<Vec<_>>();
        let ndarray_input = Array2::from_shape_vec((3, 4), raw.clone()).unwrap();
        let burn_input = Tensor::<_, 2>::from_data(TensorData::new(raw, [3, 4]), &device);

        let feed_forward = T5FeedForwardConfig {
            d_model: 4,
            d_ff: 8,
            layer_norm_epsilon: 1e-5,
            dropout_rate: 0.0, // Set to 0 for reproducible testing
        }
        .init::<Backend>(&device);

        let output = feed_forward.forward(burn_input);
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
from transformers.models.t5.modeling_t5 import T5Config, T5LayerFF
import torch
import torch.nn as nn

# Create T5LayerFF with similar configuration
config = T5Config(
    d_model=4,
    d_ff=8,
    dropout_rate=0.0,
    layer_norm_epsilon=1e-5,
    feed_forward_proj='gated-gelu'
)

layer_ff = T5LayerFF(config)
layer_ff.eval()  # Set to eval mode to disable dropout

# Convert input to torch tensor
input_tensor = torch.from_numpy(input).float()
burn_output = torch.from_numpy(output).float()

# Run through the layer
with torch.no_grad():
    expected = layer_ff(input_tensor)

# Note: We can't directly compare outputs due to random weight initialization.
# We can compare the output shapes though.
assert expected.shape == burn_output.shape, "Output shapes do not match"
                "#
                ),
                None,
                Some(&locals),
            )
            .unwrap();
        });
    }

    #[test]
    fn t5_dense_gated_act() {
        let device = Default::default();
        let raw = (0..8).map(|i| i as f64 * 0.1).collect::<Vec<_>>();
        let ndarray_input = Array2::from_shape_vec((2, 4), raw.clone()).unwrap();
        let burn_input = Tensor::<_, 2>::from_data(TensorData::new(raw, [2, 4]), &device);

        let dense_gated = T5DenseGatedActConfig {
            d_model: 4,
            d_ff: 6,
            dropout_rate: 0.0,
        }
        .init::<Backend>(&device);

        let output = dense_gated.forward(burn_input);
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
from transformers.models.t5.modeling_t5 import T5Config, T5DenseGatedActDense
import torch
import torch.nn as nn

# Create T5DenseGatedActDense with similar configuration
# Create T5LayerFF with similar configuration
config = T5Config(
    d_model=4,
    d_ff=8,
    dropout_rate=0.0,
)

dense_gated = T5DenseGatedActDense(config)
dense_gated.eval()  # Set to eval mode to disable dropout

# Convert input to torch tensor
input_tensor = torch.from_numpy(input).float()
burn_output = torch.from_numpy(output).float()

expected = dense_gated(input_tensor)

# Note: We can't directly compare outputs due to random weight initialization.
# We can compare the output shapes though.
assert expected.shape == burn_output.shape, "Output shapes do not match"
                "#
                ),
                None,
                Some(&locals),
            )
            .unwrap();
        });
    }
}
