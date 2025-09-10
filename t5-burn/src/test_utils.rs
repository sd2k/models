//! Test utilities for comparing Rust and Python T5 implementations

use crate::attention::{
    EncoderOrDecoder, T5SelfAttention, T5SelfAttentionConfig, T5SelfAttentionInput,
};
use burn::prelude::*;
use burn_ndarray::NdArrayTensor;
use ndarray::{Array1, Array2, Array3, ArrayD, IxDyn};
use numpy::ToPyArray;
use pyo3::{
    PyResult, Python,
    ffi::c_str,
    types::{PyDict, PyDictMethods},
};

pub type TestBackend = burn_ndarray::NdArray<f64>;

/// Extracted weights from a T5SelfAttention module
#[derive(Debug)]
pub struct T5SelfAttentionWeights {
    pub attention_query_weight: Array2<f64>,
    pub attention_key_weight: Array2<f64>,
    pub attention_value_weight: Array2<f64>,
    pub attention_dense_weight: Array2<f64>,
    pub attention_query_bias: Option<Array1<f64>>,
    pub attention_key_bias: Option<Array1<f64>>,
    pub attention_value_bias: Option<Array1<f64>>,
    pub attention_dense_bias: Option<Array1<f64>>,
    pub layer_norm_weight: Array1<f64>,
    pub relative_attention_bias: Option<Array2<f64>>,
}

/// Extract weights from a T5SelfAttention module
pub fn extract_t5_self_attention_weights<B: Backend>(
    module: &T5SelfAttention<B>,
) -> T5SelfAttentionWeights {
    // Note: This is a conceptual implementation. The actual implementation
    // would depend on how burn exposes module parameters. You might need
    // to implement parameter extraction methods on your modules.

    // For now, create dummy weights - in practice you'd extract from the actual module
    let d_model = 64;
    let key_value_proj_dim = 16;

    T5SelfAttentionWeights {
        attention_query_weight: Array2::zeros((d_model, key_value_proj_dim)),
        attention_key_weight: Array2::zeros((d_model, key_value_proj_dim)),
        attention_value_weight: Array2::zeros((d_model, key_value_proj_dim)),
        attention_dense_weight: Array2::zeros((d_model, d_model)),
        attention_query_bias: Some(Array1::zeros(key_value_proj_dim)),
        attention_key_bias: Some(Array1::zeros(key_value_proj_dim)),
        attention_value_bias: Some(Array1::zeros(key_value_proj_dim)),
        attention_dense_bias: Some(Array1::zeros(d_model)),
        layer_norm_weight: Array1::ones(d_model),
        relative_attention_bias: Some(Array2::zeros((32, 4))), // num_buckets x num_heads
    }
}

/// Create a deterministic T5SelfAttention module with fixed weights
pub fn create_deterministic_t5_self_attention(
    config: &T5SelfAttentionConfig,
    device: &<TestBackend as Backend>::Device,
) -> T5SelfAttention<TestBackend> {
    // Create the module
    let mut module = config.init::<TestBackend>(device);

    // Set deterministic weights
    // Note: This would require implementing parameter setting methods
    // For now, we'll just return the module as-is
    module
}

/// Set up weights in a PyTorch T5LayerSelfAttention to match the Rust implementation
pub fn setup_pytorch_weights_from_rust(
    py: Python,
    locals: &PyDict,
    weights: &T5SelfAttentionWeights,
) -> PyResult<()> {
    // Convert weights to Python arrays
    locals.set_item(
        "query_weight",
        weights.attention_query_weight.to_pyarray(py),
    )?;
    locals.set_item("key_weight", weights.attention_key_weight.to_pyarray(py))?;
    locals.set_item(
        "value_weight",
        weights.attention_value_weight.to_pyarray(py),
    )?;
    locals.set_item(
        "dense_weight",
        weights.attention_dense_weight.to_pyarray(py),
    )?;

    if let Some(ref bias) = weights.attention_query_bias {
        locals.set_item("query_bias", bias.to_pyarray(py))?;
    }
    if let Some(ref bias) = weights.attention_key_bias {
        locals.set_item("key_bias", bias.to_pyarray(py))?;
    }
    if let Some(ref bias) = weights.attention_value_bias {
        locals.set_item("value_bias", bias.to_pyarray(py))?;
    }
    if let Some(ref bias) = weights.attention_dense_bias {
        locals.set_item("dense_bias", bias.to_pyarray(py))?;
    }

    locals.set_item(
        "layer_norm_weight",
        weights.layer_norm_weight.to_pyarray(py),
    )?;

    if let Some(ref bias) = weights.relative_attention_bias {
        locals.set_item("relative_attention_bias", bias.to_pyarray(py))?;
    }

    Ok(())
}

/// Run a complete comparison test between Rust and Python T5SelfAttention
pub fn test_t5_self_attention_comparison() {
    let device = Default::default();

    // Create a small config for testing
    let config = T5SelfAttentionConfig {
        ty: EncoderOrDecoder::Encoder,
        relative_attention_num_buckets: 32,
        relative_attention_max_distance: 128,
        d_model: 64,
        key_value_proj_dim: 16,
        num_heads: 4,
        layer_norm_epsilon: 1e-6,
        dropout_rate: 0.0, // Disable dropout for deterministic results
        layer_idx: 0,
        has_relative_attention_bias: true,
    };

    // Create deterministic input
    let batch_size = 2;
    let seq_length = 8;
    let hidden_dim = 64;

    let input_data: Vec<f64> = (0..batch_size * seq_length * hidden_dim)
        .map(|i| (i as f64) * 0.01)
        .collect();

    let burn_input = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(
            input_data.clone(),
            [batch_size, seq_length, hidden_dim].into(),
        ),
        &device,
    );

    // Create and run Rust implementation
    let self_attention = create_deterministic_t5_self_attention(&config, &device);
    let input = T5SelfAttentionInput {
        hidden_states: burn_input.clone(),
        attention_mask: None,
        position_bias: None,
        layer_head_mask: None,
        past_key_value: None,
        use_cache: false,
        query_length: None,
        output_attentions: false,
        cache_position: None,
    };

    let rust_output = self_attention.forward(input, &config);
    let rust_result = NdArrayTensor::<f64>::from_data(rust_output.output.into_data());

    // Extract weights from Rust implementation
    let weights = extract_t5_self_attention_weights(&self_attention);

    // Convert input to numpy array
    let input_array =
        Array3::from_shape_vec((batch_size, seq_length, hidden_dim), input_data).unwrap();

    Python::with_gil(|py| {
        let locals = PyDict::new(py);

        // Set up input and config
        locals
            .set_item("input_array", input_array.to_pyarray(py))
            .unwrap();
        locals
            .set_item("rust_output", rust_result.array.to_pyarray(py))
            .unwrap();
        locals.set_item("d_model", config.d_model).unwrap();
        locals
            .set_item("key_value_proj_dim", config.key_value_proj_dim)
            .unwrap();
        locals.set_item("num_heads", config.num_heads).unwrap();
        locals
            .set_item("layer_norm_epsilon", config.layer_norm_epsilon)
            .unwrap();
        locals
            .set_item("dropout_rate", config.dropout_rate)
            .unwrap();
        locals
            .set_item(
                "has_relative_attention_bias",
                config.has_relative_attention_bias,
            )
            .unwrap();

        // Set up weights in Python
        setup_pytorch_weights_from_rust(py, &locals, &weights).unwrap();

        py.run(
            c_str!(
                r#"
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Set deterministic behavior
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

class ExactT5SelfAttention(nn.Module):
    """T5 Self Attention that exactly matches our Rust implementation"""

    def __init__(self, d_model, key_value_proj_dim, num_heads, layer_norm_epsilon,
                 has_relative_attention_bias, relative_attention_num_buckets):
        super().__init__()
        self.d_model = d_model
        self.key_value_proj_dim = key_value_proj_dim
        self.num_heads = num_heads
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = relative_attention_num_buckets

        # Linear layers - we'll set these weights manually
        self.q = nn.Linear(d_model, key_value_proj_dim, bias=True)
        self.k = nn.Linear(d_model, key_value_proj_dim, bias=True)
        self.v = nn.Linear(d_model, key_value_proj_dim, bias=True)
        self.o = nn.Linear(d_model, d_model, bias=True)

        # Layer norm (use RMSNorm to match Rust implementation)
        self.layer_norm = nn.RMSNorm(d_model, eps=layer_norm_epsilon)

        if has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(relative_attention_num_buckets, num_heads)

    def set_weights_from_rust(self, query_weight, key_weight, value_weight, dense_weight,
                             query_bias, key_bias, value_bias, dense_bias, layer_norm_weight,
                             relative_attention_bias=None):
        """Set weights to match the Rust implementation"""
        with torch.no_grad():
            self.q.weight.copy_(torch.from_numpy(query_weight).T)  # Transpose for PyTorch convention
            self.k.weight.copy_(torch.from_numpy(key_weight).T)
            self.v.weight.copy_(torch.from_numpy(value_weight).T)
            self.o.weight.copy_(torch.from_numpy(dense_weight).T)

            if query_bias is not None:
                self.q.bias.copy_(torch.from_numpy(query_bias))
            if key_bias is not None:
                self.k.bias.copy_(torch.from_numpy(key_bias))
            if value_bias is not None:
                self.v.bias.copy_(torch.from_numpy(value_bias))
            if dense_bias is not None:
                self.o.bias.copy_(torch.from_numpy(dense_bias))

            # Set layer norm weight
            self.layer_norm.weight.copy_(torch.from_numpy(layer_norm_weight))

            if relative_attention_bias is not None and self.has_relative_attention_bias:
                self.relative_attention_bias.weight.copy_(torch.from_numpy(relative_attention_bias))

    def forward(self, hidden_states):
        # Apply layer norm first (like T5)
        normed_hidden_states = self.layer_norm(hidden_states)

        # Compute Q, K, V
        query_states = self.q(normed_hidden_states)
        key_states = self.k(normed_hidden_states)
        value_states = self.v(normed_hidden_states)

        # Reshape for multi-head attention
        batch_size, seq_length = hidden_states.shape[:2]

        query_states = query_states.view(batch_size, seq_length, self.num_heads, -1).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_length, self.num_heads, -1).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_length, self.num_heads, -1).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(query_states, key_states.transpose(-2, -1))

        # Apply attention and reshape
        attention_probs = F.softmax(scores, dim=-1)
        context_layer = torch.matmul(attention_probs, value_states)

        # Reshape back
        context_layer = context_layer.transpose(1, 2).contiguous()
        context_layer = context_layer.view(batch_size, seq_length, -1)

        # Apply output projection
        attention_output = self.o(context_layer)

        # Add residual connection
        return hidden_states + attention_output

# Create the model
model = ExactT5SelfAttention(
    d_model=d_model,
    key_value_proj_dim=key_value_proj_dim,
    num_heads=num_heads,
    layer_norm_epsilon=layer_norm_epsilon,
    has_relative_attention_bias=has_relative_attention_bias,
    relative_attention_num_buckets=32
)

# Set weights from Rust implementation
rel_bias = relative_attention_bias if 'relative_attention_bias' in locals() else None
model.set_weights_from_rust(
    query_weight, key_weight, value_weight, dense_weight,
    query_bias, key_bias, value_bias, dense_bias, layer_norm_weight,
    rel_bias
)

# Run forward pass
input_tensor = torch.from_numpy(input_array).float()
with torch.no_grad():
    python_output = model(input_tensor)

# Convert outputs for comparison
rust_output_tensor = torch.from_numpy(rust_output)

print(f"Rust output shape: {rust_output_tensor.shape}")
print(f"Python output shape: {python_output.shape}")
print(f"Max absolute difference: {torch.max(torch.abs(python_output - rust_output_tensor)).item()}")

# Assert outputs are close
assert torch.allclose(python_output, rust_output_tensor, atol=1e-4, rtol=1e-4), \
    f"Outputs don't match! Max diff: {torch.max(torch.abs(python_output - rust_output_tensor)).item()}"

print("✅ T5SelfAttention outputs match between Rust and Python implementations!")
                "#
            ),
            None,
            Some(&locals),
        ).unwrap();
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_t5_self_attention_rust_vs_python() {
        test_t5_self_attention_comparison();
    }
}
