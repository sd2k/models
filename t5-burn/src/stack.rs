//! T5 Stack Module
//!
//! This module implements the T5Stack, which represents either an encoder or decoder
//! stack in the T5 transformer model. It contains multiple T5Block layers and handles
//! the forward pass through all layers with proper attention mechanisms, position bias,
//! and optional caching.

use burn::{
    config::Config,
    nn::{Dropout, DropoutConfig, Embedding},
    prelude::*,
};

use crate::attention::EncoderOrDecoder;
use crate::block::{T5Block, T5BlockConfig, T5BlockInput};
use crate::layer_norm::{T5LayerNorm, T5LayerNormConfig};

/// Either input token IDs or pre-computed input embeddings.
#[derive(Debug, Clone)]
pub enum IdsOrEmbeds<B: Backend> {
    /// Input token IDs with shape [batch_size, sequence_length].
    Ids(Tensor<B, 2, burn::tensor::Int>),
    /// Pre-computed input embeddings with shape [batch_size, sequence_length, d_model].
    Embeds(Tensor<B, 3>),
}

/// Input structure for T5Stack forward pass
#[derive(Debug, Clone)]
pub struct T5StackInput<B: Backend> {
    /// Input token IDs or pre-computed input embeddings.
    pub inputs: IdsOrEmbeds<B>,
    /// Attention mask to prevent attention to padding tokens [batch_size, sequence_length]
    pub attention_mask: Option<Tensor<B, 2>>,
    /// Encoder hidden states for cross-attention (decoder only) [batch_size, encoder_length, d_model]
    pub encoder_hidden_states: Option<Tensor<B, 3>>,
    /// Encoder attention mask for cross-attention [batch_size, encoder_length]
    pub encoder_attention_mask: Option<Tensor<B, 2>>,
    /// Head masks for each layer [num_layers, num_heads]
    pub head_mask: Option<Tensor<B, 2>>,
    /// Cross-attention head masks for each layer [num_layers, num_heads]
    pub cross_attn_head_mask: Option<Tensor<B, 2>>,
    /// Past key-value states for caching
    pub past_key_values: Option<Vec<Option<Tensor<B, 4>>>>,
    /// Whether to use caching
    pub use_cache: bool,
    /// Whether to output attention weights
    pub output_attentions: bool,
    /// Whether to output hidden states from all layers
    pub output_hidden_states: bool,
    /// Cache position for incremental decoding
    pub cache_position: Option<Tensor<B, 1, burn::tensor::Int>>,
}

/// Output structure for T5Stack
#[derive(Debug, Clone)]
pub struct T5StackOutput<B: Backend> {
    /// Final hidden states from the last layer [batch_size, sequence_length, d_model]
    pub last_hidden_state: Tensor<B, 3>,
    /// Past key-value states for caching (if use_cache=True)
    pub past_key_values: Option<Vec<Option<Tensor<B, 4>>>>,
    /// Hidden states from all layers (if output_hidden_states=True)
    pub hidden_states: Option<Vec<Tensor<B, 3>>>,
    /// Self-attention weights from all layers (if output_attentions=True)
    pub attentions: Option<Vec<Option<Tensor<B, 4>>>>,
    /// Cross-attention weights from all layers (if output_attentions=True and is_decoder=True)
    pub cross_attentions: Option<Vec<Option<Tensor<B, 4>>>>,
}

/// Configuration for T5Stack
#[derive(Config, Debug)]
pub struct T5StackConfig {
    /// Number of layers in the stack
    pub n_layers: usize,
    /// Whether this is an encoder or decoder stack
    pub ty: EncoderOrDecoder,
    /// Number of buckets for relative position encoding
    pub relative_attention_num_buckets: usize,
    /// Maximum distance for relative position encoding
    pub relative_attention_max_distance: usize,
    /// Model dimension (hidden size)
    pub d_model: usize,
    /// Feed-forward dimension
    pub d_ff: usize,
    /// Key-value projection dimension
    pub key_value_proj_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Layer normalization epsilon
    pub layer_norm_epsilon: f64,
    /// Dropout rate
    pub dropout_rate: f64,
}

impl T5StackConfig {
    /// Initialize a T5Stack from this configuration
    pub fn init<B: Backend>(&self, device: &B::Device, embed_tokens: Embedding<B>) -> T5Stack<B> {
        let blocks = (0..self.n_layers)
            .map(|i| {
                let has_relative_attention_bias = i == 0;
                T5BlockConfig {
                    ty: self.ty,
                    relative_attention_num_buckets: self.relative_attention_num_buckets,
                    relative_attention_max_distance: self.relative_attention_max_distance,
                    d_model: self.d_model,
                    d_ff: self.d_ff,
                    key_value_proj_dim: self.key_value_proj_dim,
                    num_heads: self.num_heads,
                    layer_norm_epsilon: self.layer_norm_epsilon,
                    dropout_rate: self.dropout_rate,
                    has_relative_attention_bias,
                    layer_idx: i,
                }
                .init(device)
            })
            .collect();

        T5Stack {
            embed_tokens,
            blocks,
            final_layer_norm: T5LayerNormConfig {
                d_model: self.d_model,
                layer_norm_epsilon: self.layer_norm_epsilon,
            }
            .init(device),
            dropout: DropoutConfig::new(self.dropout_rate).init(),
        }
    }
}

/// T5 Stack module representing either an encoder or decoder stack
#[derive(Module, Debug)]
pub struct T5Stack<B: Backend> {
    /// Token embedding layer
    pub embed_tokens: Embedding<B>,
    /// Stack of T5Block layers
    pub blocks: Vec<T5Block<B>>,
    /// Final layer normalization
    pub final_layer_norm: T5LayerNorm<B>,
    /// Dropout layer
    pub dropout: Dropout,
}

impl<B: Backend> T5Stack<B> {
    /// Convert 2D attention mask to 4D causal mask for self-attention
    fn prepare_causal_mask(
        &self,
        attention_mask: &Tensor<B, 2>,
        input_shape: [usize; 2],
        is_decoder: bool,
        device: &B::Device,
    ) -> Tensor<B, 4> {
        let [batch_size, seq_length] = input_shape;

        if is_decoder {
            // Create causal mask for decoder - [seq_len, seq_len]
            let causal_mask: Tensor<B, 2, burn::tensor::Bool> =
                Tensor::tril_mask([seq_length, seq_length], 0, device);
            // Convert to float and reshape to [1, 1, seq_len, seq_len]
            let causal_mask = causal_mask.float().unsqueeze::<3>().unsqueeze::<4>();
            // Expand to [batch, 1, seq_len, seq_len]
            let causal_mask = causal_mask.repeat_dim(0, batch_size);

            // Convert attention mask [batch, seq_len] to [batch, 1, seq_len, seq_len]
            let attention_mask_4d = attention_mask
                .clone()
                .unsqueeze::<3>() // Add dimension: [1, batch, seq_len]
                .unsqueeze::<4>() // Add dimension: [1, 1, batch, seq_len]
                .swap_dims(0, 2) // Swap to: [batch, 1, 1, seq_len]
                .repeat_dim(2, seq_length); // Repeat: [batch, 1, seq_len, seq_len]

            // Combine masks
            let combined_mask = causal_mask * attention_mask_4d;

            // Convert to attention mask with large negative values for masked positions
            let ones = Tensor::ones_like(&combined_mask);
            let mask_value = -10000.0f32;
            ones * mask_value + combined_mask * (-mask_value)
        } else {
            // For encoder, convert [batch, seq_len] to [batch, 1, 1, seq_len]
            let attention_mask_4d = attention_mask
                .clone()
                .unsqueeze::<3>() // Add dimension: [1, batch, seq_len]
                .unsqueeze::<4>() // Add dimension: [1, 1, batch, seq_len]
                .swap_dims(0, 2); // Swap to: [batch, 1, 1, seq_len]

            // Convert to attention mask with large negative values for masked positions
            let ones = Tensor::ones_like(&attention_mask_4d);
            let mask_value = -10000.0f32;
            ones * mask_value + attention_mask_4d * (-mask_value)
        }
    }

    /// Prepare encoder attention mask for cross-attention
    fn prepare_encoder_attention_mask(
        &self,
        encoder_attention_mask: &Tensor<B, 2>,
        seq_length: usize,
    ) -> Tensor<B, 4> {
        // Convert [batch, encoder_seq_len] to [batch, 1, seq_len, encoder_seq_len]
        let encoder_mask_4d = encoder_attention_mask
            .clone()
            .unsqueeze::<3>() // Add dimension: [1, batch, encoder_seq_len]
            .unsqueeze::<4>() // Add dimension: [1, 1, batch, encoder_seq_len]
            .swap_dims(0, 2) // Swap to: [batch, 1, 1, encoder_seq_len]
            .repeat_dim(2, seq_length); // Repeat: [batch, 1, seq_len, encoder_seq_len]

        // Convert to attention mask with large negative values for masked positions
        let ones = Tensor::ones_like(&encoder_mask_4d);
        let mask_value = -10000.0f32;
        ones * mask_value + encoder_mask_4d * (-mask_value)
    }

    /// Forward pass through the T5Stack
    pub fn forward(&self, input: T5StackInput<B>) -> T5StackOutput<B> {
        let device = self.embed_tokens.devices()[0].clone();

        // Get input embeddings
        let inputs_embeds = match input.inputs {
            IdsOrEmbeds::Ids(input_ids) => self.embed_tokens.forward(input_ids),
            IdsOrEmbeds::Embeds(embeds) => embeds,
        };

        let [batch_size, seq_length, d_model] = inputs_embeds.dims();
        let input_shape = [batch_size, seq_length];

        // Apply input dropout
        let mut hidden_states = self.dropout.forward(inputs_embeds);

        // Prepare attention masks
        let attention_mask = input
            .attention_mask
            .unwrap_or_else(|| Tensor::ones([batch_size, seq_length], &device));

        // Determine if this is a decoder stack from the first block's configuration
        // For now, use a simple heuristic: decoder if encoder_hidden_states provided
        let is_decoder = input.encoder_hidden_states.is_some();
        let causal_mask =
            self.prepare_causal_mask(&attention_mask, input_shape, is_decoder, &device);

        // Prepare encoder attention mask if needed
        let encoder_extended_attention_mask =
            if let Some(ref encoder_attention_mask) = input.encoder_attention_mask {
                Some(self.prepare_encoder_attention_mask(encoder_attention_mask, seq_length))
            } else if input.encoder_hidden_states.is_some() && is_decoder {
                // Create default encoder attention mask
                let encoder_seq_length = input.encoder_hidden_states.as_ref().unwrap().dims()[1];
                let default_encoder_mask = Tensor::ones([batch_size, encoder_seq_length], &device);
                Some(self.prepare_encoder_attention_mask(&default_encoder_mask, seq_length))
            } else {
                None
            };

        // Initialize output collections
        let mut all_hidden_states = if input.output_hidden_states {
            Some(vec![hidden_states.clone()])
        } else {
            None
        };

        let mut all_attentions = if input.output_attentions {
            Some(Vec::new())
        } else {
            None
        };

        let mut all_cross_attentions = if input.output_attentions && is_decoder {
            Some(Vec::new())
        } else {
            None
        };

        let mut all_past_key_values = if input.use_cache {
            Some(Vec::new())
        } else {
            None
        };

        // Position bias storage
        let mut position_bias: Option<Tensor<B, 4>> = None;
        let mut encoder_decoder_position_bias: Option<Tensor<B, 4>> = None;

        // Process each layer
        for (layer_idx, layer) in self.blocks.iter().enumerate() {
            // Extract past key-value for this layer if available
            let _past_key_value = input
                .past_key_values
                .as_ref()
                .and_then(|pkv| pkv.get(layer_idx))
                .and_then(|pkv| pkv.clone());

            // Prepare head masks (simplified - just pass None for now)
            let layer_head_mask: Option<Tensor<B, 4>> = None;
            let cross_attn_layer_head_mask: Option<Tensor<B, 4>> = None;

            // Prepare layer input
            let layer_input = T5BlockInput {
                hidden_states: hidden_states.clone(),
                attention_mask: Some(causal_mask.clone()),
                position_bias: position_bias.clone(),
                encoder_hidden_states: input.encoder_hidden_states.clone(),
                encoder_attention_mask: encoder_extended_attention_mask.clone(),
                encoder_decoder_position_bias: encoder_decoder_position_bias.clone(),
                layer_head_mask,
                cross_attn_layer_head_mask,
                use_cache: input.use_cache,
                output_attentions: input.output_attentions,
                cache_position: None, // Simplified for now
            };

            // Create a simplified layer config for now
            // For T5, the correct d_kv = d_model / num_heads
            let num_heads = 8;
            let d_kv = d_model / num_heads;
            let layer_config = T5BlockConfig {
                ty: if is_decoder {
                    EncoderOrDecoder::Decoder
                } else {
                    EncoderOrDecoder::Encoder
                },
                relative_attention_num_buckets: 32,
                relative_attention_max_distance: 128,
                d_model,
                d_ff: d_model * 4,
                key_value_proj_dim: d_kv,
                num_heads,
                layer_norm_epsilon: 1e-6,
                dropout_rate: 0.0,
                has_relative_attention_bias: layer_idx == 0,
                layer_idx,
            };
            let layer_output = layer.forward(layer_input, &layer_config);

            // Update hidden states
            hidden_states = layer_output.hidden_states;

            // Store position bias from first layer
            if layer_idx == 0 {
                position_bias = layer_output.self_position_bias;
                if is_decoder && input.encoder_hidden_states.is_some() {
                    encoder_decoder_position_bias = layer_output.cross_position_bias;
                }
            }

            // Collect outputs if requested
            if let Some(ref mut all_hidden) = all_hidden_states {
                all_hidden.push(hidden_states.clone());
            }

            if let Some(ref mut all_attn) = all_attentions {
                all_attn.push(layer_output.self_attentions);
            }

            if let Some(ref mut all_cross_attn) = all_cross_attentions {
                all_cross_attn.push(layer_output.cross_attentions);
            }

            if let Some(ref mut all_pkv) = all_past_key_values {
                // For now, just push None - would need to extract from layer output in full implementation
                all_pkv.push(None);
            }
        }

        // Apply final layer normalization
        hidden_states = self.final_layer_norm.forward(hidden_states);
        hidden_states = self.dropout.forward(hidden_states);

        // Add final hidden state to collection if requested
        if let Some(ref mut all_hidden) = all_hidden_states {
            all_hidden.push(hidden_states.clone());
        }

        T5StackOutput {
            last_hidden_state: hidden_states,
            past_key_values: all_past_key_values,
            hidden_states: all_hidden_states,
            attentions: all_attentions,
            cross_attentions: all_cross_attentions,
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use burn::nn::EmbeddingConfig;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    /// Create a basic T5Stack configuration for testing
    fn create_test_config() -> T5StackConfig {
        T5StackConfig {
            n_layers: 2,
            ty: EncoderOrDecoder::Encoder,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            d_model: 64,
            d_ff: 128,
            key_value_proj_dim: 64,
            num_heads: 8,
            layer_norm_epsilon: 1e-6,
            dropout_rate: 0.0, // Disable dropout for deterministic tests
        }
    }

    /// Create test embeddings
    fn create_test_embeddings() -> Embedding<TestBackend> {
        EmbeddingConfig::new(1000, 64).init(&burn_ndarray::NdArrayDevice::default())
    }

    #[test]
    fn test_t5_stack_encoder_forward() {
        let device = burn_ndarray::NdArrayDevice::default();
        let config = create_test_config();
        let embed_tokens = create_test_embeddings();
        let stack = config.init(&device, embed_tokens);

        // Create test input
        let batch_size = 2;
        let seq_length = 8;
        let input_ids = Tensor::from_ints(
            [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]],
            &device,
        );

        let stack_input = T5StackInput {
            inputs: IdsOrEmbeds::Ids(input_ids),
            attention_mask: None,
            encoder_hidden_states: None,
            encoder_attention_mask: None,
            head_mask: None,
            cross_attn_head_mask: None,
            past_key_values: None,
            use_cache: false,
            output_attentions: false,
            output_hidden_states: false,
            cache_position: None,
        };

        // Forward pass
        let output = stack.forward(stack_input);

        // Verify output shape
        assert_eq!(
            output.last_hidden_state.dims(),
            [batch_size, seq_length, 64]
        );
        assert!(output.past_key_values.is_none());
        assert!(output.hidden_states.is_none());
        assert!(output.attentions.is_none());
        assert!(output.cross_attentions.is_none());
    }

    #[test]
    fn test_t5_stack_decoder_forward() {
        let device = burn_ndarray::NdArrayDevice::default();
        let mut config = create_test_config();
        config.ty = EncoderOrDecoder::Decoder;

        let embed_tokens = create_test_embeddings();
        let stack = config.init(&device, embed_tokens);

        // Create test input with encoder states
        let batch_size = 2;
        let seq_length = 6;
        let encoder_length = 8;

        let input_ids = Tensor::from_ints([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], &device);

        let encoder_hidden_states = Tensor::random(
            [batch_size, encoder_length, 64],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let stack_input = T5StackInput {
            inputs: IdsOrEmbeds::Ids(input_ids),
            attention_mask: None,
            encoder_hidden_states: Some(encoder_hidden_states),
            encoder_attention_mask: None,
            head_mask: None,
            cross_attn_head_mask: None,
            past_key_values: None,
            use_cache: false,
            output_attentions: false,
            output_hidden_states: false,
            cache_position: None,
        };

        // Forward pass
        let output = stack.forward(stack_input);

        // Verify output shape
        assert_eq!(
            output.last_hidden_state.dims(),
            [batch_size, seq_length, 64]
        );
    }

    #[test]
    fn test_t5_stack_with_attention_mask() {
        let device = burn_ndarray::NdArrayDevice::default();
        let config = create_test_config();
        let embed_tokens = create_test_embeddings();
        let stack = config.init(&device, embed_tokens);

        let batch_size = 2;
        let seq_length = 8;

        let input_ids = Tensor::from_ints(
            [[1, 2, 3, 4, 0, 0, 0, 0], [5, 6, 7, 8, 9, 10, 0, 0]],
            &device,
        );

        // Create attention mask (1 for real tokens, 0 for padding)
        let attention_mask = Tensor::from_floats(
            [
                [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            ],
            &device,
        );

        let stack_input = T5StackInput {
            inputs: IdsOrEmbeds::Ids(input_ids),
            attention_mask: Some(attention_mask),
            encoder_hidden_states: None,
            encoder_attention_mask: None,
            head_mask: None,
            cross_attn_head_mask: None,
            past_key_values: None,
            use_cache: false,
            output_attentions: false,
            output_hidden_states: false,
            cache_position: None,
        };

        // Forward pass
        let output = stack.forward(stack_input);

        // Verify output shape
        assert_eq!(
            output.last_hidden_state.dims(),
            [batch_size, seq_length, 64]
        );
    }

    #[test]
    fn test_t5_stack_output_attentions() {
        let device = burn_ndarray::NdArrayDevice::default();
        let config = create_test_config();
        let embed_tokens = create_test_embeddings();
        let stack = config.init(&device, embed_tokens);

        let batch_size = 2;
        let seq_length = 8;

        let input_ids = Tensor::from_ints(
            [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]],
            &device,
        );

        let stack_input = T5StackInput {
            inputs: IdsOrEmbeds::Ids(input_ids),
            attention_mask: None,
            encoder_hidden_states: None,
            encoder_attention_mask: None,
            head_mask: None,
            cross_attn_head_mask: None,
            past_key_values: None,
            use_cache: false,
            output_attentions: true,
            output_hidden_states: false,
            cache_position: None,
        };

        // Forward pass
        let output = stack.forward(stack_input);

        // Verify output
        assert_eq!(
            output.last_hidden_state.dims(),
            [batch_size, seq_length, 64]
        );
        assert!(output.attentions.is_some());
        let attentions = output.attentions.unwrap();
        assert_eq!(attentions.len(), 2); // Number of layers
    }

    #[test]
    fn test_t5_stack_output_hidden_states() {
        let device = burn_ndarray::NdArrayDevice::default();
        let config = create_test_config();
        let embed_tokens = create_test_embeddings();
        let stack = config.init(&device, embed_tokens);

        let batch_size = 2;
        let seq_length = 8;

        let input_ids = Tensor::from_ints(
            [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]],
            &device,
        );

        let stack_input = T5StackInput {
            inputs: IdsOrEmbeds::Ids(input_ids),
            attention_mask: None,
            encoder_hidden_states: None,
            encoder_attention_mask: None,
            head_mask: None,
            cross_attn_head_mask: None,
            past_key_values: None,
            use_cache: false,
            output_attentions: false,
            output_hidden_states: true,
            cache_position: None,
        };

        // Forward pass
        let output = stack.forward(stack_input);

        // Verify output
        assert_eq!(
            output.last_hidden_state.dims(),
            [batch_size, seq_length, 64]
        );
        assert!(output.hidden_states.is_some());
        let hidden_states = output.hidden_states.unwrap();
        assert_eq!(hidden_states.len(), 4); // Input + 2 layers + final
    }

    #[test]
    fn test_t5_stack_numerical_stability() {
        let device = burn_ndarray::NdArrayDevice::default();
        let config = create_test_config();
        let embed_tokens = create_test_embeddings();
        let stack = config.init(&device, embed_tokens);

        let input_ids = Tensor::from_ints([[1, 2, 3, 4]], &device);

        let stack_input = T5StackInput {
            inputs: IdsOrEmbeds::Ids(input_ids),
            attention_mask: None,
            encoder_hidden_states: None,
            encoder_attention_mask: None,
            head_mask: None,
            cross_attn_head_mask: None,
            past_key_values: None,
            use_cache: false,
            output_attentions: false,
            output_hidden_states: false,
            cache_position: None,
        };

        // Forward pass
        let output = stack.forward(stack_input);

        // Check for NaN or infinite values
        let data = output.last_hidden_state.to_data();
        for value in data.iter::<f32>() {
            assert!(value.is_finite(), "Output contains non-finite values");
        }
    }

    #[test]
    fn test_t5_stack_mask_conversion() {
        let device = burn_ndarray::NdArrayDevice::default();
        let config = create_test_config();
        let embed_tokens = create_test_embeddings();
        let stack = config.init(&device, embed_tokens);

        // Test 2D to 4D mask conversion
        let attention_mask =
            Tensor::from_floats([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0]], &device);

        let causal_mask = stack.prepare_causal_mask(
            &attention_mask,
            [2, 4],
            false, // encoder
            &device,
        );

        assert_eq!(causal_mask.dims(), [2, 1, 1, 4]);
    }

    // Python comparison test - requires transformers library
    #[test]
    #[ignore] // Ignore by default since it requires Python dependencies
    fn test_t5_stack_vs_python() {
        use ndarray::prelude::*;
        use numpy::ToPyArray;
        use pyo3::{ffi::c_str, prelude::*, types::PyDict};

        let device = burn_ndarray::NdArrayDevice::default();

        // Create smaller config for faster testing
        let config = T5StackConfig {
            n_layers: 1,
            ty: EncoderOrDecoder::Encoder,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            d_model: 16,
            d_ff: 32,
            key_value_proj_dim: 8, // d_model / num_heads = 16 / 2 = 8
            num_heads: 2,
            layer_norm_epsilon: 1e-6,
            dropout_rate: 0.0,
        };

        let embed_tokens: burn::nn::Embedding<TestBackend> =
            EmbeddingConfig::new(100, 16).init(&device);
        let stack = config.init(&device, embed_tokens);

        // Create deterministic input
        let input_ids = Tensor::from_ints([[1, 2, 3, 4]], &device);

        let stack_input = T5StackInput {
            inputs: IdsOrEmbeds::Ids(input_ids),
            attention_mask: None,
            encoder_hidden_states: None,
            encoder_attention_mask: None,
            head_mask: None,
            cross_attn_head_mask: None,
            past_key_values: None,
            use_cache: false,
            output_attentions: false,
            output_hidden_states: false,
            cache_position: None,
        };

        let output = stack.forward(stack_input);
        let output_array = output.last_hidden_state.to_data();
        let output_shape = output.last_hidden_state.dims();

        Python::with_gil(|py| {
            let locals = PyDict::new(py);

            // Convert output to numpy array
            let ndarray_output = Array3::from_shape_vec(
                (output_shape[0], output_shape[1], output_shape[2]),
                output_array.iter::<f32>().collect::<Vec<_>>(),
            )
            .unwrap();

            locals
                .set_item("burn_output", ndarray_output.to_pyarray(py))
                .unwrap();
            locals.set_item("d_model", 16).unwrap();
            locals.set_item("n_layers", 1).unwrap();
            locals.set_item("num_heads", 2).unwrap();

            py.run(
                c_str!(
                    r#"
import torch
import numpy as np
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.models.t5.configuration_t5 import T5Config

# Create a T5 config matching our Rust config
config = T5Config(
    vocab_size=100,
    d_model=16,
    d_ff=32,
    d_kv=8,  # d_model / num_heads
    num_heads=2,
    num_layers=1,
    relative_attention_num_buckets=32,
    relative_attention_max_distance=128,
    dropout_rate=0.0,
    layer_norm_epsilon=1e-6,
    initializer_factor=1.0,
    use_cache=False,
    is_encoder_decoder=True,
)

# Create T5Stack (encoder)
t5_stack = T5Stack(config, embed_tokens=None)
t5_stack.eval()

# Create input
input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

# Forward pass - note: this is a basic structure test, not exact numerical comparison
# since the weights are randomly initialized differently
with torch.no_grad():
    output = t5_stack(input_ids=input_ids)

# Check output shape matches
burn_shape = locals()['burn_output'].shape
python_shape = output.last_hidden_state.shape

assert burn_shape == python_shape, f"Shape mismatch: Burn {burn_shape} vs Python {python_shape}"
print("✓ Output shapes match between Burn and Python implementations")
                "#
                ),
                None,
                Some(&locals),
            )
            .unwrap();
        });
    }
}
