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
    pub cache_position: Option<Tensor<B, 1>>,
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
    /// Forward pass through the T5Stack
    pub fn forward(
        &self,
        input: T5StackInput<B>,
        config: T5StackConfig,
        device: &B::Device,
    ) -> T5StackOutput<B> {
        // Validate input - either input_ids or inputs_embeds must be provided
        let inputs_embeds = match input.inputs {
            IdsOrEmbeds::Ids(input_ids) => self.embed_tokens.forward(input_ids),
            IdsOrEmbeds::Embeds(embeds) => embeds,
        };

        let [batch_size, seq_length, _] = inputs_embeds.dims();

        // Apply dropout to input embeddings
        let mut hidden_states = self.dropout.forward(inputs_embeds);

        // For now, pass attention masks as-is to the layers
        let attention_mask = input
            .attention_mask
            .unwrap_or_else(|| Tensor::ones([batch_size, seq_length], device));
        let encoder_attention_mask = input.encoder_attention_mask;

        // Initialize output collections
        let mut all_hidden_states = input
            .output_hidden_states
            .then(|| vec![hidden_states.clone()]);

        let mut all_attentions = input.output_attentions.then(Vec::new);

        let mut all_cross_attentions =
            if input.output_attentions && matches!(config.ty, EncoderOrDecoder::Decoder) {
                Some(Vec::new())
            } else {
                None
            };

        let mut all_past_key_values = input.use_cache.then(Vec::new);

        // Position bias is computed in the first layer and reused
        let mut position_bias: Option<Tensor<B, 4>> = None;
        let mut encoder_decoder_position_bias: Option<Tensor<B, 4>> = None;

        // Process each layer
        for (layer_idx, layer) in self.blocks.iter().enumerate() {
            // For now, pass head masks as-is (simplified)
            let layer_head_mask = input.head_mask.clone();
            let cross_attn_layer_head_mask = input.cross_attn_head_mask.clone();

            // Extract past key-value for this layer if available
            let past_key_value = input
                .past_key_values
                .as_ref()
                .and_then(|pkv| pkv.get(layer_idx))
                .and_then(|pkv| pkv.clone());

            // Prepare input for the layer
            let layer_input = T5BlockInput {
                hidden_states: hidden_states.clone(),
                attention_mask: None, // TODO: Convert 2D to 4D mask
                position_bias: position_bias.clone(),
                encoder_hidden_states: input.encoder_hidden_states.clone(),
                encoder_attention_mask: None, // TODO: Convert 2D to 4D mask
                encoder_decoder_position_bias: encoder_decoder_position_bias.clone(),
                layer_head_mask: None,            // TODO: Convert 2D to 4D mask
                cross_attn_layer_head_mask: None, // TODO: Convert 2D to 4D mask
                use_cache: input.use_cache,
                output_attentions: input.output_attentions,
                cache_position: None, // TODO: Implement cache position handling
            };

            // Forward through the layer
            let layer_output = layer.forward(
                layer_input,
                &T5BlockConfig {
                    ty: config.ty,
                    relative_attention_num_buckets: config.relative_attention_num_buckets,
                    relative_attention_max_distance: config.relative_attention_max_distance,
                    d_model: config.d_model,
                    d_ff: config.d_ff,
                    key_value_proj_dim: config.key_value_proj_dim,
                    num_heads: config.num_heads,
                    layer_norm_epsilon: config.layer_norm_epsilon,
                    dropout_rate: config.dropout_rate,
                    has_relative_attention_bias: layer_idx == 0,
                    layer_idx,
                },
            );

            // Update hidden states
            hidden_states = layer_output.hidden_states;

            // Store position bias from first layer
            if layer_idx == 0 {
                position_bias = layer_output.self_position_bias;
                if matches!(config.ty, EncoderOrDecoder::Decoder)
                    && input.encoder_hidden_states.is_some()
                {
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
                // TODO: Extract and store past key-values from layer output
                all_pkv.push(None); // Placeholder
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
            dropout_rate: 0.1,
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
        let output = stack.forward(stack_input, config, &device);

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
        let output = stack.forward(stack_input, config, &device);

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
        let output = stack.forward(stack_input, config, &device);

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
        let output = stack.forward(stack_input, config, &device);

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
        let output = stack.forward(stack_input, config, &device);

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
        let output = stack.forward(stack_input, config, &device);

        // Check for NaN or infinite values
        let data = output.last_hidden_state.to_data();
        for value in data.iter::<f32>() {
            assert!(value.is_finite(), "Output contains non-finite values");
        }
    }
}
