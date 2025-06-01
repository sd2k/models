//! T5 Attention Module
//!
//! This module implements the attention mechanisms used in the T5 (Text-to-Text Transfer Transformer) model.
//! It includes self-attention, cross-attention, and the core attention computation with relative position bias.
//!
//! The implementation follows the original T5 paper and PyTorch implementation, providing:
//! - Multi-head attention with relative position encoding
//! - Self-attention for encoder and decoder layers
//! - Cross-attention for decoder layers to attend to encoder outputs
//! - Configurable relative position bias for better handling of sequence relationships

use burn::{
    config::Config,
    nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig, Linear, LinearConfig},
    prelude::*,
    tensor::activation::softmax,
};

use crate::layer_norm::{T5LayerNorm, T5LayerNormConfig};

/// Specifies whether the attention layer is used in the encoder or decoder.
/// This affects the bidirectionality of relative position bias computation.
#[derive(Debug, Config, Copy, PartialEq)]
pub enum EncoderOrDecoder {
    /// Encoder attention uses bidirectional relative position bias
    Encoder,
    /// Decoder attention uses unidirectional relative position bias
    Decoder,
}

/// Output of an attention layer, containing the transformed hidden states and optionally the attention weights.
pub struct AttentionOutput<B: Backend, const D: usize> {
    /// The output hidden states after applying attention and output projection
    pub output: Tensor<B, D>,
    /// Optional attention weights tensor with shape [batch_size, num_heads, seq_length, key_length]
    /// Only populated when `output_attentions` is true
    pub attentions: Option<Tensor<B, 4>>,
}

/// Input structure for the core T5 attention mechanism.
pub struct T5AttentionInput<B: Backend, const D: usize> {
    /// Input hidden states with shape [batch_size, seq_length, hidden_size]
    pub hidden_states: Tensor<B, D>,
    /// Attention mask to prevent attention to certain positions [batch_size, 1, seq_length, key_length]
    pub attention_mask: Option<Tensor<B, 4>>,
    /// Key and value states for cross-attention (None for self-attention)
    pub key_value_states: Option<Tensor<B, D>>,
    /// Pre-computed position bias [1, num_heads, seq_length, key_length]
    pub position_bias: Option<Tensor<B, 4>>,
    /// Past key-value states for caching (used in generation)
    pub past_key_value: Option<Tensor<B, D>>,
    /// Mask to disable specific attention heads [1, num_heads, 1, 1]
    pub layer_head_mask: Option<Tensor<B, 4>>,
    /// Length of the query sequence (for relative position computation)
    pub query_length: Option<usize>,
    /// Whether to use key-value caching
    pub use_cache: bool,
    /// Whether to return attention weights
    pub output_attentions: bool,
    /// Cache position information for incremental generation
    pub cache_position: Option<bool>,
}

/// Input structure for T5 self-attention layers.
pub struct T5SelfAttentionInput<B: Backend, const D: usize> {
    /// Input hidden states with shape [batch_size, seq_length, hidden_size]
    pub hidden_states: Tensor<B, D>,
    /// Attention mask to prevent attention to certain positions
    pub attention_mask: Option<Tensor<B, 4>>,
    /// Pre-computed position bias for relative attention
    pub position_bias: Option<Tensor<B, 4>>,
    /// Mask to disable specific attention heads
    pub layer_head_mask: Option<Tensor<B, 4>>,
    /// Past key-value states for caching
    pub past_key_value: Option<Tensor<B, D>>,
    /// Whether to use key-value caching
    pub use_cache: bool,
    /// Length of the query sequence
    pub query_length: Option<usize>,
    /// Whether to return attention weights
    pub output_attentions: bool,
    /// Cache position information
    pub cache_position: Option<bool>,
}

/// Configuration for T5 self-attention layers.
#[derive(Config, Debug)]
pub struct T5SelfAttentionConfig {
    /// Whether this is used in encoder or decoder (affects relative position bias)
    pub ty: EncoderOrDecoder,
    /// Number of buckets for relative position encoding
    pub relative_attention_num_buckets: usize,
    /// Maximum distance for relative position encoding
    pub relative_attention_max_distance: usize,
    /// Model dimension (hidden size)
    pub d_model: usize,
    /// Dimension of each attention head
    pub key_value_proj_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Epsilon for layer normalization
    pub layer_norm_epsilon: f64,
    /// Dropout rate applied to attention
    pub dropout_rate: f64,
    /// Layer index in the model (used for caching)
    pub layer_idx: usize,
    /// Whether this layer computes relative attention bias
    pub has_relative_attention_bias: bool,
}

impl T5SelfAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> T5SelfAttention<B> {
        let self_attention = T5AttentionConfig {
            ty: self.ty,
            has_relative_attention_bias: self.has_relative_attention_bias,
            relative_attention_num_buckets: self.relative_attention_num_buckets,
            relative_attention_max_distance: self.relative_attention_max_distance,
            d_model: self.d_model,
            key_value_proj_dim: self.key_value_proj_dim,
            num_heads: self.num_heads,
            dropout_rate: self.dropout_rate,
            layer_idx: self.layer_idx,
        }
        .init(device);
        let layer_norm = T5LayerNormConfig {
            d_model: self.d_model,
            layer_norm_epsilon: self.layer_norm_epsilon,
        }
        .init(device);
        let dropout = DropoutConfig::new(self.dropout_rate).init();
        T5SelfAttention {
            self_attention,
            layer_norm,
            dropout,
        }
    }
}

/// T5 Self-Attention layer that applies layer normalization, self-attention, and residual connection.
/// This is used in both encoder and decoder layers for attending to the same sequence.
#[derive(Module, Debug)]
pub struct T5SelfAttention<B: Backend> {
    /// Core attention mechanism
    self_attention: T5Attention<B>,
    /// Layer normalization applied before attention
    layer_norm: T5LayerNorm<B>,
    /// Dropout applied to attention output
    dropout: Dropout,
}

impl<B: Backend> T5SelfAttention<B> {
    /// Forward pass through the self-attention layer.
    ///
    /// Applies the pre-norm pattern: layer norm -> attention -> residual connection.
    /// This follows the T5 architecture where layer normalization is applied before
    /// the attention computation rather than after.
    ///
    /// # Arguments
    /// * `input` - Self-attention input containing hidden states and optional masks
    /// * `config` - Configuration parameters for the attention layer
    ///
    /// # Returns
    /// AttentionOutput containing transformed hidden states and optional attention weights
    pub fn forward(
        &self,
        input: T5SelfAttentionInput<B, 3>,
        config: &T5SelfAttentionConfig,
    ) -> AttentionOutput<B, 3> {
        let normed_hidden_states = self.layer_norm.forward(input.hidden_states.clone());
        let attention_input = T5AttentionInput {
            hidden_states: normed_hidden_states,
            attention_mask: input.attention_mask,
            key_value_states: None, // not used in self-attention
            position_bias: input.position_bias,
            past_key_value: input.past_key_value,
            layer_head_mask: input.layer_head_mask,
            query_length: input.query_length,
            use_cache: input.use_cache,
            output_attentions: input.output_attentions,
            cache_position: input.cache_position,
        };

        let attention_config = T5AttentionConfig {
            ty: config.ty,
            has_relative_attention_bias: config.has_relative_attention_bias,
            relative_attention_num_buckets: config.relative_attention_num_buckets,
            relative_attention_max_distance: config.relative_attention_max_distance,
            d_model: config.d_model,
            key_value_proj_dim: config.key_value_proj_dim,
            num_heads: config.num_heads,
            dropout_rate: config.dropout_rate,
            layer_idx: config.layer_idx,
        };

        let AttentionOutput { output, attentions } = self
            .self_attention
            .forward(attention_input, &attention_config);
        let layer_output = input.hidden_states + self.dropout.forward(output);
        AttentionOutput {
            output: layer_output,
            attentions,
        }
    }
}

/// Input structure for T5 cross-attention layers.
pub struct T5CrossAttentionInput<B: Backend, const D: usize> {
    /// Decoder hidden states (queries) with shape [batch_size, seq_length, hidden_size]
    pub hidden_states: Tensor<B, D>,
    /// Encoder hidden states (keys and values) with shape [batch_size, encoder_length, hidden_size]
    pub key_value_states: Tensor<B, D>,
    /// Attention mask to prevent attention to certain encoder positions
    pub attention_mask: Option<Tensor<B, 4>>,
    /// Pre-computed position bias for relative attention
    pub position_bias: Option<Tensor<B, 4>>,
    /// Mask to disable specific attention heads
    pub layer_head_mask: Option<Tensor<B, 4>>,
    /// Past key-value states for caching
    pub past_key_value: Option<Tensor<B, D>>,
    /// Whether to use key-value caching
    pub use_cache: bool,
    /// Length of the query sequence
    pub query_length: Option<usize>,
    /// Whether to return attention weights
    pub output_attentions: bool,
    /// Cache position information
    pub cache_position: Option<bool>,
}

/// Configuration for T5 cross-attention layers.
#[derive(Config, Debug)]
pub struct T5CrossAttentionConfig {
    /// Whether this is used in encoder or decoder (typically Decoder for cross-attention)
    pub ty: EncoderOrDecoder,
    /// Number of buckets for relative position encoding
    pub relative_attention_num_buckets: usize,
    /// Maximum distance for relative position encoding
    pub relative_attention_max_distance: usize,
    /// Model dimension (hidden size)
    pub d_model: usize,
    /// Dimension of each attention head
    pub key_value_proj_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Epsilon for layer normalization
    pub layer_norm_epsilon: f64,
    /// Dropout rate applied to attention
    pub dropout_rate: f64,
    /// Layer index in the model (used for caching)
    pub layer_idx: usize,
}

impl T5CrossAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> T5CrossAttention<B> {
        let cross_attention = T5AttentionConfig {
            ty: self.ty,
            has_relative_attention_bias: false,
            relative_attention_num_buckets: self.relative_attention_num_buckets,
            relative_attention_max_distance: self.relative_attention_max_distance,
            d_model: self.d_model,
            key_value_proj_dim: self.key_value_proj_dim,
            num_heads: self.num_heads,
            dropout_rate: self.dropout_rate,
            layer_idx: self.layer_idx,
        }
        .init(device);
        let layer_norm = T5LayerNormConfig {
            d_model: self.d_model,
            layer_norm_epsilon: self.layer_norm_epsilon,
        }
        .init(device);
        let dropout = DropoutConfig::new(self.dropout_rate).init();
        T5CrossAttention {
            cross_attention,
            layer_norm,
            dropout,
        }
    }
}

/// T5 Cross-Attention layer used in decoder blocks to attend to encoder outputs.
/// Applies layer normalization to decoder states, then computes attention over encoder states.
#[derive(Module, Debug)]
pub struct T5CrossAttention<B: Backend> {
    /// Core attention mechanism
    cross_attention: T5Attention<B>,
    /// Layer normalization applied to decoder hidden states
    layer_norm: T5LayerNorm<B>,
    /// Dropout applied to attention output
    dropout: Dropout,
}

impl<B: Backend> T5CrossAttention<B> {
    /// Forward pass through the cross-attention layer.
    ///
    /// Normalizes the decoder hidden states, then computes attention over encoder states.
    /// The queries come from the normalized decoder states, while keys and values come
    /// from the encoder states.
    ///
    /// # Arguments
    /// * `input` - Cross-attention input with decoder states and encoder key-value states
    /// * `config` - Configuration parameters for the attention layer
    ///
    /// # Returns
    /// AttentionOutput containing transformed decoder states and optional attention weights
    pub fn forward(
        &self,
        input: T5CrossAttentionInput<B, 3>,
        config: &T5CrossAttentionConfig,
    ) -> AttentionOutput<B, 3> {
        let normed_hidden_states = self.layer_norm.forward(input.hidden_states.clone());
        let attention_input = T5AttentionInput {
            hidden_states: normed_hidden_states,
            attention_mask: input.attention_mask,
            key_value_states: Some(input.key_value_states),
            position_bias: input.position_bias,
            past_key_value: input.past_key_value,
            layer_head_mask: input.layer_head_mask,
            query_length: input.query_length,
            use_cache: input.use_cache,
            output_attentions: input.output_attentions,
            cache_position: input.cache_position,
        };

        let attention_config = T5AttentionConfig {
            ty: config.ty,
            has_relative_attention_bias: false,
            relative_attention_num_buckets: config.relative_attention_num_buckets,
            relative_attention_max_distance: config.relative_attention_max_distance,
            d_model: config.d_model,
            key_value_proj_dim: config.key_value_proj_dim,
            num_heads: config.num_heads,
            dropout_rate: config.dropout_rate,
            layer_idx: config.layer_idx,
        };

        let AttentionOutput { output, attentions } = self
            .cross_attention
            .forward(attention_input, &attention_config);
        let layer_output = input.hidden_states + self.dropout.forward(output);
        AttentionOutput {
            output: layer_output,
            attentions,
        }
    }
}

/// Configuration for the core T5 attention mechanism.
#[derive(Config, Debug)]
pub struct T5AttentionConfig {
    /// Whether this is encoder or decoder attention
    pub ty: EncoderOrDecoder,
    /// Whether this layer computes relative attention bias (usually only first layer)
    #[config(default = false)]
    pub has_relative_attention_bias: bool,
    /// Number of buckets for relative position encoding
    pub relative_attention_num_buckets: usize,
    /// Maximum distance for relative position encoding
    pub relative_attention_max_distance: usize,
    /// Model dimension (hidden size)
    pub d_model: usize,
    /// Dimension of each attention head (typically d_model / num_heads)
    pub key_value_proj_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Dropout rate applied to attention weights
    pub dropout_rate: f64,
    /// Layer index in the model (used for caching)
    pub layer_idx: usize,
}

impl T5AttentionConfig {
    /// Initialize a T5Attention module from this configuration.
    ///
    /// Creates all the necessary linear projections and optional relative position bias embedding.
    /// The inner dimension is computed as num_heads * key_value_proj_dim.
    ///
    /// # Arguments
    /// * `device` - Device to create the module on
    ///
    /// # Returns
    /// Initialized T5Attention module
    pub fn init<B: Backend>(&self, device: &B::Device) -> T5Attention<B> {
        let inner_dim = self.num_heads * self.key_value_proj_dim;
        let query = LinearConfig::new(self.d_model, inner_dim)
            .with_bias(false)
            .init(device);
        let key = LinearConfig::new(self.d_model, inner_dim)
            .with_bias(false)
            .init(device);
        let value = LinearConfig::new(self.d_model, inner_dim)
            .with_bias(false)
            .init(device);
        let dense = LinearConfig::new(inner_dim, self.d_model)
            .with_bias(false)
            .init(device);
        let relative_attention_bias = if self.has_relative_attention_bias {
            Some(
                EmbeddingConfig::new(self.relative_attention_num_buckets, self.num_heads)
                    .init(device),
            )
        } else {
            None
        };
        let dropout = DropoutConfig::new(self.dropout_rate).init();

        T5Attention {
            query,
            key,
            value,
            dense,
            relative_attention_bias,
            dropout,
        }
    }
}

/// Core T5 attention mechanism implementing multi-head attention with relative position bias.
///
/// This is the fundamental attention computation used in both self-attention and cross-attention.
/// It includes linear projections for queries, keys, and values, relative position bias computation,
/// and the scaled dot-product attention mechanism.
#[derive(Module, Debug)]
pub struct T5Attention<B: Backend> {
    /// Linear projection for queries
    query: Linear<B>,
    /// Linear projection for keys
    key: Linear<B>,
    /// Linear projection for values
    value: Linear<B>,
    /// Output projection after attention
    dense: Linear<B>,
    /// Optional embedding for relative position bias (only in certain layers)
    relative_attention_bias: Option<Embedding<B>>,
    /// Dropout applied to attention weights
    dropout: Dropout,
}

impl<B: Backend> T5Attention<B> {
    /// Compute relative position bucket for attention bias.
    ///
    /// This function maps relative positions to bucket indices for the relative attention bias.
    /// It uses a combination of linear buckets for small distances and logarithmic buckets
    /// for larger distances, following the T5 paper approach.
    ///
    /// # Arguments
    /// * `relative_position` - Tensor of relative positions [query_length, key_length]
    /// * `bidirectional` - Whether attention is bidirectional (encoder) or unidirectional (decoder)
    /// * `num_buckets` - Total number of position buckets
    /// * `max_distance` - Maximum distance to consider for bucketing
    /// * `device` - Device for tensor operations
    ///
    /// # Returns
    /// Tensor of bucket indices with same shape as input
    fn relative_position_bucket(
        relative_position: Tensor<B, 2>,
        bidirectional: bool,
        num_buckets: usize,
        max_distance: usize,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let mut relative_buckets = Tensor::zeros_like(&relative_position);
        let mut num_buckets = num_buckets;

        if bidirectional {
            num_buckets /= 2;
            // Add num_buckets if relative_position > 0
            let positive_mask = relative_position.clone().greater_elem(0.0);
            let positive_contribution = positive_mask.float() * (num_buckets as f64);
            relative_buckets = relative_buckets + positive_contribution;
        }

        // Take absolute value or clamp to 0 for unidirectional
        let relative_position = if bidirectional {
            relative_position.abs()
        } else {
            relative_position.neg().clamp_min(0.0)
        };

        // Half of buckets for exact increments
        let max_exact = num_buckets / 2;
        let is_small = relative_position.clone().lower_elem(max_exact as f64);

        // For small positions, use the position directly
        let small_buckets = relative_position.clone();

        // For large positions, use logarithmic buckets (but avoid log(0) by adding epsilon)
        let epsilon = 1e-10;
        let safe_relative_position = relative_position.clone().clamp_min(epsilon);
        let log_ratio = (safe_relative_position / (max_exact as f64)).log()
            / ((max_distance as f64) / (max_exact as f64)).ln();
        let relative_position_if_large =
            Tensor::full(relative_position.shape(), max_exact as f64, device)
                + log_ratio * ((num_buckets - max_exact) as f64);
        let relative_position_if_large =
            relative_position_if_large.clamp_max((num_buckets - 1) as f64);

        // Choose between small and large buckets using the mask
        let is_small_float = is_small.float();
        let is_large_float = Tensor::ones_like(&is_small_float) - is_small_float.clone();
        let selected = is_small_float * small_buckets + is_large_float * relative_position_if_large;
        relative_buckets = relative_buckets + selected;

        // Ensure values are in valid range and round them
        relative_buckets.clamp(0.0, (num_buckets - 1) as f64)
    }

    /// Compute position bias for relative attention.
    ///
    /// Creates position bias by computing relative positions between query and key positions,
    /// mapping them to buckets, and looking up bias values from the embedding table.
    ///
    /// # Arguments
    /// * `query_length` - Length of the query sequence
    /// * `key_length` - Length of the key sequence
    /// * `device` - Device for tensor operations
    /// * `bidirectional` - Whether attention is bidirectional
    /// * `relative_attention_num_buckets` - Number of position buckets
    /// * `relative_attention_max_distance` - Maximum relative distance
    /// * `num_heads` - Number of attention heads
    ///
    /// # Returns
    /// Position bias tensor with shape [1, num_heads, query_length, key_length]
    fn compute_bias(
        &self,
        query_length: usize,
        key_length: usize,
        device: &B::Device,
        bidirectional: bool,
        relative_attention_num_buckets: usize,
        relative_attention_max_distance: usize,
        num_heads: usize,
    ) -> Tensor<B, 4> {
        // Create position tensors
        let context_position = Tensor::arange(0..query_length as i64, device)
            .float()
            .unsqueeze_dim(1); // [query_length, 1]
        let memory_position = Tensor::arange(0..key_length as i64, device)
            .float()
            .unsqueeze_dim(0); // [1, key_length]

        // Compute relative position: memory_position - context_position
        let relative_position = memory_position - context_position; // [query_length, key_length]

        // Compute relative position buckets
        let relative_position_bucket = Self::relative_position_bucket(
            relative_position,
            bidirectional,
            relative_attention_num_buckets,
            relative_attention_max_distance,
            device,
        );

        // Get bias values from embedding
        if let Some(ref bias_embedding) = self.relative_attention_bias {
            // Round and clamp values before converting to int with additional safety checks
            let bucket_indices = relative_position_bucket
                .round()
                .clamp(0.0, (relative_attention_num_buckets - 1) as f64)
                .clamp(0.0, i32::MAX as f64); // Ensure values fit in i32 range
            let values = bias_embedding.forward(bucket_indices.int());
            // Permute dimensions from [query_length, key_length, num_heads] to [1, num_heads, query_length, key_length]
            values.swap_dims(0, 2).swap_dims(1, 2).unsqueeze_dim(0)
        } else {
            // Return zeros if no relative attention bias
            Tensor::zeros([1, num_heads, query_length, key_length], device)
        }
    }

    /// Forward pass through the attention mechanism.
    ///
    /// Performs the complete multi-head attention computation including:
    /// 1. Linear projections to queries, keys, and values
    /// 2. Reshaping to multi-head format
    /// 3. Computing attention scores with optional position bias
    /// 4. Applying attention mask and head mask
    /// 5. Computing attention weights and applying to values
    /// 6. Output projection
    ///
    /// # Arguments
    /// * `input` - Attention input containing hidden states and optional parameters
    /// * `config` - Configuration parameters for the attention computation
    ///
    /// # Returns
    /// AttentionOutput containing transformed hidden states and optional attention weights
    ///
    /// # Example
    /// ```no_compile
    /// use burn::prelude::*;
    ///
    /// // Usage with direct struct initialization
    /// let input = T5AttentionInput {
    ///     hidden_states,
    ///     attention_mask: Some(attention_mask),
    ///     key_value_states: Some(key_value_states),
    ///     position_bias: None,
    ///     past_key_value: None,
    ///     layer_head_mask: None,
    ///     query_length: None,
    ///     use_cache: true,
    ///     output_attentions: true,
    ///     cache_position: None,
    /// };
    /// let output = attention.forward(input, &config);
    /// ```
    pub fn forward(
        &self,
        input: T5AttentionInput<B, 3>,
        config: &T5AttentionConfig,
    ) -> AttentionOutput<B, 3> {
        let device = input.hidden_states.device();
        let dims = input.hidden_states.dims();
        let batch_size = dims[0];
        let seq_length = dims[1];

        // Compute query states
        let query_states = self.query.forward(input.hidden_states.clone());
        let query_states = query_states
            .reshape([
                batch_size,
                seq_length,
                config.num_heads,
                config.key_value_proj_dim,
            ])
            .swap_dims(1, 2); // [batch_size, num_heads, seq_length, key_value_proj_dim]

        // Determine which states to use for key/value computation
        let current_states = input.key_value_states.unwrap_or(input.hidden_states);

        // Compute key and value states
        let key_states = self.key.forward(current_states.clone());
        let value_states = self.value.forward(current_states);

        let key_length = key_states.dims()[1];
        let key_states = key_states
            .reshape([
                batch_size,
                key_length,
                config.num_heads,
                config.key_value_proj_dim,
            ])
            .swap_dims(1, 2); // [batch_size, num_heads, key_length, key_value_proj_dim]
        let value_states = value_states
            .reshape([
                batch_size,
                key_length,
                config.num_heads,
                config.key_value_proj_dim,
            ])
            .swap_dims(1, 2); // [batch_size, num_heads, key_length, key_value_proj_dim]

        // Compute attention scores
        let scores = query_states.matmul(key_states.swap_dims(2, 3));

        // Compute position bias
        let position_bias = if let Some(bias) = input.position_bias {
            bias
        } else if config.has_relative_attention_bias {
            self.compute_bias(
                seq_length,
                key_length,
                &device,
                config.ty == EncoderOrDecoder::Encoder,
                config.relative_attention_num_buckets,
                config.relative_attention_max_distance,
                config.num_heads,
            )
        } else {
            Tensor::zeros([1, config.num_heads, seq_length, key_length], &device)
        };

        // Add position bias to scores
        let mut scores = scores + position_bias;

        // Apply attention mask if provided
        if let Some(mask) = input.attention_mask {
            // Ensure mask has the right shape [batch_size, 1, seq_length, key_length] or can be broadcast
            scores = scores + mask;
        }

        // Apply softmax
        let attn_weights = softmax(scores, 3);

        // Apply dropout
        let attn_weights = self.dropout.forward(attn_weights);

        // Apply layer head mask if provided
        let attn_weights = if let Some(head_mask) = input.layer_head_mask {
            // Ensure head_mask has the right shape for broadcasting
            attn_weights * head_mask
        } else {
            attn_weights
        };

        // Apply attention to value states
        let attn_output = attn_weights.clone().matmul(value_states);

        // Reshape and apply output projection
        let inner_dim = config.num_heads * config.key_value_proj_dim;
        let attn_output = attn_output
            .swap_dims(1, 2) // [batch_size, seq_length, num_heads, key_value_proj_dim]
            .reshape([batch_size, seq_length, inner_dim]);
        let attn_output = self.dense.forward(attn_output);

        AttentionOutput {
            output: attn_output,
            attentions: if input.output_attentions {
                Some(attn_weights)
            } else {
                None
            },
        }
    }
}

#[cfg(test)]
mod test_t5_attention {
    use super::*;
    use burn::tensor::Distribution;
    use burn_ndarray::NdArrayDevice;

    type TestBackend = burn_ndarray::NdArray<f64>;

    #[test]
    fn test_t5_attention_forward() {
        let device = NdArrayDevice::default();

        // Create attention config
        let config = T5AttentionConfig {
            ty: EncoderOrDecoder::Encoder,
            has_relative_attention_bias: true,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            d_model: 512,
            key_value_proj_dim: 64,
            num_heads: 8,
            dropout_rate: 0.1,
            layer_idx: 0,
        };

        // Initialize attention module
        let attention = config.init::<TestBackend>(&device);

        // Create test input
        let batch_size = 2;
        let seq_length = 10;
        let hidden_dim = 512;

        let hidden_states = Tensor::random(
            [batch_size, seq_length, hidden_dim],
            Distribution::Normal(0.0, 1.0),
            &device,
        );

        let input = T5AttentionInput {
            hidden_states,
            attention_mask: None,
            key_value_states: None,
            position_bias: None,
            past_key_value: None,
            layer_head_mask: None,
            query_length: None,
            use_cache: false,
            output_attentions: true,
            cache_position: None,
        };

        // Forward pass
        let output = attention.forward(input, &config);

        // Check output shape
        assert_eq!(output.output.dims(), [batch_size, seq_length, hidden_dim]);
        assert!(output.attentions.is_some());

        let attention_weights = output.attentions.unwrap();
        assert_eq!(
            attention_weights.dims(),
            [batch_size, config.num_heads, seq_length, seq_length]
        );
    }

    #[test]
    fn test_t5_self_attention_forward() {
        let device = NdArrayDevice::default();

        // Create self attention config
        let config = T5SelfAttentionConfig {
            ty: EncoderOrDecoder::Encoder,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            d_model: 512,
            key_value_proj_dim: 64,
            num_heads: 8,
            layer_norm_epsilon: 1e-6,
            dropout_rate: 0.1,
            layer_idx: 0,
            has_relative_attention_bias: true,
        };

        // Initialize self attention module
        let self_attention = config.init::<TestBackend>(&device);

        // Create test input
        let batch_size = 2;
        let seq_length = 10;
        let hidden_dim = 512;

        let hidden_states = Tensor::random(
            [batch_size, seq_length, hidden_dim],
            Distribution::Normal(0.0, 1.0),
            &device,
        );

        let input = T5SelfAttentionInput {
            hidden_states,
            attention_mask: None,
            position_bias: None,
            layer_head_mask: None,
            past_key_value: None,
            use_cache: false,
            query_length: None,
            output_attentions: false,
            cache_position: None,
        };

        // Forward pass
        let output = self_attention.forward(input, &config);

        // Check output shape
        assert_eq!(output.output.dims(), [batch_size, seq_length, hidden_dim]);
        assert!(output.attentions.is_none());
    }

    #[test]
    fn test_t5_cross_attention_forward() {
        let device = NdArrayDevice::default();

        // Create cross attention config
        let config = T5CrossAttentionConfig {
            ty: EncoderOrDecoder::Decoder,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            d_model: 512,
            key_value_proj_dim: 64,
            num_heads: 8,
            layer_norm_epsilon: 1e-6,
            dropout_rate: 0.1,
            layer_idx: 0,
        };

        // Initialize cross attention module
        let cross_attention = config.init::<TestBackend>(&device);

        // Create test input
        let batch_size = 2;
        let seq_length = 10;
        let encoder_length = 15;
        let hidden_dim = 512;

        let hidden_states = Tensor::random(
            [batch_size, seq_length, hidden_dim],
            Distribution::Normal(0.0, 1.0),
            &device,
        );

        let key_value_states = Tensor::random(
            [batch_size, encoder_length, hidden_dim],
            Distribution::Normal(0.0, 1.0),
            &device,
        );

        let input = T5CrossAttentionInput {
            hidden_states,
            key_value_states,
            attention_mask: None,
            position_bias: None,
            layer_head_mask: None,
            past_key_value: None,
            use_cache: false,
            query_length: None,
            output_attentions: true,
            cache_position: None,
        };

        // Forward pass
        let output = cross_attention.forward(input, &config);

        // Check output shape
        assert_eq!(output.output.dims(), [batch_size, seq_length, hidden_dim]);
        assert!(output.attentions.is_some());

        let attention_weights = output.attentions.unwrap();
        assert_eq!(
            attention_weights.dims(),
            [batch_size, config.num_heads, seq_length, encoder_length]
        );
    }

    #[test]
    fn test_t5_attention_functional() {
        let device = NdArrayDevice::default();

        // Test parameters
        let batch_size = 2;
        let seq_length = 4;
        let hidden_dim = 8;
        let num_heads = 2;
        let key_value_proj_dim = 4;

        // Create attention config
        let config = T5AttentionConfig {
            ty: EncoderOrDecoder::Encoder,
            has_relative_attention_bias: true,
            relative_attention_num_buckets: 8,
            relative_attention_max_distance: 16,
            d_model: hidden_dim,
            key_value_proj_dim,
            num_heads,
            dropout_rate: 0.0,
            layer_idx: 0,
        };

        // Initialize attention module
        let attention = config.init::<TestBackend>(&device);

        // Create test input
        let input_data: Vec<f64> = (0..batch_size * seq_length * hidden_dim)
            .map(|i| (i as f64) * 0.1)
            .collect();

        let hidden_states = Tensor::from_data(
            burn::tensor::TensorData::new(input_data, [batch_size, seq_length, hidden_dim]),
            &device,
        );

        // Test self-attention
        let input = T5AttentionInput {
            hidden_states: hidden_states.clone(),
            attention_mask: None,
            key_value_states: None,
            position_bias: None,
            past_key_value: None,
            layer_head_mask: None,
            query_length: None,
            use_cache: false,
            output_attentions: true,
            cache_position: None,
        };

        // Forward pass
        let output = attention.forward(input, &config);

        // Verify output shape
        assert_eq!(output.output.dims(), [batch_size, seq_length, hidden_dim]);

        // Verify attention weights shape when requested
        assert!(output.attentions.is_some());
        let attention_weights = output.attentions.unwrap();
        assert_eq!(
            attention_weights.dims(),
            [batch_size, num_heads, seq_length, seq_length]
        );

        // Test cross-attention with different key-value states
        let encoder_length = 6;
        let key_value_states = Tensor::random(
            [batch_size, encoder_length, hidden_dim],
            Distribution::Normal(0.0, 1.0),
            &device,
        );

        let cross_input = T5AttentionInput {
            hidden_states: hidden_states.clone(),
            attention_mask: None,
            key_value_states: Some(key_value_states),
            position_bias: None,
            past_key_value: None,
            layer_head_mask: None,
            query_length: None,
            use_cache: false,
            output_attentions: true,
            cache_position: None,
        };

        let cross_output = attention.forward(cross_input, &config);

        // Verify cross-attention output
        assert_eq!(
            cross_output.output.dims(),
            [batch_size, seq_length, hidden_dim]
        );
        let cross_attention_weights = cross_output.attentions.unwrap();
        assert_eq!(
            cross_attention_weights.dims(),
            [batch_size, num_heads, seq_length, encoder_length]
        );

        // Verify outputs are different (not trivial)
        let self_attn_data = output.output.to_data();
        let cross_attn_data = cross_output.output.to_data();
        let self_values: Vec<f64> = self_attn_data.iter().collect();
        let cross_values: Vec<f64> = cross_attn_data.iter().collect();

        // Check that outputs are different (attention actually does something)
        let differences: Vec<_> = self_values
            .iter()
            .zip(cross_values.iter())
            .map(|(a, b)| (a - b).abs())
            .filter(|&diff| diff > 1e-6)
            .collect();

        assert!(
            !differences.is_empty(),
            "Self-attention and cross-attention should produce different outputs"
        );

        println!("✅ T5Attention functional verification test passed!");
    }

    #[test]
    fn test_relative_position_bucket_debug() {
        let device = NdArrayDevice::default();

        // Simple test parameters
        let query_length = 2;
        let key_length = 3;

        // Create position tensors directly
        let context_position = Tensor::arange(0..query_length as i64, &device)
            .float()
            .unsqueeze_dim(1); // [query_length, 1]
        let memory_position = Tensor::arange(0..key_length as i64, &device)
            .float()
            .unsqueeze_dim(0); // [1, key_length]

        // Compute relative position: memory_position - context_position
        let relative_position = memory_position - context_position;

        // Extract actual values for debugging
        let rel_pos_data = relative_position.to_data();
        println!("Relative position shape: {:?}", rel_pos_data.shape);
        let rel_pos_values: Vec<f64> = rel_pos_data.iter().collect();
        println!("Relative position values: {:?}", rel_pos_values);

        // Test bucket computation with simple parameters
        let bucket_result = T5Attention::<TestBackend>::relative_position_bucket(
            relative_position,
            true, // bidirectional
            8,    // num_buckets
            16,   // max_distance
            &device,
        );

        // Extract bucket result values
        let bucket_data = bucket_result.to_data();
        let bucket_values: Vec<f64> = bucket_data.iter().collect();
        println!("Bucket result values: {:?}", bucket_values);

        // Test conversion to int with careful checking
        let bucket_indices = bucket_result
            .round()
            .clamp(0.0, 7.0) // 8 buckets means indices 0-7
            .clamp(0.0, 31.0); // Use smaller max value for safety

        let indices_data = bucket_indices.to_data();
        let indices_values: Vec<f64> = indices_data.iter().collect();
        println!(
            "Bucket indices values before int conversion: {:?}",
            indices_values
        );

        // Check if all values are in valid range
        for value in &indices_values {
            assert!(
                *value >= 0.0 && *value <= 31.0 && value.is_finite(),
                "Invalid value for int conversion: {}",
                value
            );
        }

        // This is where the error occurs
        let _int_indices = bucket_indices.int();

        println!("✅ Relative position bucket test passed!");
    }

    #[test]
    fn test_relative_position_bucket_verification() {
        let device = NdArrayDevice::default();

        // Test parameters
        let query_length = 4;
        let key_length = 6;
        let num_buckets = 8;
        let max_distance = 16;

        // Test bidirectional case
        let context_position = Tensor::arange(0..query_length as i64, &device)
            .float()
            .unsqueeze_dim(1);
        let memory_position = Tensor::arange(0..key_length as i64, &device)
            .float()
            .unsqueeze_dim(0);

        let relative_position = memory_position - context_position;

        let bucket_result = T5Attention::<TestBackend>::relative_position_bucket(
            relative_position.clone(),
            true, // bidirectional
            num_buckets,
            max_distance,
            &device,
        );

        // Verify shape is preserved
        assert_eq!(bucket_result.dims(), [query_length, key_length]);

        // Extract values for verification
        let bucket_data = bucket_result.to_data();
        let bucket_values: Vec<f64> = bucket_data.iter().collect();

        // Verify all values are finite and in valid range
        for &value in &bucket_values {
            assert!(
                value.is_finite(),
                "Bucket value should be finite: {}",
                value
            );
            assert!(
                value >= 0.0,
                "Bucket value should be non-negative: {}",
                value
            );
            assert!(
                value < num_buckets as f64,
                "Bucket value should be less than num_buckets: {}",
                value
            );
        }

        // Test unidirectional case (decoder)
        let unidirectional_result = T5Attention::<TestBackend>::relative_position_bucket(
            relative_position,
            false, // unidirectional
            num_buckets,
            max_distance,
            &device,
        );

        let unidirectional_data = unidirectional_result.to_data();
        let unidirectional_values: Vec<f64> = unidirectional_data.iter().collect();

        // Verify unidirectional results are also valid
        for &value in &unidirectional_values {
            assert!(
                value.is_finite(),
                "Unidirectional bucket value should be finite: {}",
                value
            );
            assert!(
                value >= 0.0,
                "Unidirectional bucket value should be non-negative: {}",
                value
            );
            assert!(
                value < num_buckets as f64,
                "Unidirectional bucket value should be less than num_buckets: {}",
                value
            );
        }

        // Verify bidirectional and unidirectional produce different results (they should)
        let difference_count = bucket_values
            .iter()
            .zip(unidirectional_values.iter())
            .filter(|(a, b)| (*a - *b).abs() > 1e-6)
            .count();

        assert!(
            difference_count > 0,
            "Bidirectional and unidirectional should produce different results"
        );

        println!("✅ Relative position bucket verification test passed!");
    }
}
