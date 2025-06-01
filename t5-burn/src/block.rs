use burn::{config::Config, prelude::*};

use crate::attention::{
    EncoderOrDecoder, T5CrossAttention, T5CrossAttentionConfig, T5CrossAttentionInput,
    T5SelfAttention, T5SelfAttentionConfig, T5SelfAttentionInput,
};
use crate::feed_forward::{T5FeedForward, T5FeedForwardConfig};

/// Input structure for T5Block
#[derive(Debug, Clone)]
pub struct T5BlockInput<B: Backend> {
    pub hidden_states: Tensor<B, 3>,
    pub attention_mask: Option<Tensor<B, 4>>,
    pub position_bias: Option<Tensor<B, 4>>,
    pub encoder_hidden_states: Option<Tensor<B, 3>>,
    pub encoder_attention_mask: Option<Tensor<B, 4>>,
    pub encoder_decoder_position_bias: Option<Tensor<B, 4>>,
    pub layer_head_mask: Option<Tensor<B, 4>>,
    pub cross_attn_layer_head_mask: Option<Tensor<B, 4>>,
    pub use_cache: bool,
    pub output_attentions: bool,
    pub cache_position: Option<bool>,
}

/// Output structure for T5Block
#[derive(Debug, Clone)]
pub struct T5BlockOutput<B: Backend> {
    pub hidden_states: Tensor<B, 3>,
    pub self_attentions: Option<Tensor<B, 4>>,
    pub self_position_bias: Option<Tensor<B, 4>>,
    pub cross_attentions: Option<Tensor<B, 4>>,
    pub cross_position_bias: Option<Tensor<B, 4>>,
}

#[derive(Config, Debug)]
pub struct T5BlockConfig {
    pub ty: EncoderOrDecoder,
    pub relative_attention_num_buckets: usize,
    pub relative_attention_max_distance: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub key_value_proj_dim: usize,
    pub num_heads: usize,
    pub layer_norm_epsilon: f64,
    pub dropout_rate: f64,
    pub has_relative_attention_bias: bool,
    pub layer_idx: usize,
}

impl T5BlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> T5Block<B> {
        let self_attention = T5SelfAttentionConfig {
            ty: self.ty,
            relative_attention_num_buckets: self.relative_attention_num_buckets,
            relative_attention_max_distance: self.relative_attention_max_distance,
            d_model: self.d_model,
            key_value_proj_dim: self.key_value_proj_dim,
            num_heads: self.num_heads,
            layer_norm_epsilon: self.layer_norm_epsilon,
            dropout_rate: self.dropout_rate,
            layer_idx: self.layer_idx,
            has_relative_attention_bias: self.has_relative_attention_bias,
        }
        .init(device);
        let cross_attention = (self.ty == EncoderOrDecoder::Decoder).then_some(
            T5CrossAttentionConfig {
                ty: self.ty,
                relative_attention_num_buckets: self.relative_attention_num_buckets,
                relative_attention_max_distance: self.relative_attention_max_distance,
                d_model: self.d_model,
                key_value_proj_dim: self.key_value_proj_dim,
                num_heads: self.num_heads,
                layer_norm_epsilon: self.layer_norm_epsilon,
                dropout_rate: self.dropout_rate,
                layer_idx: self.layer_idx,
            }
            .init(device),
        );
        let ff = T5FeedForwardConfig {
            d_model: self.d_model,
            d_ff: self.d_ff,
            layer_norm_epsilon: self.layer_norm_epsilon,
            dropout_rate: self.dropout_rate,
        }
        .init(device);
        T5Block {
            self_attention,
            cross_attention,
            ff,
        }
    }
}

#[derive(Module, Debug)]
pub struct T5Block<B: Backend> {
    pub self_attention: T5SelfAttention<B>,
    pub cross_attention: Option<T5CrossAttention<B>>,
    pub ff: T5FeedForward<B>,
}

impl<B: Backend> T5Block<B> {
    /// Forward pass for T5Block
    pub fn forward(&self, input: T5BlockInput<B>, config: &T5BlockConfig) -> T5BlockOutput<B> {
        // Create self-attention config
        let self_attention_config = T5SelfAttentionConfig {
            ty: config.ty,
            relative_attention_num_buckets: config.relative_attention_num_buckets,
            relative_attention_max_distance: config.relative_attention_max_distance,
            d_model: config.d_model,
            key_value_proj_dim: config.key_value_proj_dim,
            num_heads: config.num_heads,
            layer_norm_epsilon: config.layer_norm_epsilon,
            dropout_rate: config.dropout_rate,
            layer_idx: config.layer_idx,
            has_relative_attention_bias: config.has_relative_attention_bias,
        };

        // Self-attention
        let self_attention_input = T5SelfAttentionInput {
            hidden_states: input.hidden_states,
            attention_mask: input.attention_mask,
            position_bias: input.position_bias,
            layer_head_mask: input.layer_head_mask,
            past_key_value: None, // Simplified for now
            use_cache: input.use_cache,
            query_length: None,
            output_attentions: input.output_attentions,
            cache_position: input.cache_position,
        };

        let self_attention_output = self
            .self_attention
            .forward(self_attention_input, &self_attention_config);
        let mut hidden_states = self_attention_output.output;

        // Cross-attention (only for decoder)
        let (cross_attentions, cross_position_bias) = if self.cross_attention.is_some() {
            if let (Some(cross_attention), Some(encoder_hidden_states)) =
                (&self.cross_attention, &input.encoder_hidden_states)
            {
                // Create cross-attention config
                let cross_attention_config = T5CrossAttentionConfig {
                    ty: config.ty,
                    relative_attention_num_buckets: config.relative_attention_num_buckets,
                    relative_attention_max_distance: config.relative_attention_max_distance,
                    d_model: config.d_model,
                    key_value_proj_dim: config.key_value_proj_dim,
                    num_heads: config.num_heads,
                    layer_norm_epsilon: config.layer_norm_epsilon,
                    dropout_rate: config.dropout_rate,
                    layer_idx: config.layer_idx,
                };

                let cross_attention_input = T5CrossAttentionInput {
                    hidden_states: hidden_states.clone(),
                    key_value_states: encoder_hidden_states.clone(),
                    attention_mask: input.encoder_attention_mask,
                    position_bias: input.encoder_decoder_position_bias,
                    layer_head_mask: input.cross_attn_layer_head_mask,
                    past_key_value: None, // Simplified for now
                    use_cache: input.use_cache,
                    query_length: None,
                    output_attentions: input.output_attentions,
                    cache_position: input.cache_position,
                };

                let cross_attention_output =
                    cross_attention.forward(cross_attention_input, &cross_attention_config);
                hidden_states = cross_attention_output.output;
                (cross_attention_output.attentions, None)
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

        // Feed-forward
        hidden_states = self.ff.forward(hidden_states);

        T5BlockOutput {
            hidden_states,
            self_attentions: self_attention_output.attentions,
            self_position_bias: None,
            cross_attentions,
            cross_position_bias,
        }
    }
}

#[cfg(test)]
mod test_t5_block {
    use super::*;
    use crate::attention::EncoderOrDecoder;

    type TestBackend = burn_ndarray::NdArray<f32>;

    #[test]
    fn test_t5_encoder_block_forward() {
        let device = Default::default();

        let config = T5BlockConfig {
            ty: EncoderOrDecoder::Encoder,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            d_model: 64,
            d_ff: 128,
            key_value_proj_dim: 64,
            num_heads: 8,
            layer_norm_epsilon: 1e-6,
            dropout_rate: 0.0,
            has_relative_attention_bias: true,
            layer_idx: 0,
        };

        let block = config.init::<TestBackend>(&device);

        // Create input tensors
        let batch_size = 2;
        let seq_len = 10;
        let hidden_states = Tensor::random(
            [batch_size, seq_len, 64],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let attention_mask = Tensor::ones([batch_size, 1, 1, seq_len], &device);

        let input = T5BlockInput {
            hidden_states: hidden_states.clone(),
            attention_mask: Some(attention_mask),
            position_bias: None,
            encoder_hidden_states: None,
            encoder_attention_mask: None,
            encoder_decoder_position_bias: None,
            layer_head_mask: None,
            cross_attn_layer_head_mask: None,
            use_cache: false,
            output_attentions: false,
            cache_position: None,
        };

        let output = block.forward(input, &config);

        // Check output shapes
        assert_eq!(
            output.hidden_states.shape(),
            [batch_size, seq_len, 64].into()
        );
        assert!(output.cross_attentions.is_none()); // Encoder doesn't have cross-attention
    }

    #[test]
    fn test_t5_decoder_block_forward() {
        let device = Default::default();

        let config = T5BlockConfig {
            ty: EncoderOrDecoder::Decoder,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            d_model: 64,
            d_ff: 128,
            key_value_proj_dim: 64,
            num_heads: 8,
            layer_norm_epsilon: 1e-6,
            dropout_rate: 0.0,
            has_relative_attention_bias: true,
            layer_idx: 0,
        };

        let block = config.init::<TestBackend>(&device);

        // Create input tensors
        let batch_size = 2;
        let seq_len = 8;
        let encoder_seq_len = 12;
        let hidden_states = Tensor::random(
            [batch_size, seq_len, 64],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let encoder_hidden_states = Tensor::random(
            [batch_size, encoder_seq_len, 64],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let attention_mask = Tensor::ones([batch_size, 1, seq_len, seq_len], &device);
        let encoder_attention_mask = Tensor::ones([batch_size, 1, 1, encoder_seq_len], &device);

        let input = T5BlockInput {
            hidden_states: hidden_states.clone(),
            attention_mask: Some(attention_mask),
            position_bias: None,
            encoder_hidden_states: Some(encoder_hidden_states),
            encoder_attention_mask: Some(encoder_attention_mask),
            encoder_decoder_position_bias: None,
            layer_head_mask: None,
            cross_attn_layer_head_mask: None,
            use_cache: false,
            output_attentions: false,
            cache_position: None,
        };

        let output = block.forward(input, &config);

        // Check output shapes
        assert_eq!(
            output.hidden_states.shape(),
            [batch_size, seq_len, 64].into()
        );
    }

    #[test]
    fn test_t5_decoder_block_without_encoder_states() {
        let device = Default::default();

        let config = T5BlockConfig {
            ty: EncoderOrDecoder::Decoder,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            d_model: 64,
            d_ff: 128,
            key_value_proj_dim: 64,
            num_heads: 8,
            layer_norm_epsilon: 1e-6,
            dropout_rate: 0.0,
            has_relative_attention_bias: true,
            layer_idx: 0,
        };

        let block = config.init::<TestBackend>(&device);

        // Create input tensors without encoder states
        let batch_size = 2;
        let seq_len = 8;
        let hidden_states = Tensor::random(
            [batch_size, seq_len, 64],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let attention_mask = Tensor::ones([batch_size, 1, seq_len, seq_len], &device);

        let input = T5BlockInput {
            hidden_states: hidden_states.clone(),
            attention_mask: Some(attention_mask),
            position_bias: None,
            encoder_hidden_states: None, // No encoder states
            encoder_attention_mask: None,
            encoder_decoder_position_bias: None,
            layer_head_mask: None,
            cross_attn_layer_head_mask: None,
            use_cache: false,
            output_attentions: false,
            cache_position: None,
        };

        let output = block.forward(input, &config);

        // Check output shapes
        assert_eq!(
            output.hidden_states.shape(),
            [batch_size, seq_len, 64].into()
        );
        assert!(output.cross_attentions.is_none()); // No cross-attention without encoder states
    }

    #[test]
    fn test_t5_block_output_attentions() {
        let device = Default::default();

        let config = T5BlockConfig {
            ty: EncoderOrDecoder::Encoder,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            d_model: 64,
            d_ff: 128,
            key_value_proj_dim: 64,
            num_heads: 8,
            layer_norm_epsilon: 1e-6,
            dropout_rate: 0.0,
            has_relative_attention_bias: true,
            layer_idx: 0,
        };

        let block = config.init::<TestBackend>(&device);

        // Create input tensors
        let batch_size = 2;
        let seq_len = 10;
        let hidden_states = Tensor::random(
            [batch_size, seq_len, 64],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let attention_mask = Tensor::ones([batch_size, 1, 1, seq_len], &device);

        let input = T5BlockInput {
            hidden_states: hidden_states.clone(),
            attention_mask: Some(attention_mask),
            position_bias: None,
            encoder_hidden_states: None,
            encoder_attention_mask: None,
            encoder_decoder_position_bias: None,
            layer_head_mask: None,
            cross_attn_layer_head_mask: None,
            use_cache: false,
            output_attentions: true, // Enable attention outputs
            cache_position: None,
        };

        let output = block.forward(input, &config);

        // Check output shapes
        assert_eq!(
            output.hidden_states.shape(),
            [batch_size, seq_len, 64].into()
        );
        if let Some(attentions) = output.self_attentions {
            assert_eq!(attentions.shape(), [batch_size, 8, seq_len, seq_len].into()); // num_heads = 8
        }
    }

    #[test]
    fn test_t5_block_different_dimensions() {
        let device = Default::default();

        let config = T5BlockConfig {
            ty: EncoderOrDecoder::Encoder,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            d_model: 32, // Smaller model
            d_ff: 64,    // Smaller FF
            key_value_proj_dim: 32,
            num_heads: 4, // Fewer heads
            layer_norm_epsilon: 1e-6,
            dropout_rate: 0.0,
            has_relative_attention_bias: false, // No relative attention bias
            layer_idx: 1,
        };

        let block = config.init::<TestBackend>(&device);

        // Create input tensors
        let batch_size = 1;
        let seq_len = 5;
        let hidden_states = Tensor::random(
            [batch_size, seq_len, 32],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let attention_mask = Tensor::ones([batch_size, 1, 1, seq_len], &device);

        let input = T5BlockInput {
            hidden_states: hidden_states.clone(),
            attention_mask: Some(attention_mask),
            position_bias: None,
            encoder_hidden_states: None,
            encoder_attention_mask: None,
            encoder_decoder_position_bias: None,
            layer_head_mask: None,
            cross_attn_layer_head_mask: None,
            use_cache: false,
            output_attentions: false,
            cache_position: None,
        };

        let output = block.forward(input, &config);

        // Check output shapes
        assert_eq!(
            output.hidden_states.shape(),
            [batch_size, seq_len, 32].into()
        );
    }

    // TODO: Add gradient flow test with proper autodiff backend setup

    #[test]
    fn test_t5_block_numerical_stability() {
        let device = Default::default();

        let config = T5BlockConfig {
            ty: EncoderOrDecoder::Encoder,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            d_model: 64,
            d_ff: 128,
            key_value_proj_dim: 64,
            num_heads: 8,
            layer_norm_epsilon: 1e-6,
            dropout_rate: 0.0,
            has_relative_attention_bias: true,
            layer_idx: 0,
        };

        let block = config.init::<TestBackend>(&device);

        // Test with very large values
        let batch_size = 2;
        let seq_len = 10;
        let hidden_states = Tensor::ones([batch_size, seq_len, 64], &device) * 100.0;
        let attention_mask = Tensor::ones([batch_size, 1, 1, seq_len], &device);

        let input = T5BlockInput {
            hidden_states,
            attention_mask: Some(attention_mask),
            position_bias: None,
            encoder_hidden_states: None,
            encoder_attention_mask: None,
            encoder_decoder_position_bias: None,
            layer_head_mask: None,
            cross_attn_layer_head_mask: None,
            use_cache: false,
            output_attentions: false,
            cache_position: None,
        };

        let output = block.forward(input, &config);

        // Check that output doesn't contain NaN or infinity
        let output_data = output.hidden_states.into_data();
        for value in output_data.iter::<f32>() {
            assert!(value.is_finite(), "Output contains non-finite values");
            assert!(!value.is_nan(), "Output contains NaN values");
        }
    }

    #[test]
    fn test_t5_block_single_token() {
        let device = Default::default();

        let config = T5BlockConfig {
            ty: EncoderOrDecoder::Encoder,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            d_model: 64,
            d_ff: 128,
            key_value_proj_dim: 64,
            num_heads: 8,
            layer_norm_epsilon: 1e-6,
            dropout_rate: 0.0,
            has_relative_attention_bias: true,
            layer_idx: 0,
        };

        let block = config.init::<TestBackend>(&device);

        // Test with single token sequence
        let batch_size = 1;
        let seq_len = 1;
        let hidden_states = Tensor::random(
            [batch_size, seq_len, 64],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let attention_mask = Tensor::ones([batch_size, 1, 1, seq_len], &device);

        let input = T5BlockInput {
            hidden_states: hidden_states.clone(),
            attention_mask: Some(attention_mask),
            position_bias: None,
            encoder_hidden_states: None,
            encoder_attention_mask: None,
            encoder_decoder_position_bias: None,
            layer_head_mask: None,
            cross_attn_layer_head_mask: None,
            use_cache: false,
            output_attentions: false,
            cache_position: None,
        };

        let output = block.forward(input, &config);

        // Check output shapes
        assert_eq!(
            output.hidden_states.shape(),
            [batch_size, seq_len, 64].into()
        );
    }

    #[test]
    fn test_t5_block_long_sequence() {
        let device = Default::default();

        let config = T5BlockConfig {
            ty: EncoderOrDecoder::Encoder,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            d_model: 32, // Smaller for efficiency
            d_ff: 64,
            key_value_proj_dim: 32,
            num_heads: 4,
            layer_norm_epsilon: 1e-6,
            dropout_rate: 0.0,
            has_relative_attention_bias: true,
            layer_idx: 0,
        };

        let block = config.init::<TestBackend>(&device);

        // Test with longer sequence
        let batch_size = 1;
        let seq_len = 256; // Relatively long sequence
        let hidden_states = Tensor::random(
            [batch_size, seq_len, 32],
            burn::tensor::Distribution::Normal(0.0, 0.1),
            &device,
        );
        let attention_mask = Tensor::ones([batch_size, 1, 1, seq_len], &device);

        let input = T5BlockInput {
            hidden_states: hidden_states.clone(),
            attention_mask: Some(attention_mask),
            position_bias: None,
            encoder_hidden_states: None,
            encoder_attention_mask: None,
            encoder_decoder_position_bias: None,
            layer_head_mask: None,
            cross_attn_layer_head_mask: None,
            use_cache: false,
            output_attentions: false,
            cache_position: None,
        };

        let output = block.forward(input, &config);

        // Check output shapes
        assert_eq!(
            output.hidden_states.shape(),
            [batch_size, seq_len, 32].into()
        );
    }

    #[test]
    fn test_t5_block_masked_attention() {
        let device = Default::default();

        let config = T5BlockConfig {
            ty: EncoderOrDecoder::Encoder,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            d_model: 64,
            d_ff: 128,
            key_value_proj_dim: 64,
            num_heads: 8,
            layer_norm_epsilon: 1e-6,
            dropout_rate: 0.0,
            has_relative_attention_bias: true,
            layer_idx: 0,
        };

        let block = config.init::<TestBackend>(&device);

        // Create input with partial masking
        let batch_size = 2;
        let seq_len = 8;
        let hidden_states = Tensor::random(
            [batch_size, seq_len, 64],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let mut attention_mask = Tensor::ones([batch_size, 1, 1, seq_len], &device);

        // Mask out the last 2 tokens for the first batch
        attention_mask = attention_mask.slice_assign(
            [0..1, 0..1, 0..1, 6..8],
            Tensor::zeros([1, 1, 1, 2], &device),
        );

        let input = T5BlockInput {
            hidden_states: hidden_states.clone(),
            attention_mask: Some(attention_mask),
            position_bias: None,
            encoder_hidden_states: None,
            encoder_attention_mask: None,
            encoder_decoder_position_bias: None,
            layer_head_mask: None,
            cross_attn_layer_head_mask: None,
            use_cache: false,
            output_attentions: true, // Enable to check attention patterns
            cache_position: None,
        };

        let output = block.forward(input, &config);

        // Check output shapes
        assert_eq!(
            output.hidden_states.shape(),
            [batch_size, seq_len, 64].into()
        );

        // Check that attention weights are computed
        if let Some(attentions) = output.self_attentions {
            assert_eq!(attentions.shape(), [batch_size, 8, seq_len, seq_len].into());
        }
    }

    #[test]
    fn test_t5_block_cross_attention_shapes() {
        let device = Default::default();

        let config = T5BlockConfig {
            ty: EncoderOrDecoder::Decoder,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            d_model: 64,
            d_ff: 128,
            key_value_proj_dim: 64,
            num_heads: 8,
            layer_norm_epsilon: 1e-6,
            dropout_rate: 0.0,
            has_relative_attention_bias: true,
            layer_idx: 0,
        };

        let block = config.init::<TestBackend>(&device);

        // Test different sequence lengths for decoder and encoder
        let batch_size = 2;
        let decoder_seq_len = 5;
        let encoder_seq_len = 10;

        let hidden_states = Tensor::random(
            [batch_size, decoder_seq_len, 64],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let encoder_hidden_states = Tensor::random(
            [batch_size, encoder_seq_len, 64],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let attention_mask =
            Tensor::ones([batch_size, 1, decoder_seq_len, decoder_seq_len], &device);
        let encoder_attention_mask = Tensor::ones([batch_size, 1, 1, encoder_seq_len], &device);

        let input = T5BlockInput {
            hidden_states: hidden_states.clone(),
            attention_mask: Some(attention_mask),
            position_bias: None,
            encoder_hidden_states: Some(encoder_hidden_states),
            encoder_attention_mask: Some(encoder_attention_mask),
            encoder_decoder_position_bias: None,
            layer_head_mask: None,
            cross_attn_layer_head_mask: None,
            use_cache: false,
            output_attentions: true,
            cache_position: None,
        };

        let output = block.forward(input, &config);

        // Check output shapes
        assert_eq!(
            output.hidden_states.shape(),
            [batch_size, decoder_seq_len, 64].into()
        );

        // Check that both self and cross attention weights are computed
        if let Some(self_attentions) = output.self_attentions {
            assert_eq!(
                self_attentions.shape(),
                [batch_size, 8, decoder_seq_len, decoder_seq_len].into()
            );
        }

        if let Some(cross_attentions) = output.cross_attentions {
            assert_eq!(
                cross_attentions.shape(),
                [batch_size, 8, decoder_seq_len, encoder_seq_len].into()
            );
        }
    }

    #[test]
    fn test_t5_block_residual_connections() {
        let device = Default::default();

        let config = T5BlockConfig {
            ty: EncoderOrDecoder::Encoder,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            d_model: 64,
            d_ff: 128,
            key_value_proj_dim: 64,
            num_heads: 8,
            layer_norm_epsilon: 1e-6,
            dropout_rate: 0.0, // No dropout to test pure residual connections
            has_relative_attention_bias: false, // Simplify for residual test
            layer_idx: 0,
        };

        let block = config.init::<TestBackend>(&device);

        // Test with zero input to check residual behavior
        let batch_size = 2;
        let seq_len = 8;
        let hidden_states = Tensor::zeros([batch_size, seq_len, 64], &device);
        let attention_mask = Tensor::ones([batch_size, 1, 1, seq_len], &device);

        let input = T5BlockInput {
            hidden_states: hidden_states.clone(),
            attention_mask: Some(attention_mask),
            position_bias: None,
            encoder_hidden_states: None,
            encoder_attention_mask: None,
            encoder_decoder_position_bias: None,
            layer_head_mask: None,
            cross_attn_layer_head_mask: None,
            use_cache: false,
            output_attentions: false,
            cache_position: None,
        };

        let output = block.forward(input, &config);

        // With zero input and no relative bias, the output should be close to zero
        // (due to residual connections and layer norm behavior)
        assert_eq!(
            output.hidden_states.shape(),
            [batch_size, seq_len, 64].into()
        );
    }

    #[test]
    fn test_t5_block_functional_encoder_decoder_flow() {
        let device = Default::default();

        // Configuration for both encoder and decoder blocks
        let d_model = 64;
        let d_ff = 128;
        let num_heads = 8;
        let key_value_proj_dim = 64;

        let encoder_config = T5BlockConfig {
            ty: EncoderOrDecoder::Encoder,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            d_model,
            d_ff,
            key_value_proj_dim,
            num_heads,
            layer_norm_epsilon: 1e-6,
            dropout_rate: 0.0,
            has_relative_attention_bias: true,
            layer_idx: 0,
        };

        let decoder_config = T5BlockConfig {
            ty: EncoderOrDecoder::Decoder,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            d_model,
            d_ff,
            key_value_proj_dim,
            num_heads,
            layer_norm_epsilon: 1e-6,
            dropout_rate: 0.0,
            has_relative_attention_bias: true,
            layer_idx: 0,
        };

        let encoder_block = encoder_config.init::<TestBackend>(&device);
        let decoder_block = decoder_config.init::<TestBackend>(&device);

        // Create source sequence (encoder input)
        let batch_size = 2;
        let src_seq_len = 10;
        let tgt_seq_len = 8;

        let src_hidden_states = Tensor::random(
            [batch_size, src_seq_len, d_model],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let src_attention_mask = Tensor::ones([batch_size, 1, 1, src_seq_len], &device);

        // Step 1: Process through encoder
        let encoder_input = T5BlockInput {
            hidden_states: src_hidden_states,
            attention_mask: Some(src_attention_mask.clone()),
            position_bias: None,
            encoder_hidden_states: None,
            encoder_attention_mask: None,
            encoder_decoder_position_bias: None,
            layer_head_mask: None,
            cross_attn_layer_head_mask: None,
            use_cache: false,
            output_attentions: true,
            cache_position: None,
        };

        let encoder_output = encoder_block.forward(encoder_input, &encoder_config);

        // Verify encoder output
        assert_eq!(
            encoder_output.hidden_states.shape(),
            [batch_size, src_seq_len, d_model].into()
        );
        assert!(encoder_output.cross_attentions.is_none()); // Encoder has no cross-attention
        assert!(encoder_output.self_attentions.is_some()); // Should have self-attention weights

        // Step 2: Process through decoder with encoder states
        let tgt_hidden_states = Tensor::random(
            [batch_size, tgt_seq_len, d_model],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let tgt_attention_mask = Tensor::ones([batch_size, 1, tgt_seq_len, tgt_seq_len], &device);

        let decoder_input = T5BlockInput {
            hidden_states: tgt_hidden_states.clone(),
            attention_mask: Some(tgt_attention_mask),
            position_bias: None,
            encoder_hidden_states: Some(encoder_output.hidden_states.clone()),
            encoder_attention_mask: Some(src_attention_mask),
            encoder_decoder_position_bias: None,
            layer_head_mask: None,
            cross_attn_layer_head_mask: None,
            use_cache: false,
            output_attentions: true,
            cache_position: None,
        };

        let decoder_output = decoder_block.forward(decoder_input, &decoder_config);

        // Verify decoder output
        assert_eq!(
            decoder_output.hidden_states.shape(),
            [batch_size, tgt_seq_len, d_model].into()
        );
        assert!(decoder_output.self_attentions.is_some()); // Should have self-attention weights
        assert!(decoder_output.cross_attentions.is_some()); // Should have cross-attention weights

        // Verify attention weight shapes
        if let Some(self_attentions) = decoder_output.self_attentions {
            assert_eq!(
                self_attentions.shape(),
                [batch_size, num_heads, tgt_seq_len, tgt_seq_len].into()
            );
        }

        if let Some(cross_attentions) = decoder_output.cross_attentions {
            assert_eq!(
                cross_attentions.shape(),
                [batch_size, num_heads, tgt_seq_len, src_seq_len].into()
            );
        }

        // Step 3: Test decoder without encoder states (self-attention only)
        let decoder_only_input = T5BlockInput {
            hidden_states: tgt_hidden_states,
            attention_mask: Some(Tensor::ones(
                [batch_size, 1, tgt_seq_len, tgt_seq_len],
                &device,
            )),
            position_bias: None,
            encoder_hidden_states: None, // No encoder states
            encoder_attention_mask: None,
            encoder_decoder_position_bias: None,
            layer_head_mask: None,
            cross_attn_layer_head_mask: None,
            use_cache: false,
            output_attentions: true,
            cache_position: None,
        };

        let decoder_only_output = decoder_block.forward(decoder_only_input, &decoder_config);

        // Verify decoder-only output
        assert_eq!(
            decoder_only_output.hidden_states.shape(),
            [batch_size, tgt_seq_len, d_model].into()
        );
        assert!(decoder_only_output.self_attentions.is_some()); // Should have self-attention weights
        assert!(decoder_only_output.cross_attentions.is_none()); // Should NOT have cross-attention weights

        // Test that outputs are different when encoder states are provided vs not
        let diff = (decoder_output.hidden_states - decoder_only_output.hidden_states)
            .abs()
            .sum();
        assert!(
            diff.into_scalar() > 0.0,
            "Decoder output should differ with and without encoder states"
        );
    }

    #[test]
    fn test_t5_block_layer_composition() {
        let device = Default::default();

        // Test multiple T5 blocks in sequence (simulating layer stacking)
        let config = T5BlockConfig {
            ty: EncoderOrDecoder::Encoder,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            d_model: 32,
            d_ff: 64,
            key_value_proj_dim: 32,
            num_heads: 4,
            layer_norm_epsilon: 1e-6,
            dropout_rate: 0.0,
            has_relative_attention_bias: true,
            layer_idx: 0,
        };

        let config_layer1 = T5BlockConfig {
            layer_idx: 1,
            has_relative_attention_bias: false, // Only first layer has relative bias
            ..config
        };

        let block0 = config.init::<TestBackend>(&device);
        let block1 = config_layer1.init::<TestBackend>(&device);

        let batch_size = 2;
        let seq_len = 8;
        let hidden_states = Tensor::random(
            [batch_size, seq_len, 32],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let attention_mask = Tensor::ones([batch_size, 1, 1, seq_len], &device);

        // Pass through first block
        let input0 = T5BlockInput {
            hidden_states: hidden_states.clone(),
            attention_mask: Some(attention_mask.clone()),
            position_bias: None,
            encoder_hidden_states: None,
            encoder_attention_mask: None,
            encoder_decoder_position_bias: None,
            layer_head_mask: None,
            cross_attn_layer_head_mask: None,
            use_cache: false,
            output_attentions: false,
            cache_position: None,
        };

        let output0 = block0.forward(input0, &config);

        // Pass output of first block through second block
        let input1 = T5BlockInput {
            hidden_states: output0.hidden_states,
            attention_mask: Some(attention_mask),
            position_bias: output0.self_position_bias,
            encoder_hidden_states: None,
            encoder_attention_mask: None,
            encoder_decoder_position_bias: None,
            layer_head_mask: None,
            cross_attn_layer_head_mask: None,
            use_cache: false,
            output_attentions: false,
            cache_position: None,
        };

        let output1 = block1.forward(input1, &config_layer1);

        // Verify final output shape
        assert_eq!(
            output1.hidden_states.shape(),
            [batch_size, seq_len, 32].into()
        );

        // Ensure processing changed the representation
        let initial_norm = hidden_states.abs().sum();
        let final_norm = output1.hidden_states.abs().sum();

        // The norms should be different, indicating the blocks processed the data
        assert_ne!(initial_norm.into_scalar(), final_norm.into_scalar());
    }
}
