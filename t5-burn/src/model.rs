use burn::{
    config::Config,
    nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig, Linear, LinearConfig},
    prelude::*,
};

use crate::feed_forward::{T5FeedForward, T5FeedForwardConfig};
use crate::layer_norm::{T5LayerNorm, T5LayerNormConfig};

#[derive(Config, Debug)]
pub struct T5ModelConfig {
    /// Vocabulary size of the model.
    ///
    /// Defines the number of different tokens that can be represented by the
    /// model.
    #[config(default = 32128)]
    vocab_size: usize,

    /// Size of the encoder layers and the pooler layer.
    #[config(default = 512)]
    d_model: usize,

    /// Size of the key, query, value projections per attention head.
    ///
    /// The `inner_dim` of the projection layer will be defined as
    /// `num_heads * d_kv`.
    #[config(default = 64)]
    d_kv: usize,

    /// Size of the intermediate feed forward layer in each `T5Block`.
    #[config(default = 2048)]
    d_ff: usize,

    /// Number of hidden layers in the Transformer encoder.
    #[config(default = 8)]
    n_layers: usize,

    /// Number of hidden layers in the transformer decoder.
    ///
    /// Defaults to the same value as `n_layers`.
    n_decoder_layers: Option<usize>,

    /// Number of attention heads for each attention layer in the Transformer encoder.
    #[config(default = 8)]
    n_heads: usize,

    /// The number of buckets to use for each attention layer.
    #[config(default = 32)]
    relative_attention_num_buckets: usize,

    /// The maximum distance of the longer sequences for the bucket separation.
    #[config(default = 128)]
    relative_attention_max_distance: usize,

    /// The ratio for all dropout layers.
    #[config(default = 0.1)]
    dropout_rate: f64,

    /// The dropout ratio for the classifier.
    #[config(default = 0.0)]
    classifier_dropout: f64,

    /// The epsilon used by the layer normalization layers.
    #[config(default = 1e-6)]
    layer_norm_eps: f64,

    /// A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
    /// testing).
    #[cfg(debug_assertions)]
    #[config(default = 1.0)]
    initializer_factor: f32,

    /// Type of feed forward layer to be used.
    ///
    /// Should be one of `"relu"` or `"gated-gelu"`. T5v1.1 uses the
    /// `"gated-gelu"` feed forward projection. Original T5 uses `"relu"`.
    #[config(default = "FeedForwardLayer::ReLU")]
    feed_forward_proj: FeedForwardLayer,

    /// Whether or not the model should return the last key/values attentions (not used by all models).
    #[config(default = true)]
    use_cache: bool,
}

/// The feed forward layer configuration.
#[derive(Config, Debug)]
pub enum FeedForwardLayer {
    /// The ReLU feed forward layer.
    ///
    /// This is the default feed forward layer in the original T5 paper.
    ReLU,
    /// The gated GELU feed forward layer.
    ///
    /// This is the feed forward layer used in T5v1.1.
    GatedGelu,
}

#[derive(Debug, Config, Copy, PartialEq)]
enum EncoderOrDecoder {
    Encoder,
    Decoder,
}

impl T5ModelConfig {
    /// Create a new `T5Model` with this configuration.
    pub fn init<B: Backend>(&self, device: &B::Device) -> T5Model<B> {
        let shared = EmbeddingConfig::new(self.vocab_size, self.d_model).init(device);
        T5Model {
            shared: shared.clone(),
            encoder: T5StackConfig::new(
                self.n_layers,
                EncoderOrDecoder::Encoder,
                self.relative_attention_num_buckets,
                self.relative_attention_max_distance,
                self.d_model,
                self.d_ff,
                self.d_kv,
                self.n_heads,
                self.layer_norm_eps,
                self.dropout_rate,
            )
            .init(device, shared.clone()),
            decoder: T5StackConfig::new(
                self.n_decoder_layers.unwrap_or(self.n_layers),
                EncoderOrDecoder::Decoder,
                self.relative_attention_num_buckets,
                self.relative_attention_max_distance,
                self.d_model,
                self.d_ff,
                self.d_kv,
                self.n_heads,
                self.layer_norm_eps,
                self.dropout_rate,
            )
            .init(device, shared),
        }
    }
}

#[derive(Module, Debug)]
pub struct T5Model<B: Backend> {
    shared: Embedding<B>,
    encoder: T5Stack<B>,
    decoder: T5Stack<B>,
}

#[derive(Module, Debug)]
struct T5Stack<B: Backend> {
    embed_tokens: Embedding<B>,
    blocks: Vec<T5Block<B>>,
    final_layer_norm: T5LayerNorm<B>,
    dropout: Dropout,
}

#[derive(Config, Debug)]
struct T5StackConfig {
    n_layers: usize,
    ty: EncoderOrDecoder,
    relative_attention_num_buckets: usize,
    relative_attention_max_distance: usize,
    d_model: usize,
    d_ff: usize,
    key_value_proj_dim: usize,
    num_heads: usize,
    layer_norm_epsilon: f64,
    dropout_rate: f64,
}

impl T5StackConfig {
    fn init<B: Backend>(&self, device: &B::Device, embed_tokens: Embedding<B>) -> T5Stack<B> {
        let blocks = (0..self.n_layers)
            .map(|i| {
                let has_relative_attention_bias = i == 0;
                T5BlockConfig::new(
                    self.ty,
                    self.relative_attention_num_buckets,
                    self.relative_attention_max_distance,
                    self.d_model,
                    self.d_ff,
                    self.key_value_proj_dim,
                    self.num_heads,
                    self.layer_norm_epsilon,
                    self.dropout_rate,
                    has_relative_attention_bias,
                    i,
                )
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

#[derive(Module, Debug)]
struct T5Block<B: Backend> {
    self_attention: T5SelfAttention<B>,
    cross_attention: Option<T5CrossAttention<B>>,
    ff: T5FeedForward<B>,
}

#[derive(Config, Debug)]
struct T5BlockConfig {
    ty: EncoderOrDecoder,
    relative_attention_num_buckets: usize,
    relative_attention_max_distance: usize,
    d_model: usize,
    d_ff: usize,
    key_value_proj_dim: usize,
    num_heads: usize,
    layer_norm_epsilon: f64,
    dropout_rate: f64,
    has_relative_attention_bias: bool,
    layer_idx: usize,
}

impl T5BlockConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> T5Block<B> {
        let self_attention = T5SelfAttentionConfig::new(
            self.ty,
            self.relative_attention_num_buckets,
            self.relative_attention_max_distance,
            self.d_model,
            self.key_value_proj_dim,
            self.num_heads,
            self.layer_norm_epsilon,
            self.dropout_rate,
            self.layer_idx,
            self.has_relative_attention_bias,
        )
        .init(device);
        let cross_attention = if self.ty == EncoderOrDecoder::Decoder {
            Some(
                T5CrossAttentionConfig::new(
                    self.ty,
                    self.relative_attention_num_buckets,
                    self.relative_attention_max_distance,
                    self.d_model,
                    self.key_value_proj_dim,
                    self.num_heads,
                    self.layer_norm_epsilon,
                    self.dropout_rate,
                    self.layer_idx,
                )
                .init(device),
            )
        } else {
            None
        };
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

struct AttentionOutput<B: Backend, const D: usize> {
    output: Tensor<B, D>,
    attentions: Option<Tensor<B, D>>,
}

struct T5AttentionInput<B: Backend, const D: usize> {
    hidden_states: Tensor<B, D>,
    attention_mask: Option<Tensor<B, D>>,
    key_value_states: Option<Tensor<B, D>>,
    position_bias: Option<Tensor<B, D>>,
    past_key_value: Option<Tensor<B, D>>,
    layer_head_mask: Option<Tensor<B, D>>,
    query_length: Option<usize>,
    use_cache: bool,
    output_attentions: bool,
    cache_position: Option<bool>,
}

struct T5SelfAttentionInput<B: Backend, const D: usize> {
    hidden_states: Tensor<B, D>,
    attention_mask: Option<Tensor<B, D>>,
    position_bias: Option<Tensor<B, D>>,
    layer_head_mask: Option<Tensor<B, D>>,
    past_key_value: Option<Tensor<B, D>>,
    use_cache: bool,
    query_length: Option<usize>,
    output_attentions: bool,
    cache_position: Option<bool>,
}

struct T5CrossAttentionInput<B: Backend, const D: usize> {
    hidden_states: Tensor<B, D>,
    key_value_states: Tensor<B, D>,
    attention_mask: Option<Tensor<B, D>>,
    position_bias: Option<Tensor<B, D>>,
    layer_head_mask: Option<Tensor<B, D>>,
    past_key_value: Option<Tensor<B, D>>,
    use_cache: bool,
    query_length: Option<usize>,
    output_attentions: bool,
    cache_position: Option<bool>,
}

#[derive(Module, Debug)]
struct T5SelfAttention<B: Backend> {
    self_attention: T5Attention<B>,
    layer_norm: T5LayerNorm<B>,
    dropout: Dropout,
}

impl<B: Backend> T5SelfAttention<B> {
    fn forward<const D: usize>(&self, input: T5SelfAttentionInput<B, D>) -> AttentionOutput<B, D> {
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

        let AttentionOutput { output, attentions } = self.self_attention.forward(attention_input);
        let layer_output = input.hidden_states + self.dropout.forward(output);
        AttentionOutput {
            output: layer_output,
            attentions,
        }
    }
}

#[derive(Config, Debug)]
struct T5SelfAttentionConfig {
    ty: EncoderOrDecoder,
    relative_attention_num_buckets: usize,
    relative_attention_max_distance: usize,
    d_model: usize,
    key_value_proj_dim: usize,
    num_heads: usize,
    layer_norm_epsilon: f64,
    dropout_rate: f64,
    layer_idx: usize,
    has_relative_attention_bias: bool,
}

impl T5SelfAttentionConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> T5SelfAttention<B> {
        let self_attention = T5AttentionConfig::new(
            self.ty,
            self.relative_attention_num_buckets,
            self.relative_attention_max_distance,
            self.d_model,
            self.key_value_proj_dim,
            self.num_heads,
            self.dropout_rate,
            self.layer_idx,
        )
        .with_has_relative_attention_bias(self.has_relative_attention_bias)
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

#[derive(Module, Debug)]
struct T5CrossAttention<B: Backend> {
    cross_attention: T5Attention<B>,
    layer_norm: T5LayerNorm<B>,
    dropout: Dropout,
}

impl<B: Backend> T5CrossAttention<B> {
    fn forward<const D: usize>(&self, input: T5CrossAttentionInput<B, D>) -> AttentionOutput<B, D> {
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

        let AttentionOutput { output, attentions } = self.cross_attention.forward(attention_input);
        let layer_output = input.hidden_states + self.dropout.forward(output);
        AttentionOutput {
            output: layer_output,
            attentions,
        }
    }
}

#[derive(Config, Debug)]
struct T5CrossAttentionConfig {
    ty: EncoderOrDecoder,
    relative_attention_num_buckets: usize,
    relative_attention_max_distance: usize,
    d_model: usize,
    key_value_proj_dim: usize,
    num_heads: usize,
    layer_norm_epsilon: f64,
    dropout_rate: f64,
    layer_idx: usize,
}

impl T5CrossAttentionConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> T5CrossAttention<B> {
        let cross_attention = T5AttentionConfig::new(
            self.ty,
            self.relative_attention_num_buckets,
            self.relative_attention_max_distance,
            self.d_model,
            self.key_value_proj_dim,
            self.num_heads,
            self.dropout_rate,
            self.layer_idx,
        )
        .with_has_relative_attention_bias(false)
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

#[derive(Module, Debug)]
struct T5Attention<B: Backend> {
    query: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
    dense: Linear<B>,
    relative_attention_bias: Option<Embedding<B>>,
}

impl<B: Backend> T5Attention<B> {
    /// Forward pass through the attention mechanism
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
    /// let output = attention.forward(input);
    /// ```
    fn forward<const D: usize>(&self, input: T5AttentionInput<B, D>) -> AttentionOutput<B, D> {
        todo!()
    }
}

#[derive(Config, Debug)]
struct T5AttentionConfig {
    ty: EncoderOrDecoder,
    #[config(default = false)]
    has_relative_attention_bias: bool,
    relative_attention_num_buckets: usize,
    relative_attention_max_distance: usize,
    d_model: usize,
    key_value_proj_dim: usize,
    num_heads: usize,
    dropout_rate: f64,
    layer_idx: usize,
}

impl T5AttentionConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> T5Attention<B> {
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

        T5Attention {
            query,
            key,
            value,
            dense,
            relative_attention_bias,
        }
    }
}
