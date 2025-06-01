use burn::{
    config::Config,
    nn::{
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, Gelu, Linear, LinearConfig, RmsNorm,
        RmsNormConfig,
    },
    prelude::*,
};

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
            final_layer_norm: T5LayerNormConfig::new(self.d_model, self.layer_norm_epsilon)
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
        let ff = T5FeedForwardConfig::new(
            self.d_model,
            self.d_ff,
            self.layer_norm_epsilon,
            self.dropout_rate,
        )
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
    attentions: Vec<Tensor<B, D>>,
}

#[derive(Module, Debug)]
struct T5SelfAttention<B: Backend> {
    self_attention: T5Attention<B>,
    layer_norm: T5LayerNorm<B>,
    dropout: Dropout,
}

impl<B: Backend> T5SelfAttention<B> {
    fn forward<const D: usize>(
        &self,
        hidden_states: Tensor<B, D>,
        attention_mask: Option<Tensor<B, D>>,
        position_bias: Option<Tensor<B, D>>,
        layer_head_mask: Option<Tensor<B, D>>,
        past_key_value: Option<Tensor<B, D>>,
        use_cache: bool,
        query_length: Option<usize>,
        output_attentions: bool,
        cache_position: Option<bool>,
    ) -> AttentionOutput<B, D> {
        let normed_hidden_states = self.layer_norm.forward(hidden_states.clone());
        let AttentionOutput { output, attentions } = self.self_attention.forward(
            normed_hidden_states,
            attention_mask,
            None,
            position_bias,
            past_key_value,
            layer_head_mask,
            query_length,
            use_cache,
            output_attentions,
            cache_position,
        );
        let layer_output = hidden_states + self.dropout.forward(output);
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
        let layer_norm = T5LayerNormConfig::new(self.d_model, self.layer_norm_epsilon).init(device);
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
    fn forward<const D: usize>(
        &self,
        hidden_states: Tensor<B, D>,
        key_value_states: Tensor<B, D>,
        attention_mask: Option<Tensor<B, D>>,
        position_bias: Option<Tensor<B, D>>,
    ) -> AttentionOutput<B, D> {
        let normed_hidden_states = self.layer_norm.forward(hidden_states);
        let AttentionOutput { output, attentions } = self.cross_attention.forward(
            normed_hidden_states,
            Some(key_value_states),
            attention_mask,
            position_bias,
            layer_head_mask,
            past_key_value,
            use_cache,
            query_length,
            output_attentions,
            cache_position,
        );
        let layer_output = hidden_states + self.dropout.forward(output);
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
        let layer_norm = T5LayerNormConfig::new(self.d_model, self.layer_norm_epsilon).init(device);
        let dropout = DropoutConfig::new(self.dropout_rate).init();
        T5CrossAttention {
            cross_attention,
            layer_norm,
            dropout,
        }
    }
}

#[derive(Module, Debug)]
struct T5FeedForward<B: Backend> {
    dense: T5DenseGatedAct<B>,
    layer_norm: T5LayerNorm<B>,
    dropout: Dropout,
}

#[derive(Config, Debug)]
struct T5FeedForwardConfig {
    d_model: usize,
    d_ff: usize,
    layer_norm_epsilon: f64,
    dropout_rate: f64,
}

impl T5FeedForwardConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> T5FeedForward<B> {
        let dense =
            T5DenseGatedActConfig::new(self.d_model, self.d_ff, self.dropout_rate).init(device);
        let layer_norm = T5LayerNormConfig::new(self.d_model, self.layer_norm_epsilon).init(device);
        let dropout = DropoutConfig::new(self.dropout_rate).init();
        T5FeedForward {
            dense,
            layer_norm,
            dropout,
        }
    }
}

#[derive(Module, Debug)]
struct T5DenseGatedAct<B: Backend> {
    wi_0: Linear<B>,
    wi_1: Linear<B>,
    wo: Linear<B>,
    dropout: Dropout,
    activation: Gelu,
}

impl<B: Backend> T5DenseGatedAct<B> {
    fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let hidden_gelu = self.activation.forward(self.wi_0.forward(x));
        let hidden_linear = self.wi_1.forward(hidden_gelu.clone());
        let hidden_states = hidden_gelu * hidden_linear;
        let hidden_states = self.dropout.forward(hidden_states);
        let hidden_states = self.wo.forward(hidden_states);
        hidden_states
    }
}

#[derive(Config, Debug)]
struct T5DenseGatedActConfig {
    d_model: usize,
    d_ff: usize,
    dropout_rate: f64,
}

impl T5DenseGatedActConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> T5DenseGatedAct<B> {
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

#[derive(Module, Debug)]
struct T5Attention<B: Backend> {
    query: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
    dense: Linear<B>,
    relative_attention_bias: Option<Embedding<B>>,
}

impl<B: Backend> T5Attention<B> {
    fn forward<const D: usize>(
        &self,
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
    ) -> AttentionOutput<B, D> {
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

/// A layer normalization layer in the T5 style (no bias or subtraction of mean).
#[derive(Module, Debug)]
struct T5LayerNorm<B: Backend> {
    layer_norm: RmsNorm<B>,
}

impl<B: Backend> T5LayerNorm<B> {
    fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        self.layer_norm.forward(x)
    }
}

/// Configuration to create a [`T5LayerNorm`] layer.
#[derive(Config, Debug)]
struct T5LayerNormConfig {
    d_model: usize,
    layer_norm_epsilon: f64,
}

impl T5LayerNormConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> T5LayerNorm<B> {
        T5LayerNorm {
            // TODO: can we actually use `RmsNorm` here? The `transformers` implementation uses
            // `apex.normalization.FusedRMSNorm` which _might_ be the same thing?
            layer_norm: RmsNormConfig::new(self.d_model)
                .with_epsilon(self.layer_norm_epsilon)
                .init(device),
        }
    }
}
