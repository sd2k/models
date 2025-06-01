use burn::{
    config::Config,
    nn::{Embedding, EmbeddingConfig},
    prelude::*,
};

use crate::attention::EncoderOrDecoder;
use crate::stack::{T5Stack, T5StackConfig};

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

impl T5ModelConfig {
    /// Create a new `T5Model` with this configuration.
    pub fn init<B: Backend>(&self, device: &B::Device) -> T5Model<B> {
        let shared = EmbeddingConfig::new(self.vocab_size, self.d_model).init(device);
        T5Model {
            shared: shared.clone(),
            encoder: T5StackConfig {
                n_layers: self.n_layers,
                ty: EncoderOrDecoder::Encoder,
                relative_attention_num_buckets: self.relative_attention_num_buckets,
                relative_attention_max_distance: self.relative_attention_max_distance,
                d_model: self.d_model,
                d_ff: self.d_ff,
                key_value_proj_dim: self.d_kv,
                num_heads: self.n_heads,
                layer_norm_epsilon: self.layer_norm_eps,
                dropout_rate: self.dropout_rate,
            }
            .init(device, shared.clone()),
            decoder: T5StackConfig {
                n_layers: self.n_decoder_layers.unwrap_or(self.n_layers),
                ty: EncoderOrDecoder::Decoder,
                relative_attention_num_buckets: self.relative_attention_num_buckets,
                relative_attention_max_distance: self.relative_attention_max_distance,
                d_model: self.d_model,
                d_ff: self.d_ff,
                key_value_proj_dim: self.d_kv,
                num_heads: self.n_heads,
                layer_norm_epsilon: self.layer_norm_eps,
                dropout_rate: self.dropout_rate,
            }
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
