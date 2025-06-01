use burn::{
    config::Config,
    nn::{Dropout, DropoutConfig, Embedding},
    prelude::*,
};

use crate::attention::EncoderOrDecoder;
use crate::block::{T5Block, T5BlockConfig};
use crate::layer_norm::{T5LayerNorm, T5LayerNormConfig};

#[derive(Module, Debug)]
pub struct T5Stack<B: Backend> {
    pub embed_tokens: Embedding<B>,
    pub blocks: Vec<T5Block<B>>,
    pub final_layer_norm: T5LayerNorm<B>,
    pub dropout: Dropout,
}

#[derive(Config, Debug)]
pub struct T5StackConfig {
    pub n_layers: usize,
    pub ty: EncoderOrDecoder,
    pub relative_attention_num_buckets: usize,
    pub relative_attention_max_distance: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub key_value_proj_dim: usize,
    pub num_heads: usize,
    pub layer_norm_epsilon: f64,
    pub dropout_rate: f64,
}

impl T5StackConfig {
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
