use burn::{config::Config, prelude::*};

use crate::attention::{
    EncoderOrDecoder, T5CrossAttention, T5CrossAttentionConfig, T5SelfAttention,
    T5SelfAttentionConfig,
};
use crate::feed_forward::{T5FeedForward, T5FeedForwardConfig};

#[derive(Module, Debug)]
pub struct T5Block<B: Backend> {
    pub self_attention: T5SelfAttention<B>,
    pub cross_attention: Option<T5CrossAttention<B>>,
    pub ff: T5FeedForward<B>,
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
        let cross_attention = if self.ty == EncoderOrDecoder::Decoder {
            Some(
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
