use burn::{
    config::Config,
    nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig, Linear, LinearConfig},
    prelude::*,
};

use crate::layer_norm::{T5LayerNorm, T5LayerNormConfig};

#[derive(Debug, Config, Copy, PartialEq)]
pub enum EncoderOrDecoder {
    Encoder,
    Decoder,
}

pub struct AttentionOutput<B: Backend, const D: usize> {
    pub output: Tensor<B, D>,
    pub attentions: Option<Tensor<B, D>>,
}

pub struct T5AttentionInput<B: Backend, const D: usize> {
    pub hidden_states: Tensor<B, D>,
    pub attention_mask: Option<Tensor<B, D>>,
    pub key_value_states: Option<Tensor<B, D>>,
    pub position_bias: Option<Tensor<B, D>>,
    pub past_key_value: Option<Tensor<B, D>>,
    pub layer_head_mask: Option<Tensor<B, D>>,
    pub query_length: Option<usize>,
    pub use_cache: bool,
    pub output_attentions: bool,
    pub cache_position: Option<bool>,
}

pub struct T5SelfAttentionInput<B: Backend, const D: usize> {
    pub hidden_states: Tensor<B, D>,
    pub attention_mask: Option<Tensor<B, D>>,
    pub position_bias: Option<Tensor<B, D>>,
    pub layer_head_mask: Option<Tensor<B, D>>,
    pub past_key_value: Option<Tensor<B, D>>,
    pub use_cache: bool,
    pub query_length: Option<usize>,
    pub output_attentions: bool,
    pub cache_position: Option<bool>,
}

pub struct T5CrossAttentionInput<B: Backend, const D: usize> {
    pub hidden_states: Tensor<B, D>,
    pub key_value_states: Tensor<B, D>,
    pub attention_mask: Option<Tensor<B, D>>,
    pub position_bias: Option<Tensor<B, D>>,
    pub layer_head_mask: Option<Tensor<B, D>>,
    pub past_key_value: Option<Tensor<B, D>>,
    pub use_cache: bool,
    pub query_length: Option<usize>,
    pub output_attentions: bool,
    pub cache_position: Option<bool>,
}

#[derive(Module, Debug)]
pub struct T5SelfAttention<B: Backend> {
    self_attention: T5Attention<B>,
    layer_norm: T5LayerNorm<B>,
    dropout: Dropout,
}

impl<B: Backend> T5SelfAttention<B> {
    pub fn forward<const D: usize>(
        &self,
        input: T5SelfAttentionInput<B, D>,
    ) -> AttentionOutput<B, D> {
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
pub struct T5SelfAttentionConfig {
    pub ty: EncoderOrDecoder,
    pub relative_attention_num_buckets: usize,
    pub relative_attention_max_distance: usize,
    pub d_model: usize,
    pub key_value_proj_dim: usize,
    pub num_heads: usize,
    pub layer_norm_epsilon: f64,
    pub dropout_rate: f64,
    pub layer_idx: usize,
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

#[derive(Module, Debug)]
pub struct T5CrossAttention<B: Backend> {
    cross_attention: T5Attention<B>,
    layer_norm: T5LayerNorm<B>,
    dropout: Dropout,
}

impl<B: Backend> T5CrossAttention<B> {
    pub fn forward<const D: usize>(
        &self,
        input: T5CrossAttentionInput<B, D>,
    ) -> AttentionOutput<B, D> {
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
pub struct T5CrossAttentionConfig {
    pub ty: EncoderOrDecoder,
    pub relative_attention_num_buckets: usize,
    pub relative_attention_max_distance: usize,
    pub d_model: usize,
    pub key_value_proj_dim: usize,
    pub num_heads: usize,
    pub layer_norm_epsilon: f64,
    pub dropout_rate: f64,
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

#[derive(Module, Debug)]
pub struct T5Attention<B: Backend> {
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
    pub fn forward<const D: usize>(&self, _input: T5AttentionInput<B, D>) -> AttentionOutput<B, D> {
        todo!()
    }
}

#[derive(Config, Debug)]
pub struct T5AttentionConfig {
    pub ty: EncoderOrDecoder,
    #[config(default = false)]
    pub has_relative_attention_bias: bool,
    pub relative_attention_num_buckets: usize,
    pub relative_attention_max_distance: usize,
    pub d_model: usize,
    pub key_value_proj_dim: usize,
    pub num_heads: usize,
    pub dropout_rate: f64,
    pub layer_idx: usize,
}

impl T5AttentionConfig {
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

        T5Attention {
            query,
            key,
            value,
            dense,
            relative_attention_bias,
        }
    }
}
