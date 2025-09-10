use burn::{
    config::Config,
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig},
    prelude::*,
};

use crate::attention::EncoderOrDecoder;
use crate::stack::{IdsOrEmbeds, T5Stack, T5StackConfig, T5StackInput, T5StackOutput};

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

/// Input structure for T5Model forward pass
#[derive(Debug, Clone)]
pub struct T5ModelInput<B: Backend> {
    /// Input token IDs for encoder [batch_size, encoder_sequence_length]
    pub input_ids: Option<Tensor<B, 2, burn::tensor::Int>>,
    /// Attention mask for encoder input [batch_size, encoder_sequence_length]
    pub attention_mask: Option<Tensor<B, 2>>,
    /// Input token IDs for decoder [batch_size, decoder_sequence_length]
    pub decoder_input_ids: Option<Tensor<B, 2, burn::tensor::Int>>,
    /// Attention mask for decoder input [batch_size, decoder_sequence_length]
    pub decoder_attention_mask: Option<Tensor<B, 2>>,
    /// Head masks for encoder layers [num_layers, num_heads]
    pub head_mask: Option<Tensor<B, 2>>,
    /// Head masks for decoder layers [num_layers, num_heads]
    pub decoder_head_mask: Option<Tensor<B, 2>>,
    /// Cross-attention head masks [num_layers, num_heads]
    pub cross_attn_head_mask: Option<Tensor<B, 2>>,
    /// Pre-computed encoder outputs
    pub encoder_outputs: Option<T5StackOutput<B>>,
    /// Past key-value states for caching
    pub past_key_values: Option<Vec<Option<Tensor<B, 4>>>>,
    /// Input embeddings for encoder (alternative to input_ids)
    pub inputs_embeds: Option<Tensor<B, 3>>,
    /// Input embeddings for decoder (alternative to decoder_input_ids)
    pub decoder_inputs_embeds: Option<Tensor<B, 3>>,
    /// Whether to use caching
    pub use_cache: bool,
    /// Whether to output attention weights
    pub output_attentions: bool,
    /// Whether to output hidden states from all layers
    pub output_hidden_states: bool,
    /// Cache position for incremental generation
    pub cache_position: Option<Tensor<B, 1, burn::tensor::Int>>,
}

impl<B: Backend> T5ModelInput<B> {
    pub fn new(_device: &B::Device) -> Self {
        Self {
            input_ids: None,
            attention_mask: None,
            decoder_input_ids: None,
            decoder_attention_mask: None,
            head_mask: None,
            decoder_head_mask: None,
            cross_attn_head_mask: None,
            encoder_outputs: None,
            past_key_values: None,
            inputs_embeds: None,
            decoder_inputs_embeds: None,
            use_cache: false,
            output_attentions: false,
            output_hidden_states: false,
            cache_position: None,
        }
    }
}

/// Output structure for T5Model
#[derive(Debug, Clone)]
pub struct T5ModelOutput<B: Backend> {
    /// Final decoder hidden states [batch_size, decoder_sequence_length, d_model]
    pub last_hidden_state: Tensor<B, 3>,
    /// Past key-value states for caching
    pub past_key_values: Option<Vec<Option<Tensor<B, 4>>>>,
    /// Decoder hidden states from all layers (if output_hidden_states=True)
    pub decoder_hidden_states: Option<Vec<Tensor<B, 3>>>,
    /// Decoder self-attention weights from all layers (if output_attentions=True)
    pub decoder_attentions: Option<Vec<Option<Tensor<B, 4>>>>,
    /// Cross-attention weights from all layers (if output_attentions=True)
    pub cross_attentions: Option<Vec<Option<Tensor<B, 4>>>>,
    /// Encoder last hidden state [batch_size, encoder_sequence_length, d_model]
    pub encoder_last_hidden_state: Option<Tensor<B, 3>>,
    /// Encoder hidden states from all layers (if output_hidden_states=True)
    pub encoder_hidden_states: Option<Vec<Tensor<B, 3>>>,
    /// Encoder self-attention weights from all layers (if output_attentions=True)
    pub encoder_attentions: Option<Vec<Option<Tensor<B, 4>>>>,
}

impl<B: Backend> T5Model<B> {
    /// Forward pass through the T5 model
    /// 
    /// This method processes input through both the encoder and decoder stacks.
    /// Following the T5 architecture:
    /// 1. Encode input sequence with the encoder stack
    /// 2. Decode target sequence with the decoder stack, attending to encoder outputs
    ///
    /// # Arguments
    /// * `input` - T5ModelInput containing encoder and decoder inputs
    ///
    /// # Returns
    /// T5ModelOutput containing decoder outputs and optionally encoder outputs
    pub fn forward(&self, input: T5ModelInput<B>) -> T5ModelOutput<B> {
        let _device = self.shared.devices()[0].clone();
        
        // Process encoder unless encoder_outputs are provided  
        let encoder_outputs_provided = input.encoder_outputs.is_some();
        let encoder_outputs = if let Some(encoder_outputs) = input.encoder_outputs {
            encoder_outputs
        } else {
            // Determine encoder input
            let encoder_inputs = if let Some(input_ids) = input.input_ids {
                IdsOrEmbeds::Ids(input_ids)
            } else if let Some(inputs_embeds) = input.inputs_embeds {
                IdsOrEmbeds::Embeds(inputs_embeds)
            } else {
                panic!("Must provide either input_ids or inputs_embeds for encoder");
            };
            
            let encoder_input = T5StackInput {
                inputs: encoder_inputs,
                attention_mask: input.attention_mask.clone(),
                encoder_hidden_states: None, // Encoder doesn't use cross-attention
                encoder_attention_mask: None,
                head_mask: input.head_mask,
                cross_attn_head_mask: None,
                past_key_values: None, // Encoder typically doesn't use caching
                use_cache: false,
                output_attentions: input.output_attentions,
                output_hidden_states: input.output_hidden_states,
                cache_position: input.cache_position.clone(),
            };
            
            self.encoder.forward(encoder_input)
        };

        // Process decoder
        let decoder_inputs = if let Some(decoder_input_ids) = input.decoder_input_ids {
            IdsOrEmbeds::Ids(decoder_input_ids)
        } else if let Some(decoder_inputs_embeds) = input.decoder_inputs_embeds {
            IdsOrEmbeds::Embeds(decoder_inputs_embeds)
        } else {
            panic!("Must provide either decoder_input_ids or decoder_inputs_embeds for decoder");
        };
        
        let decoder_input = T5StackInput {
            inputs: decoder_inputs,
            attention_mask: input.decoder_attention_mask,
            encoder_hidden_states: Some(encoder_outputs.last_hidden_state.clone()),
            encoder_attention_mask: input.attention_mask, // Use encoder attention mask for cross-attention
            head_mask: input.decoder_head_mask,
            cross_attn_head_mask: input.cross_attn_head_mask,
            past_key_values: input.past_key_values,
            use_cache: input.use_cache,
            output_attentions: input.output_attentions,
            output_hidden_states: input.output_hidden_states,
            cache_position: input.cache_position,
        };
        
        let decoder_outputs = self.decoder.forward(decoder_input);

        T5ModelOutput {
            last_hidden_state: decoder_outputs.last_hidden_state,
            past_key_values: decoder_outputs.past_key_values,
            decoder_hidden_states: decoder_outputs.hidden_states,
            decoder_attentions: decoder_outputs.attentions,
            cross_attentions: decoder_outputs.cross_attentions,
            encoder_last_hidden_state: if !encoder_outputs_provided {
                Some(encoder_outputs.last_hidden_state)
            } else {
                None
            },
            encoder_hidden_states: if !encoder_outputs_provided && input.output_hidden_states {
                encoder_outputs.hidden_states
            } else {
                None
            },
            encoder_attentions: if !encoder_outputs_provided && input.output_attentions {
                encoder_outputs.attentions
            } else {
                None
            },
        }
    }
}

/// Configuration for T5ForConditionalGeneration
#[derive(Config, Debug)]
pub struct T5ForConditionalGenerationConfig {
    /// Base T5 model configuration
    pub model: T5ModelConfig,
}

impl T5ForConditionalGenerationConfig {
    /// Create a new T5ForConditionalGeneration with this configuration
    pub fn init<B: Backend>(&self, device: &B::Device) -> T5ForConditionalGeneration<B> {
        let model = self.model.init(device);
        let lm_head = LinearConfig::new(self.model.d_model, self.model.vocab_size)
            .with_bias(false)
            .init(device);
        
        T5ForConditionalGeneration {
            model,
            lm_head,
        }
    }
}

/// T5 Model with a language modeling head on top for text generation
/// 
/// This model extends the base T5Model with a linear layer that projects
/// the decoder hidden states to vocabulary logits for text generation.
#[derive(Module, Debug)]
pub struct T5ForConditionalGeneration<B: Backend> {
    /// Base T5 model (encoder-decoder)
    model: T5Model<B>,
    /// Language modeling head (linear projection to vocabulary)
    lm_head: Linear<B>,
}

/// Input structure for T5ForConditionalGeneration
#[derive(Debug, Clone)]
pub struct T5ForConditionalGenerationInput<B: Backend> {
    /// Base model inputs
    pub model_input: T5ModelInput<B>,
    /// Target labels for loss computation [batch_size, sequence_length]
    pub labels: Option<Tensor<B, 2, burn::tensor::Int>>,
}

impl<B: Backend> T5ForConditionalGenerationInput<B> {
    pub fn new(model_input: T5ModelInput<B>) -> Self {
        Self {
            model_input,
            labels: None,
        }
    }
}

/// Output structure for T5ForConditionalGeneration
#[derive(Debug, Clone)]
pub struct T5ForConditionalGenerationOutput<B: Backend> {
    /// Language modeling logits [batch_size, sequence_length, vocab_size]
    pub logits: Tensor<B, 3>,
    /// Loss value (if labels provided) []
    pub loss: Option<Tensor<B, 1>>,
    /// Past key-value states for caching
    pub past_key_values: Option<Vec<Option<Tensor<B, 4>>>>,
    /// Decoder hidden states from all layers (if output_hidden_states=True)
    pub decoder_hidden_states: Option<Vec<Tensor<B, 3>>>,
    /// Decoder self-attention weights from all layers (if output_attentions=True)
    pub decoder_attentions: Option<Vec<Option<Tensor<B, 4>>>>,
    /// Cross-attention weights from all layers (if output_attentions=True)
    pub cross_attentions: Option<Vec<Option<Tensor<B, 4>>>>,
    /// Encoder last hidden state [batch_size, encoder_sequence_length, d_model]
    pub encoder_last_hidden_state: Option<Tensor<B, 3>>,
    /// Encoder hidden states from all layers (if output_hidden_states=True)
    pub encoder_hidden_states: Option<Vec<Tensor<B, 3>>>,
    /// Encoder self-attention weights from all layers (if output_attentions=True)
    pub encoder_attentions: Option<Vec<Option<Tensor<B, 4>>>>,
}

impl<B: Backend> T5ForConditionalGeneration<B> {
    /// Forward pass through T5 model with language modeling head
    /// 
    /// Processes input through the base T5 model and applies the language modeling
    /// head to produce vocabulary logits. Optionally computes cross-entropy loss
    /// if labels are provided.
    ///
    /// # Arguments
    /// * `input` - T5ForConditionalGenerationInput containing model inputs and optional labels
    ///
    /// # Returns
    /// T5ForConditionalGenerationOutput containing logits and optionally loss
    pub fn forward(&self, input: T5ForConditionalGenerationInput<B>) -> T5ForConditionalGenerationOutput<B> {
        // Forward pass through base model
        let model_output = self.model.forward(input.model_input);
        
        // Apply language modeling head to get vocabulary logits
        let logits = self.lm_head.forward(model_output.last_hidden_state.clone());
        
        // Compute loss if labels are provided
        let loss = if let Some(labels) = input.labels {
            // Get dimensions before moving tensors
            let logits_dims = logits.dims();
            let labels_dims = labels.dims();
            
            // Shift logits and labels for autoregressive loss
            // logits: [batch, seq_len-1, vocab_size], labels: [batch, seq_len-1]
            let shift_logits = logits.clone().slice([0..logits_dims[0], 0..logits_dims[1]-1, 0..logits_dims[2]]);
            let shift_labels = labels.slice([0..labels_dims[0], 1..labels_dims[1]]);
            
            // Get dimensions after slicing
            let shift_logits_dims = shift_logits.dims();
            let shift_labels_dims = shift_labels.dims();
            
            // Reshape for cross entropy: logits [batch*seq_len, vocab_size], labels [batch*seq_len, 1]
            let flat_logits = shift_logits.reshape([shift_logits_dims[0] * shift_logits_dims[1], shift_logits_dims[2]]);
            let flat_labels = shift_labels.reshape([shift_labels_dims[0] * shift_labels_dims[1], 1]).float();
            
            // Compute cross-entropy loss
            let loss = burn::tensor::loss::cross_entropy_with_logits(flat_logits, flat_labels);
            Some(loss)
        } else {
            None
        };
        
        T5ForConditionalGenerationOutput {
            logits,
            loss,
            past_key_values: model_output.past_key_values,
            decoder_hidden_states: model_output.decoder_hidden_states,
            decoder_attentions: model_output.decoder_attentions,
            cross_attentions: model_output.cross_attentions,
            encoder_last_hidden_state: model_output.encoder_last_hidden_state,
            encoder_hidden_states: model_output.encoder_hidden_states,
            encoder_attentions: model_output.encoder_attentions,
        }
    }
    
    /// Generate text using the model
    /// 
    /// Simple greedy generation implementation. For more advanced generation
    /// methods (beam search, sampling, etc.), additional methods would be needed.
    ///
    /// # Arguments
    /// * `input_ids` - Input token IDs [batch_size, input_length]
    /// * `max_length` - Maximum length of generated sequence
    /// * `device` - Device for tensor operations
    ///
    /// # Returns
    /// Generated token IDs [batch_size, generated_length]
    pub fn generate(
        &self,
        input_ids: Tensor<B, 2, burn::tensor::Int>,
        max_length: usize,
        device: &B::Device,
    ) -> Tensor<B, 2, burn::tensor::Int> {
        let [batch_size, _input_length] = input_ids.dims();
        
        // Start with a simple decoder input (could be improved with proper start tokens)
        let mut decoder_input_ids: Tensor<B, 2, burn::tensor::Int> = Tensor::zeros([batch_size, 1], device);
        
        for _ in 0..max_length {
            let model_input = T5ModelInput {
                input_ids: Some(input_ids.clone()),
                decoder_input_ids: Some(decoder_input_ids.clone()),
                attention_mask: None,
                decoder_attention_mask: None,
                head_mask: None,
                decoder_head_mask: None,
                cross_attn_head_mask: None,
                encoder_outputs: None,
                past_key_values: None,
                inputs_embeds: None,
                decoder_inputs_embeds: None,
                use_cache: false,
                output_attentions: false,
                output_hidden_states: false,
                cache_position: None,
            };
            
            let generation_input = T5ForConditionalGenerationInput::new(model_input);
            let output = self.forward(generation_input);
            
            // Get next token (greedy: take argmax of last position)
            let [_, seq_len, vocab_size] = output.logits.dims();
            let last_logits = output.logits.slice([0..batch_size, seq_len-1..seq_len, 0..vocab_size]);
            let next_token = last_logits.squeeze::<2>(1).argmax(1).unsqueeze::<2>();
            
            // Concatenate next token to decoder input
            decoder_input_ids = Tensor::cat(vec![decoder_input_ids, next_token], 1);
            
            // Simple stopping criterion (this could be improved)
            if decoder_input_ids.dims()[1] >= max_length {
                break;
            }
        }
        
        decoder_input_ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_t5_model_forward() {
        let device = Default::default();
        
        // Create a small T5 model configuration matching working test pattern
        let config = T5ModelConfig {
            vocab_size: 100,
            d_model: 64,
            d_kv: 64,   // Use d_model as key_value_proj_dim like working tests
            d_ff: 128,
            n_layers: 2,
            n_decoder_layers: None,
            n_heads: 8,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            dropout_rate: 0.0,
            classifier_dropout: 0.0,
            layer_norm_eps: 1e-6,
            #[cfg(debug_assertions)]
            initializer_factor: 1.0,
            feed_forward_proj: FeedForwardLayer::ReLU,
            use_cache: true,
        };
        
        // Initialize the model
        let model = config.init::<TestBackend>(&device);
        
        // Create test inputs
        let batch_size = 2;
        let encoder_seq_len = 10;
        let decoder_seq_len = 8;
        
        let input_ids = Tensor::from_ints(
            [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]],
            &device,
        );
        
        let decoder_input_ids = Tensor::from_ints(
            [[1, 2, 3, 4, 5, 6, 7, 8],
             [9, 10, 11, 12, 13, 14, 15, 16]],
            &device,
        );
        
        let model_input = T5ModelInput {
            input_ids: Some(input_ids),
            attention_mask: None,
            decoder_input_ids: Some(decoder_input_ids),
            decoder_attention_mask: None,
            head_mask: None,
            decoder_head_mask: None,
            cross_attn_head_mask: None,
            encoder_outputs: None,
            past_key_values: None,
            inputs_embeds: None,
            decoder_inputs_embeds: None,
            use_cache: false,
            output_attentions: false,
            output_hidden_states: false,
            cache_position: None,
        };
        
        // Forward pass
        let output = model.forward(model_input);
        
        // Verify output shape
        assert_eq!(
            output.last_hidden_state.dims(),
            [batch_size, decoder_seq_len, 64]
        );
        
        // Check that encoder outputs are included when not pre-provided
        assert!(output.encoder_last_hidden_state.is_some());
        let encoder_output = output.encoder_last_hidden_state.unwrap();
        assert_eq!(
            encoder_output.dims(),
            [batch_size, encoder_seq_len, 64]
        );
    }
    
    #[test]
    fn test_t5_for_conditional_generation_forward() {
        let device = Default::default();
        
        // Create a small T5 model configuration matching working test pattern
        let model_config = T5ModelConfig {
            vocab_size: 100,
            d_model: 64,
            d_kv: 64,  // Use d_model as key_value_proj_dim like working tests
            d_ff: 128,
            n_layers: 1,  // Smaller for faster test
            n_decoder_layers: None,
            n_heads: 8,   // Use 8 heads like working tests
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            dropout_rate: 0.0,
            classifier_dropout: 0.0,
            layer_norm_eps: 1e-6,
            #[cfg(debug_assertions)]
            initializer_factor: 1.0,
            feed_forward_proj: FeedForwardLayer::ReLU,
            use_cache: true,
        };
        
        let config = T5ForConditionalGenerationConfig {
            model: model_config,
        };
        
        // Initialize the model
        let model = config.init::<TestBackend>(&device);
        
        // Create test inputs
        let batch_size = 2;
        let encoder_seq_len = 6;
        let decoder_seq_len = 4;
        
        let input_ids = Tensor::from_ints(
            [[1, 2, 3, 4, 5, 6],
             [7, 8, 9, 10, 11, 12]],
            &device,
        );
        
        let decoder_input_ids = Tensor::from_ints(
            [[1, 2, 3, 4],
             [5, 6, 7, 8]],
            &device,
        );
        
        let model_input = T5ModelInput {
            input_ids: Some(input_ids),
            attention_mask: None,
            decoder_input_ids: Some(decoder_input_ids),
            decoder_attention_mask: None,
            head_mask: None,
            decoder_head_mask: None,
            cross_attn_head_mask: None,
            encoder_outputs: None,
            past_key_values: None,
            inputs_embeds: None,
            decoder_inputs_embeds: None,
            use_cache: false,
            output_attentions: false,
            output_hidden_states: false,
            cache_position: None,
        };
        
        let generation_input = T5ForConditionalGenerationInput::new(model_input);
        
        // Forward pass
        let output = model.forward(generation_input);
        
        // Verify output shape - logits should be [batch_size, decoder_seq_len, vocab_size]
        assert_eq!(
            output.logits.dims(),
            [batch_size, decoder_seq_len, 100]  // vocab_size = 100
        );
        
        // Loss should be None since no labels were provided
        assert!(output.loss.is_none());
        
        // Check that encoder outputs are included
        assert!(output.encoder_last_hidden_state.is_some());
        let encoder_output = output.encoder_last_hidden_state.unwrap();
        assert_eq!(
            encoder_output.dims(),
            [batch_size, encoder_seq_len, 64]
        );
        
        println!("✅ T5ForConditionalGeneration forward pass test passed!");
    }
    
    #[test]
    fn test_t5_for_conditional_generation_with_loss() {
        let device = Default::default();
        
        // Create a very small model for fast testing, matching working test pattern
        let model_config = T5ModelConfig {
            vocab_size: 50,
            d_model: 32,
            d_kv: 32,  // Use d_model as key_value_proj_dim like working tests
            d_ff: 64,
            n_layers: 1,
            n_decoder_layers: None,
            n_heads: 8,
            relative_attention_num_buckets: 16,
            relative_attention_max_distance: 64,
            dropout_rate: 0.0,
            classifier_dropout: 0.0,
            layer_norm_eps: 1e-6,
            #[cfg(debug_assertions)]
            initializer_factor: 1.0,
            feed_forward_proj: FeedForwardLayer::ReLU,
            use_cache: false,
        };
        
        let config = T5ForConditionalGenerationConfig {
            model: model_config,
        };
        
        let model = config.init::<TestBackend>(&device);
        
        // Create test inputs
        let batch_size = 2;
        let seq_len = 4;
        
        let input_ids = Tensor::from_ints(
            [[1, 2, 3, 4],
             [5, 6, 7, 8]],
            &device,
        );
        
        let decoder_input_ids = Tensor::from_ints(
            [[1, 2, 3, 4],
             [5, 6, 7, 8]],
            &device,
        );
        
        let labels = Tensor::from_ints(
            [[2, 3, 4, 1],  // Shifted version for autoregressive loss
             [6, 7, 8, 1]],
            &device,
        );
        
        let model_input = T5ModelInput {
            input_ids: Some(input_ids),
            attention_mask: None,
            decoder_input_ids: Some(decoder_input_ids),
            decoder_attention_mask: None,
            head_mask: None,
            decoder_head_mask: None,
            cross_attn_head_mask: None,
            encoder_outputs: None,
            past_key_values: None,
            inputs_embeds: None,
            decoder_inputs_embeds: None,
            use_cache: false,
            output_attentions: false,
            output_hidden_states: false,
            cache_position: None,
        };
        
        let generation_input = T5ForConditionalGenerationInput {
            model_input,
            labels: Some(labels),
        };
        
        // Forward pass
        let output = model.forward(generation_input);
        
        // Verify logits shape
        assert_eq!(
            output.logits.dims(),
            [batch_size, seq_len, 50]
        );
        
        // Loss should be computed since labels were provided
        assert!(output.loss.is_some());
        let loss = output.loss.unwrap();
        assert_eq!(loss.dims(), [1]);  // Loss dimension
        
        // Verify loss is finite and positive (cross-entropy should be > 0)
        let loss_value = loss.into_scalar();
        assert!(loss_value.is_finite());
        assert!(loss_value > 0.0);
        
        println!("✅ T5ForConditionalGeneration with loss computation test passed! Loss: {:.4}", loss_value);
    }
}
