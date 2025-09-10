# T5 Attention API Usage Examples

This document provides practical examples of how to use the refactored T5 attention APIs in real scenarios.

## Basic Usage Examples

### Simple Self-Attention

```rust
use burn::prelude::*;

// Simple self-attention with minimal parameters
fn simple_self_attention<B: Backend, const D: usize>(
    self_attention: &T5SelfAttention<B>,
    hidden_states: Tensor<B, D>,
) -> AttentionOutput<B, D> {
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
    
    self_attention.forward(input)
}
```

### Cross-Attention with Encoder-Decoder

```rust
// Cross-attention between decoder and encoder states
fn encoder_decoder_cross_attention<B: Backend, const D: usize>(
    cross_attention: &T5CrossAttention<B>,
    decoder_hidden_states: Tensor<B, D>,
    encoder_hidden_states: Tensor<B, D>,
    attention_mask: Option<Tensor<B, D>>,
) -> AttentionOutput<B, D> {
    let input = T5CrossAttentionInput {
        hidden_states: decoder_hidden_states,
        key_value_states: encoder_hidden_states,
        attention_mask,
        position_bias: None,
        layer_head_mask: None,
        past_key_value: None,
        use_cache: false,
        query_length: None,
        output_attentions: false,
        cache_position: None,
    };
    
    cross_attention.forward(input)
}
```

## Complete T5 Block Implementation

```rust
// Example of how a T5 block might use these APIs
struct T5Block<B: Backend> {
    self_attention: T5SelfAttention<B>,
    cross_attention: Option<T5CrossAttention<B>>, // Only for decoder
    feed_forward: T5FeedForward<B>,
}

impl<B: Backend> T5Block<B> {
    /// Forward pass through a T5 block
    fn forward<const D: usize>(
        &self,
        hidden_states: Tensor<B, D>,
        encoder_hidden_states: Option<Tensor<B, D>>, // For decoder blocks
        attention_mask: Option<Tensor<B, D>>,
        position_bias: Option<Tensor<B, D>>,
        use_cache: bool,
        output_attentions: bool,
    ) -> T5BlockOutput<B, D> {
        // Self-attention layer
        let self_attention_input = T5SelfAttentionInput {
            hidden_states: hidden_states.clone(),
            attention_mask: attention_mask.clone(),
            position_bias: position_bias.clone(),
            layer_head_mask: None,
            past_key_value: None,
            use_cache,
            query_length: None,
            output_attentions,
            cache_position: None,
        };
        
        let AttentionOutput { 
            output: hidden_states, 
            attentions: self_attentions 
        } = self.self_attention.forward(self_attention_input);
        
        // Cross-attention layer (if this is a decoder block)
        let (hidden_states, cross_attentions) = if let Some(cross_attention) = &self.cross_attention {
            if let Some(encoder_states) = encoder_hidden_states {
                let cross_attention_input = T5CrossAttentionInput {
                    hidden_states: hidden_states.clone(),
                    key_value_states: encoder_states,
                    attention_mask: attention_mask.clone(),
                    position_bias: None, // Cross-attention typically doesn't use position bias
                    layer_head_mask: None,
                    past_key_value: None,
                    use_cache,
                    query_length: None,
                    output_attentions,
                    cache_position: None,
                };
                
                let AttentionOutput { output, attentions } = cross_attention.forward(cross_attention_input);
                (output, attentions)
            } else {
                (hidden_states, None)
            }
        } else {
            (hidden_states, None)
        };
        
        // Feed-forward layer
        let hidden_states = self.feed_forward.forward(hidden_states);
        
        T5BlockOutput {
            hidden_states,
            self_attentions,
            cross_attentions,
        }
    }
}

struct T5BlockOutput<B: Backend, const D: usize> {
    hidden_states: Tensor<B, D>,
    self_attentions: Option<Tensor<B, D>>,
    cross_attentions: Option<Tensor<B, D>>,
}
```

## Advanced Usage with Caching

```rust
// Example showing how to use caching for inference
fn inference_with_cache<B: Backend, const D: usize>(
    self_attention: &T5SelfAttention<B>,
    hidden_states: Tensor<B, D>,
    past_key_value: Option<Tensor<B, D>>,
    cache_position: Option<bool>,
) -> AttentionOutput<B, D> {
    let input = T5SelfAttentionInput {
        hidden_states,
        attention_mask: None,
        position_bias: None,
        layer_head_mask: None,
        past_key_value,
        use_cache: true, // Enable caching for faster inference
        query_length: None,
        output_attentions: false, // Disable for inference
        cache_position,
    };
    
    self_attention.forward(input)
}
```

## Training vs Inference Configuration

```rust
// Helper functions to create common configurations

fn training_config<B: Backend, const D: usize>(
    hidden_states: Tensor<B, D>,
    attention_mask: Option<Tensor<B, D>>,
    position_bias: Option<Tensor<B, D>>,
) -> T5SelfAttentionInput<B, D> {
    T5SelfAttentionInput {
        hidden_states,
        attention_mask,
        position_bias,
        layer_head_mask: None,
        past_key_value: None,
        use_cache: false, // No caching during training
        query_length: None,
        output_attentions: true, // Useful for analysis during training
        cache_position: None,
    }
}

fn inference_config<B: Backend, const D: usize>(
    hidden_states: Tensor<B, D>,
    attention_mask: Option<Tensor<B, D>>,
    past_key_value: Option<Tensor<B, D>>,
) -> T5SelfAttentionInput<B, D> {
    T5SelfAttentionInput {
        hidden_states,
        attention_mask,
        position_bias: None, // Often not needed for inference
        layer_head_mask: None,
        past_key_value,
        use_cache: true, // Enable caching for faster inference
        query_length: None,
        output_attentions: false, // Disable for speed
        cache_position: None,
    }
}
```

## Benefits in Practice

### Before Refactor
```rust
// Hard to read and error-prone
let output = self_attention.forward(
    hidden_states,
    Some(attention_mask),
    Some(position_bias),
    None, // What does this None represent?
    None, // And this one?
    false, // use_cache?
    None, // query_length?
    true, // output_attentions?
    None, // cache_position?
);
```

### After Refactor
```rust
// Clear and self-documenting
let input = T5SelfAttentionInput {
    hidden_states,
    attention_mask: Some(attention_mask),
    position_bias: Some(position_bias),
    layer_head_mask: None, // Clearly not using layer head masking
    past_key_value: None,  // No cached key-values
    use_cache: false,      // Not caching for this forward pass
    query_length: None,    // Using full sequence length
    output_attentions: true, // Want attention weights for analysis
    cache_position: None,  // Not using positional caching
};

let output = self_attention.forward(input);
```

## Common Patterns

### Template for Self-Attention
```rust
// Create a template for common self-attention usage
fn default_self_attention_input<B: Backend, const D: usize>(
    hidden_states: Tensor<B, D>
) -> T5SelfAttentionInput<B, D> {
    T5SelfAttentionInput {
        hidden_states,
        attention_mask: None,
        position_bias: None,
        layer_head_mask: None,
        past_key_value: None,
        use_cache: false,
        query_length: None,
        output_attentions: false,
        cache_position: None,
    }
}

// Usage:
let mut input = default_self_attention_input(hidden_states);
input.attention_mask = Some(attention_mask);
input.use_cache = true;
let output = self_attention.forward(input);
```

This refactored API makes T5 attention code much more maintainable and easier to understand, especially in complex scenarios where multiple attention mechanisms are used together.