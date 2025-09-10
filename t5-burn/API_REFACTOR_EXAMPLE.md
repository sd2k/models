# T5 Attention API Refactor

This document demonstrates the refactoring of the T5 attention layer forward methods from using multiple positional parameters to using structured input parameters. This includes `T5Attention`, `T5SelfAttention`, and `T5CrossAttention`.

## Before: Multiple Positional Parameters

### Old API
```rust
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
        // Implementation
    }
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
        // Implementation
    }
}

impl<B: Backend> T5CrossAttention<B> {
    fn forward<const D: usize>(
        &self,
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
    ) -> AttentionOutput<B, D> {
        // Implementation
    }
}
```

### Usage Example (Old)
```rust
// T5Attention: Calling with many positional arguments - error-prone and hard to read
let output = attention.forward(
    hidden_states,
    Some(attention_mask),
    None, // key_value_states
    Some(position_bias),
    None, // past_key_value
    None, // layer_head_mask
    Some(128), // query_length
    true, // use_cache
    false, // output_attentions
    None, // cache_position
);

// T5SelfAttention: Still many parameters
let output = self_attention.forward(
    hidden_states,
    Some(attention_mask),
    Some(position_bias),
    None, // layer_head_mask
    None, // past_key_value
    true, // use_cache
    Some(128), // query_length
    false, // output_attentions
    None, // cache_position
);

// T5CrossAttention: Even more parameters
let output = cross_attention.forward(
    hidden_states,
    key_value_states,
    Some(attention_mask),
    Some(position_bias),
    None, // layer_head_mask
    None, // past_key_value
    true, // use_cache
    Some(128), // query_length
    false, // output_attentions
    None, // cache_position
);
```

### Problems with Old API
- **Too many parameters**: 9-10 parameters make the method signatures unwieldy
- **Parameter ordering**: Easy to mix up the order of optional parameters
- **Poor readability**: Hard to understand what each `None` or boolean represents
- **Maintenance burden**: Adding new parameters requires updating all call sites
- **Inconsistent signatures**: Different attention types have slightly different parameter sets

## After: Structured Input Parameters

### New API
```rust
// Input structs for each attention type
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

impl<B: Backend> T5Attention<B> {
    fn forward<const D: usize>(
        &self,
        input: T5AttentionInput<B, D>,
    ) -> AttentionOutput<B, D> {
        // Implementation
    }
}

impl<B: Backend> T5SelfAttention<B> {
    fn forward<const D: usize>(
        &self,
        input: T5SelfAttentionInput<B, D>,
    ) -> AttentionOutput<B, D> {
        // Implementation
    }
}

impl<B: Backend> T5CrossAttention<B> {
    fn forward<const D: usize>(
        &self,
        input: T5CrossAttentionInput<B, D>,
    ) -> AttentionOutput<B, D> {
        // Implementation
    }
}
```

### Usage Examples (New)

#### T5Attention - Direct Struct Initialization
```rust
// Clean, readable struct initialization with named fields
let input = T5AttentionInput {
    hidden_states,
    attention_mask: Some(attention_mask),
    key_value_states: None,
    position_bias: Some(position_bias),
    past_key_value: None,
    layer_head_mask: None,
    query_length: Some(128),
    use_cache: true,
    output_attentions: false,
    cache_position: None,
};

let output = attention.forward(input);
```

#### T5SelfAttention - Simplified for Self-Attention
```rust
// Self-attention doesn't need key_value_states
let input = T5SelfAttentionInput {
    hidden_states,
    attention_mask: Some(attention_mask),
    position_bias: Some(position_bias),
    layer_head_mask: None,
    past_key_value: None,
    use_cache: true,
    query_length: Some(128),
    output_attentions: false,
    cache_position: None,
};

let output = self_attention.forward(input);
```

#### T5CrossAttention - Includes Key-Value States
```rust
// Cross-attention requires key_value_states
let input = T5CrossAttentionInput {
    hidden_states,
    key_value_states,
    attention_mask: Some(attention_mask),
    position_bias: Some(position_bias),
    layer_head_mask: None,
    past_key_value: None,
    use_cache: true,
    query_length: Some(128),
    output_attentions: false,
    cache_position: None,
};

let output = cross_attention.forward(input);
```

#### Minimal Usage
```rust
// Simple case with mostly default/None values
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

let output = self_attention.forward(input);
```

## Benefits of the New API

### 1. **Improved Readability**
- Struct field names are self-documenting
- Clear intent when setting optional parameters
- No more guessing what `None` or `true` means in context

### 2. **Better Maintainability**
- Adding new parameters doesn't break existing code (with `..Default::default()` if needed)
- Clear field names make the purpose of each parameter obvious
- Simple struct initialization is easy to understand

### 3. **Type Safety**
- Impossible to pass parameters in wrong order
- Compile-time checks ensure all fields are provided
- Clear separation of concerns in the struct definition

### 4. **Simplicity**
- No complex builder pattern to learn
- Direct struct initialization is idiomatic Rust
- Easy to copy/paste and modify parameter sets

### 5. **Consistency**
- Same pattern applied across all attention types
- Each attention type has its own appropriate input struct
- Follows standard Rust idioms for parameter passing
- Aligns with internal code patterns

### 6. **Type-Specific Design**
- `T5SelfAttentionInput` omits `key_value_states` since it's not used
- `T5CrossAttentionInput` makes `key_value_states` required
- `T5AttentionInput` keeps `key_value_states` optional for flexibility
- Clear separation of concerns for different attention mechanisms

## Migration Path

Existing code can be migrated straightforwardly:

1. **Replace method calls** with direct struct initialization
2. **Named fields** make it clear what each parameter represents
3. **Copy existing parameter values** directly into struct fields

This refactor significantly improves the ergonomics of all T5 attention APIs while maintaining all existing functionality and keeping the code simple and idiomatic. Each attention type now has a tailored input struct that reflects its specific requirements, making the API both more type-safe and more intuitive to use.