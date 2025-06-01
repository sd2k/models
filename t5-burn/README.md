# T5-Burn: T5 Transformer Architecture in Burn

This is a Rust implementation of the T5 (Text-to-Text Transfer Transformer) architecture using the Burn deep learning framework.

## Overview

T5 (Text-to-Text Transfer Transformer) is a transformer model introduced by Google Research that frames all NLP tasks in a unified text-to-text format. T5 achieves state-of-the-art results on many NLP benchmarks and is particularly effective for tasks like:

- Translation
- Summarization
- Question answering
- Text classification

This implementation in Burn allows Rust developers to use T5 models in their applications with high performance, portability, and type safety.

## Features

- Full T5 encoder-decoder architecture
- Relative position biases in attention mechanism
- T5-specific layer normalization
- Gated feed-forward networks
- Support for multiple backends (CUDA, CPU)
- Easy integration with Burn's training and inference pipeline

## Usage

```rust
use burn::tensor::{backend::Backend, Int, Tensor};
use t5_burn::{T5Config, T5ForConditionalGeneration};

fn main() {
    // Initialize the device (CPU, CUDA, etc.)
    let device = Default::default();
    
    // Create a T5 configuration
    let config = T5Config::t5_small(32000); // 32000 is vocab size
    
    // Initialize the model
    let model = T5ForConditionalGeneration::new(config, &device);
    
    // Prepare input tensors
    let input_ids = Tensor::<Backend, 2, Int>::ones([1, 10], &device);
    
    // Forward pass
    let logits = model.forward(input_ids, None, None, None);
    
    // Process logits for generation
    // ...
}
```

## Model Sizes

The implementation includes configuration helpers for all standard T5 model sizes:

- T5-Small (60M parameters)
- T5-Base (220M parameters)
- T5-Large (770M parameters)
- T5-3B (3B parameters)
- T5-11B (11B parameters)

## References

- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)
- [The Burn ML Framework](https://github.com/tracel-ai/burn)
- [HuggingFace Transformers T5 Implementation](https://github.com/huggingface/transformers/tree/main/src/transformers/models/t5)

## License

MIT License