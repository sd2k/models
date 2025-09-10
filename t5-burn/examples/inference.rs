//! Example demonstrating T5 model inference
//!
//! This example shows how to:
//! 1. Load a pretrained T5 model
//! 2. Perform text-to-text generation
//! 3. Use the model for various tasks

use burn::prelude::*;
use burn_ndarray::NdArray;
use t5_burn::{
    load_pretrained_t5_for_conditional_generation, T5ForConditionalGeneration,
    T5ForConditionalGenerationInput, T5ModelInput, T5Variant,
};

type TestBackend = NdArray<f32>;

fn main() {
    let device = Default::default();

    // Load a pretrained T5 model
    println!("Loading T5-small model...");
    let model = load_pretrained_t5_for_conditional_generation::<TestBackend>(
        T5Variant::Small,
        &device,
    )
    .expect("Failed to load model");

    println!("Model loaded successfully!");

    // Example 1: Simple forward pass
    example_forward_pass(&model, &device);

    // Example 2: Text generation (simplified - in practice you'd use a tokenizer)
    example_generation(&model, &device);

    // Example 3: Computing loss for training
    example_training_loss(&model, &device);
}

fn example_forward_pass<B: Backend>(model: &T5ForConditionalGeneration<B>, device: &B::Device) {
    println!("\n=== Example 1: Forward Pass ===");

    let batch_size = 2;
    let encoder_seq_len = 10;
    let decoder_seq_len = 8;
    let vocab_size = 32128;

    // Create dummy input IDs (in practice, these would come from a tokenizer)
    let encoder_input_ids = Tensor::<B, 2, Int>::zeros([batch_size, encoder_seq_len], device);
    let decoder_input_ids = Tensor::<B, 2, Int>::zeros([batch_size, decoder_seq_len], device);

    // Create attention masks (1 for real tokens, 0 for padding)
    let encoder_attention_mask = Tensor::<B, 2>::ones([batch_size, encoder_seq_len], device);
    let decoder_attention_mask = Tensor::<B, 2>::ones([batch_size, decoder_seq_len], device);

    // Forward pass
    let model_input = T5ModelInput {
        input_ids: Some(encoder_input_ids),
        attention_mask: Some(encoder_attention_mask),
        decoder_input_ids: Some(decoder_input_ids),
        decoder_attention_mask: Some(decoder_attention_mask),
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
    let input = T5ForConditionalGenerationInput {
        model_input,
        labels: None,
    };

    let output = model.forward(input);

    // Check output shape
    let logits_shape = output.logits.shape();
    println!(
        "Output logits shape: [{}, {}, {}]",
        logits_shape.dims[0], logits_shape.dims[1], logits_shape.dims[2]
    );
    println!("Expected: [{}, {}, {}]", batch_size, decoder_seq_len, vocab_size);
}

fn example_generation<B: Backend>(model: &T5ForConditionalGeneration<B>, device: &B::Device) {
    println!("\n=== Example 2: Text Generation ===");

    let batch_size = 1;
    let encoder_seq_len = 5;
    let max_length = 20;

    // Create encoder input (in practice, this would be tokenized text)
    let encoder_input_ids = Tensor::<B, 2, Int>::zeros([batch_size, encoder_seq_len], device);
    let encoder_attention_mask = Tensor::<B, 2>::ones([batch_size, encoder_seq_len], device);

    println!("Generating text (greedy decoding)...");

    // Generate text using the model's generate method
    let generated_ids = model.generate(
        encoder_input_ids,
        max_length,
        device,
    );

    let generated_shape = generated_ids.shape();
    println!(
        "Generated token IDs shape: [{}, {}]",
        generated_shape.dims[0], generated_shape.dims[1]
    );

    // In practice, you would decode these IDs back to text using a tokenizer
    println!("Note: To see actual text, integrate with a T5 tokenizer");
}

fn example_training_loss<B: Backend>(model: &T5ForConditionalGeneration<B>, device: &B::Device) {
    println!("\n=== Example 3: Training Loss Computation ===");

    let batch_size = 2;
    let encoder_seq_len = 10;
    let decoder_seq_len = 8;

    // Create input data
    let encoder_input_ids = Tensor::<B, 2, Int>::zeros([batch_size, encoder_seq_len], device);
    let decoder_input_ids = Tensor::<B, 2, Int>::zeros([batch_size, decoder_seq_len], device);
    
    // Create labels (shifted decoder_input_ids in practice)
    let labels = Tensor::<B, 2, Int>::zeros([batch_size, decoder_seq_len], device);

    // Create attention masks
    let encoder_attention_mask = Tensor::<B, 2>::ones([batch_size, encoder_seq_len], device);
    let decoder_attention_mask = Tensor::<B, 2>::ones([batch_size, decoder_seq_len], device);

    // Forward pass with labels to compute loss
    let model_input = T5ModelInput {
        input_ids: Some(encoder_input_ids),
        attention_mask: Some(encoder_attention_mask),
        decoder_input_ids: Some(decoder_input_ids),
        decoder_attention_mask: Some(decoder_attention_mask),
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
    let input = T5ForConditionalGenerationInput {
        model_input,
        labels: Some(labels),
    };

    let output = model.forward(input);

    if let Some(loss) = output.loss {
        let loss_tensor = loss.clone().mean();
        let loss_value = loss_tensor.to_data().to_vec::<f32>().unwrap()[0];
        println!("Training loss: {:.4}", loss_value);
    } else {
        println!("No loss computed (labels not provided)");
    }
}

// Additional examples could include:
// - Beam search generation
// - Using different T5 variants (Base, Large)
// - Loading custom weights from disk
// - Fine-tuning on specific tasks
// - Using T5 for translation, summarization, etc.