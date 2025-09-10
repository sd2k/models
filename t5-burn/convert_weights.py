#!/usr/bin/env python3
"""
Script to convert HuggingFace T5 weights to Burn format.

This script downloads T5 models from HuggingFace and converts them to 
SafeTensors format that can be loaded by the Burn framework.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import torch
from transformers import T5Model, T5ForConditionalGeneration, T5Config
from safetensors.torch import save_file


def rename_weights_for_burn(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Rename HuggingFace T5 weight keys to match Burn's naming conventions.
    
    HuggingFace naming -> Burn naming mappings:
    - shared.weight -> shared.weight
    - encoder.block.{i}.layer.0.SelfAttention -> encoder.blocks.{i}.self_attention
    - decoder.block.{i}.layer.0.SelfAttention -> decoder.blocks.{i}.self_attention
    - decoder.block.{i}.layer.1.EncDecAttention -> decoder.blocks.{i}.cross_attention
    - encoder/decoder.block.{i}.layer.{1,2}.layer_norm -> layer_norm
    - encoder/decoder.block.{i}.layer.{1,2}.DenseReluDense -> feed_forward
    """
    burn_state_dict = {}
    
    for key, value in state_dict.items():
        new_key = key
        
        # Shared embeddings
        if key == "shared.weight":
            burn_state_dict["shared.weight"] = value
            continue
            
        # LM head (for conditional generation models)
        if key == "lm_head.weight":
            burn_state_dict["lm_head.weight"] = value
            continue
            
        # Encoder/Decoder blocks
        if "encoder.block" in key or "decoder.block" in key:
            # Parse the key structure
            parts = key.split(".")
            stack_type = parts[0]  # encoder or decoder
            block_idx = parts[2]   # block index
            
            # Rebuild key for Burn
            if "layer.0.SelfAttention" in key:
                # Self-attention weights
                sub_key = key.split("layer.0.SelfAttention.")[-1]
                
                # Map attention components
                if sub_key == "q.weight":
                    new_key = f"{stack_type}.blocks.{block_idx}.self_attention.q.weight"
                elif sub_key == "k.weight":
                    new_key = f"{stack_type}.blocks.{block_idx}.self_attention.k.weight"
                elif sub_key == "v.weight":
                    new_key = f"{stack_type}.blocks.{block_idx}.self_attention.v.weight"
                elif sub_key == "o.weight":
                    new_key = f"{stack_type}.blocks.{block_idx}.self_attention.o.weight"
                elif sub_key == "relative_attention_bias.weight":
                    new_key = f"{stack_type}.blocks.{block_idx}.self_attention.relative_attention_bias.weight"
                    
            elif "layer.1.EncDecAttention" in key:
                # Cross-attention weights (decoder only)
                sub_key = key.split("layer.1.EncDecAttention.")[-1]
                
                if sub_key == "q.weight":
                    new_key = f"{stack_type}.blocks.{block_idx}.cross_attention.q.weight"
                elif sub_key == "k.weight":
                    new_key = f"{stack_type}.blocks.{block_idx}.cross_attention.k.weight"
                elif sub_key == "v.weight":
                    new_key = f"{stack_type}.blocks.{block_idx}.cross_attention.v.weight"
                elif sub_key == "o.weight":
                    new_key = f"{stack_type}.blocks.{block_idx}.cross_attention.o.weight"
                    
            elif "layer.0.layer_norm" in key:
                # Self-attention layer norm
                sub_key = key.split("layer.0.layer_norm.")[-1]
                new_key = f"{stack_type}.blocks.{block_idx}.self_attention_layer_norm.{sub_key}"
                
            elif "layer.1.layer_norm" in key and "decoder" in stack_type:
                # Cross-attention layer norm (decoder only)
                sub_key = key.split("layer.1.layer_norm.")[-1]
                new_key = f"{stack_type}.blocks.{block_idx}.cross_attention_layer_norm.{sub_key}"
                
            elif "layer.1.layer_norm" in key and "encoder" in stack_type:
                # Feed-forward layer norm (encoder)
                sub_key = key.split("layer.1.layer_norm.")[-1]
                new_key = f"{stack_type}.blocks.{block_idx}.feed_forward_layer_norm.{sub_key}"
                
            elif "layer.2.layer_norm" in key:
                # Feed-forward layer norm (decoder)
                sub_key = key.split("layer.2.layer_norm.")[-1]
                new_key = f"{stack_type}.blocks.{block_idx}.feed_forward_layer_norm.{sub_key}"
                
            elif "DenseReluDense" in key or "DenseGatedActDense" in key:
                # Feed-forward network
                if "DenseReluDense" in key:
                    sub_key = key.split("DenseReluDense.")[-1]
                else:
                    sub_key = key.split("DenseGatedActDense.")[-1]
                    
                if sub_key == "wi.weight":
                    new_key = f"{stack_type}.blocks.{block_idx}.feed_forward.wi.weight"
                elif sub_key == "wo.weight":
                    new_key = f"{stack_type}.blocks.{block_idx}.feed_forward.wo.weight"
                elif sub_key == "wi_0.weight":  # Gated variant
                    new_key = f"{stack_type}.blocks.{block_idx}.feed_forward.wi_0.weight"
                elif sub_key == "wi_1.weight":  # Gated variant
                    new_key = f"{stack_type}.blocks.{block_idx}.feed_forward.wi_1.weight"
                    
        # Final layer norms
        elif key == "encoder.final_layer_norm.weight":
            new_key = "encoder.final_layer_norm.weight"
        elif key == "decoder.final_layer_norm.weight":
            new_key = "decoder.final_layer_norm.weight"
            
        # Embeddings
        elif key == "encoder.embed_tokens.weight":
            new_key = "encoder.embed_tokens.weight"
        elif key == "decoder.embed_tokens.weight":
            new_key = "decoder.embed_tokens.weight"
            
        burn_state_dict[new_key] = value
        
    return burn_state_dict


def save_config(config: T5Config, output_dir: Path):
    """Save T5 configuration in a format readable by Burn."""
    burn_config = {
        "vocab_size": config.vocab_size,
        "d_model": config.d_model,
        "d_kv": config.d_kv,
        "d_ff": config.d_ff,
        "n_layers": config.num_layers,
        "n_decoder_layers": config.num_decoder_layers,
        "n_heads": config.num_heads,
        "relative_attention_num_buckets": config.relative_attention_num_buckets,
        "relative_attention_max_distance": config.relative_attention_max_distance,
        "dropout_rate": config.dropout_rate,
        "layer_norm_eps": config.layer_norm_epsilon,
        "feed_forward_proj": "GatedGelu" if config.feed_forward_proj == "gated-gelu" else "ReLU",
        "use_cache": config.use_cache,
        "is_gated_act": config.is_gated_act if hasattr(config, "is_gated_act") else False,
    }
    
    with open(output_dir / "config.json", "w") as f:
        json.dump(burn_config, f, indent=2)


def convert_t5_model(
    model_name: str,
    output_dir: str,
    use_conditional_generation: bool = False,
    fp16: bool = False
):
    """
    Convert a HuggingFace T5 model to Burn format.
    
    Args:
        model_name: HuggingFace model name (e.g., 't5-small', 't5-base')
        output_dir: Directory to save converted weights
        use_conditional_generation: Use T5ForConditionalGeneration instead of T5Model
        fp16: Save weights in FP16 format
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model '{model_name}' from HuggingFace...")
    
    # Load model and config
    if use_conditional_generation:
        model = T5ForConditionalGeneration.from_pretrained(model_name)
    else:
        model = T5Model.from_pretrained(model_name)
    
    config = model.config
    
    # Get state dict
    state_dict = model.state_dict()
    
    # Convert to FP16 if requested
    if fp16:
        print("Converting weights to FP16...")
        state_dict = {k: v.half() for k, v in state_dict.items()}
    
    # Rename weights for Burn
    print("Renaming weights for Burn compatibility...")
    burn_state_dict = rename_weights_for_burn(state_dict)
    
    # Save weights
    weights_file = output_dir / "model.safetensors"
    print(f"Saving weights to {weights_file}...")
    save_file(burn_state_dict, str(weights_file))
    
    # Save config
    print(f"Saving configuration to {output_dir / 'config.json'}...")
    save_config(config, output_dir)
    
    # Print summary
    print(f"\nConversion complete!")
    print(f"Model: {model_name}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Output directory: {output_dir}")
    print(f"Files created:")
    print(f"  - model.safetensors ({weights_file.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"  - config.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HuggingFace T5 models to Burn format")
    parser.add_argument(
        "--model",
        type=str,
        default="t5-small",
        help="HuggingFace model name (e.g., t5-small, t5-base, google/flan-t5-small)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./t5-weights",
        help="Output directory for converted weights"
    )
    parser.add_argument(
        "--conditional-generation",
        action="store_true",
        help="Use T5ForConditionalGeneration (includes lm_head)"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Save weights in FP16 format"
    )
    
    args = parser.parse_args()
    
    convert_t5_model(
        model_name=args.model,
        output_dir=args.output,
        use_conditional_generation=args.conditional_generation,
        fp16=args.fp16
    )