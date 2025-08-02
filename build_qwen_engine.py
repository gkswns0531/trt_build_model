#!/usr/bin/env python3
"""
TensorRT-LLM Engine Builder Script - H100 Optimized Configuration
Complete implementation with all 70+ available parameters

H100 Specific Optimizations:
- FP8 Native Support (4th Gen Tensor Cores)
- Low Latency GEMM Plugins (SM90 Hopper)
- 3TB/s Memory Bandwidth Utilization
- Advanced Fusion Optimizations

Based on TensorRT-LLM 1.0.0rc5 official documentation:
https://nvidia.github.io/TensorRT-LLM/commands/trtllm-build.html
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def build_trtllm_engine(quantization="fp16"):
    """Build TensorRT-LLM engine from checkpoint with all available parameters"""
    
    # ==========================================================================
    # PATH CONFIGURATION
    # ==========================================================================
    checkpoint_dir = f"./{model_name}_checkpoints"
    engine_dir = f"./{model_name}_engine"

    if quantization == "fp8":
        checkpoint_dir = f"./{model_name}_checkpoints_fp8"
        engine_dir = f"./{model_name}_engine_fp8"
    elif quantization == "fp16":
        checkpoint_dir = f"./{model_name}_checkpoints_fp16"
        engine_dir = f"./{model_name}_engine_fp16"
    else:
        print(f"Error: Invalid quantization type: {quantization}")
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_dir):
        print(f"Error: Checkpoint directory {checkpoint_dir} not found!")
        print("Please run convert_qwen_trtllm.py first")
        return False
    
    # Create engine directory
    os.makedirs(engine_dir, exist_ok=True)
    
    print("Building TensorRT-LLM engine...")
    print(f"Quantization type: {quantization.upper()}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Engine directory: {engine_dir}")
    
    # ==========================================================================
    # TENSORRT-LLM BUILD COMMAND - COMPLETE OPTIONS IMPLEMENTATION
    # ==========================================================================
    cmd = [
        "trtllm-build",
        
        # ======================================================================
        # MANDATORY PARAMETERS
        # ======================================================================
        "--checkpoint_dir", checkpoint_dir,  # TensorRT-LLM checkpoint directory path (required)
        "--output_dir", engine_dir,          # Engine and config file save directory (default: engine_outputs)
        
        # ======================================================================
        # MODEL CONFIGURATION
        # ======================================================================
        # "--model_config", "path/to/config.json",     # TensorRT-LLM checkpoint config file path (default: None)
        # "--build_config", "path/to/build.json",     # TensorRT-LLM build config file path (default: None)
        # "--model_cls_file", "path/to/model.py",     # Custom TensorRT-LLM model file path (default: None)
        # "--model_cls_name", "CustomModel",          # Custom TensorRT-LLM model class name (default: None)
        
        # ======================================================================
        # BATCH AND SEQUENCE CONFIGURATION - BATCH SIZE 16 + 2K INPUT OPTIMIZED
        # ======================================================================
        "--max_batch_size", "16",           # Maximum number of requests the engine can schedule - increased for batch processing (default: 2048)
        "--max_input_len", "2500",           # Maximum input length for a single request - 2K + buffer (default: 1024)
        "--max_seq_len", "4096",             # Maximum total length including prompt+output - 2K input + 2K output (default: None, inferred from model config)
        "--max_beam_width", "1",             # Maximum beam count for beam search decoding (default: 1)
        "--max_num_tokens", "65536",         # Maximum tokens per batch after padding removal - 16 * 4096 for 2K input processing (default: 8192)
        "--opt_num_tokens", "32768",         # Optimal tokens per batch after padding removal - 16 * 2048 optimal (default: max_batch_size * max_beam_width)
        # "--max_encoder_input_len", "1024", # Maximum encoder input length for encoder-decoder models (default: 1024)
        # "--max_prompt_embedding_table_size", "0", # Maximum size for prompt tuning or multimodal input (default: 0)
        
        # ======================================================================
        # KV CACHE CONFIGURATION - BATCH SIZE 16 OPTIMIZED
        # ======================================================================
        "--kv_cache_type", "paged",          # KV cache type (continuous/paged/disabled, default: None)
        "--tokens_per_block", "32",         # Tokens per block for paged KV cache - increased for batch processing (default: 32)
        "--paged_kv_cache", "enable",      # Enable paged KV cache (enable/disable, default: enable if kv_cache_type=paged)
        
        # ======================================================================
        # TIMING CACHE AND PROFILING
        # ======================================================================
        # "--input_timing_cache", "qwen_timing.cache",   # Timing cache file path to read (default: None)
        # "--output_timing_cache", "qwen_timing.cache",  # Timing cache file path to write (default: model.cache)
        "--profiling_verbosity", "detailed", # Profiling verbosity for generated TensorRT engine (layer_names_only/detailed/none, default: layer_names_only)
        
        # ======================================================================
        # PLUGIN CONFIGURATION - ATTENTION
        # ======================================================================
        "--gpt_attention_plugin", "auto",    # Attention plugin for GPT-style decoder models (default: auto)
        "--bert_attention_plugin", "auto",   # Attention plugin for BERT-style encoder models (default: auto)
        
        # ======================================================================
        # PLUGIN CONFIGURATION - GEMM
        # ======================================================================
    ]
    
    # Add quantization-specific GEMM plugin settings
    if quantization == "fp8":
        cmd.extend([
            # H100 optimized FP8 GEMM plugins
            "--gemm_plugin", "fp8",                      # General GEMM plugin for FP8 precision (default: auto)
            "--fp8_rowwise_gemm_plugin", "auto",         # Rowwise GEMM plugin for FP8 (default: disable)
            "--low_latency_gemm_plugin", "fp8",          # Low latency GEMM plugin - H100 SM90 optimized (default: disable)
            "--gemm_swiglu_plugin", "fp8",               # GEMM + SwiGLU fusion - fallback option (default: disable)
            "--low_latency_gemm_swiglu_plugin", "fp8",   # Low latency GEMM + SwiGLU fusion - priority option (default: disable)
        ])
    else:  # fp16, bf16, or other quantization
        cmd.extend([
            "--gemm_plugin", "auto",                     # General GEMM plugin - auto for non-FP8 (default: auto)
            "--fp8_rowwise_gemm_plugin", "disable",      # Disable FP8 rowwise for non-FP8 quantization (default: disable)
        ])
    
    cmd.extend([
        # ======================================================================
        # PLUGIN CONFIGURATION - OTHERS
        # ======================================================================
        "--nccl_plugin", "auto",             # NCCL plugin for multi-GPU communication (default: auto)
        "--moe_plugin", "auto",              # MoE layer acceleration plugin (default: auto)
        "--mamba_conv1d_plugin", "auto",     # Mamba conv1d operator acceleration plugin (default: auto)
        "--lora_plugin", "disable",          # LoRA plugin enable (default: None)
        "--dora_plugin", "disable",          # DoRA plugin enable (default: disable)
        "--gemm_allreduce_plugin", "disable", # GEMM + AllReduce kernel fusion plugin (default: None)
        
        # ======================================================================
        # FUSION OPTIMIZATION CONFIGURATION
        # ======================================================================
        "--context_fmha", "enable",          # Enable fused multi-head attention in context phase (default: enable)
        "--use_paged_context_fmha", "enable", # Allow advanced features like KV cache reuse and chunked context (default: enable)
    ])
    
    # Add quantization-specific context FMHA settings
    if quantization == "fp8":
        cmd.extend([
            "--use_fp8_context_fmha", "enable",   # Accelerate attention with FP8 context FMHA (default: disable)
        ])
    else:  # fp16, bf16, or other quantization
        cmd.extend([
            "--use_fp8_context_fmha", "disable",  # Disable FP8 context FMHA for non-FP8 quantization (default: disable)
        ])
    
    cmd.extend([
        "--norm_quant_fusion", "enable",      # Fuse LayerNorm and quantization kernels (default: disable)
        "--reduce_fusion", "enable",          # Fuse ResidualAdd and LayerNorm kernels after AllReduce (default: disable)
        "--use_fused_mlp", "enable",          # Fuse two Matmul operations into one in Gated-MLP (default: enable)
        "--user_buffer", "enable",            # Remove additional copy between local-shared buffers (default: disable)
        "--remove_input_padding", "enable",   # Pack tokens together to reduce computation and memory (default: enable)
        "--fuse_fp4_quant", "disable",       # Fuse FP4 quantization into attention kernels (default: disable)
        "--bert_context_fmha_fp32_acc", "disable", # FP32 accumulator for context FMHA in BERT (default: disable)
        
        # ======================================================================
        # ADVANCED OPTIMIZATION CONFIGURATION
        # ======================================================================
        "--multiple_profiles", "disable",    # Enable multiple TensorRT optimization profiles (default: disable)
        "--paged_state", "enable",           # Paged state for memory-efficient RNN state management (default: enable)
        "--streamingllm", "disable",         # StreamingLLM - windowed attention for long text (default: disable)
        "--pp_reduce_scatter", "disable",    # Reduce scatter optimization in pipeline parallelization (default: disable)
        
        # ======================================================================
        # BUILD OPTIMIZATION CONFIGURATION
        # ======================================================================
        "--workers", "8",                     # Number of workers for parallel building - H100 optimized (default: 1)
        # Note: weight_sparsity is a flag option, not requiring a value
        # Note: fast_build is a flag option, not requiring a value  
        # Note: weight_streaming is a flag option, not requiring a value
        # Note: strip_plan is a flag option, not requiring a value
        
        # ======================================================================
        # LOGGING AND DEBUGGING
        # ======================================================================
        "--log_level", "info",                # Logging level (default: info)
        # "--monitor_memory",                  # Enable memory monitor during engine build (default: False)
        # "--enable_debug_output",            # Enable debug output (default: False)
        # "--dry_run",                        # Run build process without actual engine build (default: False)
        # "--visualize_network", "path/to/dir", # Export TensorRT network to ONNX for debugging (default: None)
        
        # ======================================================================
        # LOGITS CONFIGURATION
        # ======================================================================
        # "--logits_dtype", "float16",        # Data type for logits (default: None)
        # "--gather_context_logits",          # Enable context logits gathering (default: False)
        # "--gather_generation_logits",       # Enable generation logits gathering (default: False)
        # "--gather_all_token_logits",        # Enable all token logits gathering (default: False)
        
        # ======================================================================
        # LORA CONFIGURATION
        # ======================================================================
        # "--lora_dir", "path/to/lora1", "path/to/lora2", # LoRA weight directories (default: None)
        # "--lora_ckpt_source", "hf",         # LoRA checkpoint source type (default: hf)
        # "--lora_target_modules", "attn_qkv", "attn_dense", # Target module names for LoRA (default: None)
        # "--max_lora_rank", "64",            # Maximum LoRA rank for workspace calculation (default: 64)
        
        # ======================================================================
        # SPECULATIVE DECODING CONFIGURATION
        # ======================================================================
        # "--speculative_decoding_mode", "none", # Speculative decoding mode (default: None)
        # "--max_draft_len", "0",             # Maximum draft token length for speculative decoding (default: 0)
        
        # ======================================================================
        # AUTO PARALLELIZATION CONFIGURATION
        # ======================================================================
        # "--auto_parallel", "1",             # MPI world size for auto parallelization (default: 1)
        # "--gpus_per_node", "8",             # Number of GPUs per node in multi-node setup (default: 8)
        # "--cluster_key", "H100-SXM",       # Target GPU type - H100 optimized (default: None, auto-detected)
    ])
    
    print("=" * 80)
    print("TRTLLM-BUILD COMMAND:")
    print("=" * 80)
    print(f"Command: {' '.join(cmd[:3])}")
    print("Active Parameters:")
    for i in range(3, len(cmd), 2):
        if i + 1 < len(cmd):
            print(f"  {cmd[i]} = {cmd[i+1]}")
    print("=" * 80)
    
    # Execute build
    result = subprocess.run(cmd, check=True)
    print("\n" + "=" * 80)
    print("ENGINE BUILD COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Engine saved to: {engine_dir}")
    print("Ready for inference!")
    return True

def main():
    parser = argparse.ArgumentParser(description="TensorRT-LLM Engine Builder - H100 Optimized Configuration")
    parser.add_argument("--quantization", type=str, default="fp16", 
                       choices=["fp16", "fp8", "bf16"],
                       help="Quantization type (default: fp16)")
    parser.add_argument("--model_name", type=str, default="qwen3-30b-a3b",
                       help="Model name (default: qwen3-30b-a3b)")
    
    args = parser.parse_args()
    
    global model_name
    model_name = args.model_name
    
    print(f"Building TensorRT-LLM engine with {args.quantization.upper()} quantization")
    print(f"Model: {model_name}")
    print("-" * 50)
    
    success = build_trtllm_engine(quantization=args.quantization)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 