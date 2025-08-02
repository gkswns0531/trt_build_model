#!/usr/bin/env python3
import time
import argparse
import torch
import os
import gc

def read_prompts(file_path="prompts.txt", num_prompts=16):
    """Read 16 prompts from file (5-line prompts separated by blank lines)"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    prompt_blocks = [block.strip() for block in content.split('\n\n') if block.strip()]
    
    prompts = []
    for block in prompt_blocks:
        single_prompt = ' '.join(line.strip() for line in block.split('\n') if line.strip())
        if single_prompt:
            prompts.append(single_prompt[:2000])
    
    while len(prompts) < num_prompts:
        if prompts:
            prompts.append(prompts[0])
        else:
            single_prompt = ' '.join(line.strip() for line in content.split('\n') if line.strip())
            prompts = [single_prompt[:2000]] * num_prompts
            break
    
    return prompts[:num_prompts]

def setup_vllm(model_name, quantization="fp16"):
    from vllm import LLM
    if quantization == "fp8":
        return LLM(
            model=f"./{model_name}",
            tensor_parallel_size=1,
            quantization="fp8",
            kv_cache_dtype="auto",
            max_model_len=4096,
            gpu_memory_utilization=0.85,
            max_num_batched_tokens=65536
        )
    else:
        return LLM(
            model=f"./{model_name}",
            tensor_parallel_size=1,
            dtype="float16",
            max_model_len=4096,
            gpu_memory_utilization=0.85,
            max_num_batched_tokens=65536
        )

def setup_trtllm_torch(model_name, quantization="fp16"):
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    os.environ["TORCH_DYNAMO_DISABLE"] = "1"
    from tensorrt_llm import LLM
    
    if quantization == "fp8":
        return LLM(
            model=f"./{model_name}-fp8",
            backend="pytorch",
            max_num_tokens=65536,
            max_batch_size=16
        )
    else:
        return LLM(
            model=f"./{model_name}",
            backend="pytorch",
            max_num_tokens=65536,
            max_batch_size=16
        )

def setup_trtllm_trt(model_name, quantization="fp16"):
    from tensorrt_llm._tensorrt_engine import LLM
    return LLM(
        model=f"./{model_name}_engine_{quantization}",
        tokenizer=f"./{model_name}"
    )

def clean_vram():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def measure_ttft(model, prompts, backend_name):
    """Measure Time To First Token for batch of 16 prompts"""
    if "vLLM" in backend_name:
        from vllm import SamplingParams
    else:
        from tensorrt_llm import SamplingParams
    
    sampling_params = SamplingParams(max_tokens=1, temperature=0.1, top_p=0.9)
    
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    model.generate(prompts, sampling_params)
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    return (end_time - start_time) * 1000

def measure_tps(model, prompts, backend_name):
    """Measure Tokens Per Second for batch of 16 prompts"""
    if "vLLM" in backend_name:
        from vllm import SamplingParams
    else:
        from tensorrt_llm import SamplingParams
    
    sampling_params = SamplingParams(
        max_tokens=1024, 
        temperature=0.1, 
        top_p=0.9,
        stop=["<|endoftext|>", "<|im_end|>"]
    )
    
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    outputs = model.generate(prompts, sampling_params)
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    
    total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    avg_tokens_per_request = total_tokens / len(outputs)
    total_tps = total_tokens / total_time
    
    generated_text = outputs[0].outputs[0].text
    
    return total_tps, total_tokens, generated_text, total_time

def measure_inference(model, prompts, backend_name):
    """Measure inference performance for batch of 16 prompts"""
    print(f"\n{'='*50}")
    print(f"Testing {backend_name} (Batch Size: {len(prompts)})")
    print(f"{'='*50}")
    
    for i in range(2):
        measure_tps(model, prompts, backend_name)
        print(f"Warmup {i+1}/2 completed")
    
    print("Measuring TTFT (1 token)...")
    ttft = measure_ttft(model, prompts, backend_name)
    
    print("Measuring TPS (full generation)...")
    tps, num_tokens, generated_text, total_time = measure_tps(model, prompts, backend_name)
    
    print(f"Generated tokens: {num_tokens} (total across {len(prompts)} requests)")
    print(f"Average tokens per request: {num_tokens / len(prompts):.1f}")
    print(f"Total generation time: {total_time:.3f}s")
    print(f"TTFT: {ttft:.3f} ms")
    print(f"TPS: {tps:.2f} tokens/sec (total throughput)")
    print(f"Generated text preview: {generated_text[:100]}...")
    
    return {
        'backend': backend_name,
        'ttft': ttft,
        'tps': tps,
        'num_tokens': num_tokens,
        'total_time': total_time,
        'generated_text': generated_text
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3-30b-a3b", help="Model name/path")
    parser.add_argument("--prompt_file", default="prompts.txt", help="Prompt file path")
    parser.add_argument("--quantization", default="fp16", choices=["fp16", "fp8"], help="Quantization type")
    parser.add_argument("--backends", nargs="+", default=["vllm", "torch", "trt"], 
                      choices=["vllm", "torch", "trt"], help="Backends to test")
    args = parser.parse_args()
    
    prompts = read_prompts(args.prompt_file)
    print(f"Loaded {len(prompts)} prompts from {args.prompt_file}")
    print(f"Quantization: {args.quantization.upper()}")
    print(f"First prompt preview: {prompts[0][:100]}...")
    
    results = []
    
    for backend in args.backends:
        if backend == "vllm":
            model = setup_vllm(args.model, args.quantization)
            result = measure_inference(model, prompts, f"vLLM ({args.quantization.upper()})")
            
        elif backend == "torch":
            model = setup_trtllm_torch(args.model, args.quantization)
            result = measure_inference(model, prompts, f"Torch ({args.quantization.upper()})")
            
        elif backend == "trt":
            model = setup_trtllm_trt(args.model, args.quantization)
            result = measure_inference(model, prompts, f"TRT ({args.quantization.upper()})")
        
        results.append(result)
        
        # VRAM 청소
        del model
        clean_vram()
        print(f"VRAM cleaned after {backend}")
    
    # 결과 요약
    print(f"\n{'='*80}")
    print("BATCH PERFORMANCE COMPARISON SUMMARY (16 Prompts)")
    print(f"{'='*80}")
    print(f"{'Backend':<20} {'TTFT (ms)':<12} {'Total TPS':<12} {'Total Tokens':<12} {'Total (s)':<10}")
    print(f"{'-'*80}")
    
    for result in results:
        print(f"{result['backend']:<20} {result['ttft']:<12.3f} {result['tps']:<12.2f} {result['num_tokens']:<12} {result['total_time']:<10.3f}")
    
    # 최고 성능
    fastest_tps = max(results, key=lambda x: x['tps'])
    fastest_ttft = min(results, key=lambda x: x['ttft'])
    
    print(f"\nFastest Total TPS: {fastest_tps['backend']} ({fastest_tps['tps']:.2f} tokens/sec)")
    print(f"Fastest TTFT: {fastest_ttft['backend']} ({fastest_ttft['ttft']:.3f}ms)")
    print(f"Average tokens per request: {fastest_tps['num_tokens'] / 16:.1f}")

if __name__ == "__main__":
    main()