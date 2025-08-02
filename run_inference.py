#!/usr/bin/env python3
import time
import argparse
import torch
import os

def read_prompt(file_path="prompts.txt"):
    with open(file_path, 'r', encoding='utf-8') as f:
        prompt = f.read().strip()
    if len(prompt) > 2000:
        prompt = prompt[:2000]
    return prompt

def setup_vllm(quantization="fp16"):
    from vllm import LLM, SamplingParams
    
    dtype = "float16" if quantization == "fp16" else "float8_e4m3fn"
    
    model = LLM(
        model="./qwen3-30b-a3b",
        tensor_parallel_size=1,
        dtype=dtype,
        max_model_len=3072,
        gpu_memory_utilization=0.8
    )
    
    return model

def setup_trtllm_torch(quantization="fp16"):
    from tensorrt_llm import LLM
    
    if quantization == "fp8":
        model_path = "./qwen3-30b-a3b_checkpoints_fp8"
    else:
        model_path = "./qwen3-30b-a3b"
    
    model = LLM(
        model=model_path,
        backend="pytorch"  # "torch" -> "pytorch"
    )
    
    return model

def setup_trtllm_trt(quantization="fp16"):
    from tensorrt_llm import LLM
    
    model_path = f"./qwen3-30b-a3b_engine_{quantization}"
    
    model = LLM(
        model=model_path
        # backend는 기본값이 TensorRT이므로 생략
    )
    
    return model

def measure_ttft(model, prompt, backend_name):
    if "vLLM" in backend_name:
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            max_tokens=1,
            temperature=0.1,
            top_p=0.9
        )
    else:
        from tensorrt_llm import SamplingParams
        sampling_params = SamplingParams(
            max_tokens=1,
            temperature=0.1,
            top_p=0.9
        )
    
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    outputs = model.generate([prompt], sampling_params)
    
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    ttft = end_time - start_time
    return ttft

def measure_tps(model, prompt, backend_name):
    if "vLLM" in backend_name:
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            max_tokens=1024,
            temperature=0.1,
            top_p=0.9,
            stop=["<|endoftext|>", "<|im_end|>"]
        )
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
    
    outputs = model.generate([prompt], sampling_params)
    
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    
    # 출력 구조 처리 - 두 라이브러리 모두 비슷한 구조 사용
    if "vLLM" in backend_name:
        generated_text = outputs[0].outputs[0].text
        num_tokens = len(outputs[0].outputs[0].token_ids) if hasattr(outputs[0].outputs[0], 'token_ids') else len(generated_text.split())
    else:
        # TensorRT-LLM의 RequestOutput 구조
        if hasattr(outputs[0], 'outputs'):
            generated_text = outputs[0].outputs[0].text
            num_tokens = len(outputs[0].outputs[0].token_ids) if hasattr(outputs[0].outputs[0], 'token_ids') else len(generated_text.split())
        else:
            # 대체 구조인 경우
            generated_text = str(outputs[0])
            num_tokens = len(generated_text.split())
    
    tps = num_tokens / total_time if total_time > 0 else 0
    
    return tps, num_tokens, generated_text, total_time

def measure_inference(model, prompt, backend_name):
    print(f"\n{'='*50}")
    print(f"Testing {backend_name}")
    print(f"{'='*50}")
    
    # Warmup runs (2회)
    for i in range(2):
        try:
            _ = measure_ttft(model, prompt, backend_name)
            print(f"Warmup {i+1}/2 completed")
        except:
            print(f"Warmup {i+1}/2 failed, continuing...")
    
    # TTFT measurement (1 token)
    print("Measuring TTFT (1 token)...")
    ttft = measure_ttft(model, prompt, backend_name)
    
    # TPS measurement (full generation)
    print("Measuring TPS (full generation)...")
    tps, num_tokens, generated_text, total_time = measure_tps(model, prompt, backend_name)
    
    print(f"Generated tokens: {num_tokens}")
    print(f"Total generation time: {total_time:.3f}s")
    print(f"TTFT: {ttft:.3f}s")
    print(f"TPS: {tps:.2f} tokens/sec")
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
    parser.add_argument("--prompt_file", default="prompts.txt", help="Prompt file path")
    parser.add_argument("--quantization", default="fp16", choices=["fp16", "fp8"], help="Quantization type")
    parser.add_argument("--backends", nargs="+", default=["vllm", "torch", "trt"], 
                      choices=["vllm", "torch", "trt"], help="Backends to test")
    args = parser.parse_args()
    
    # Read prompt from file
    try:
        prompt = read_prompt(args.prompt_file)
        print(f"Loaded prompt from {args.prompt_file}")
        print(f"Quantization: {args.quantization.upper()}")
        print(f"Prompt: {prompt[:100]}...")
    except FileNotFoundError:
        print(f"Prompt file {args.prompt_file} not found!")
        return
    
    results = []
    
    # Test each backend
    for backend in args.backends:
        try:
            if backend == "vllm":
                if not os.path.exists("./qwen3-30b-a3b"):
                    print(f"Skipping vLLM: model directory not found")
                    continue
                model = setup_vllm(args.quantization)
                result = measure_inference(model, prompt, f"vLLM ({args.quantization.upper()})")
                
            elif backend == "torch":
                model_path = "./qwen3-30b-a3b_checkpoints_fp8" if args.quantization == "fp8" else "./qwen3-30b-a3b"
                if not os.path.exists(model_path):
                    print(f"Skipping TensorRT-LLM Torch: {model_path} not found")
                    continue
                model = setup_trtllm_torch(args.quantization)
                result = measure_inference(model, prompt, f"TensorRT-LLM Torch ({args.quantization.upper()})")
                
            elif backend == "trt":
                engine_path = f"./qwen3-30b-a3b_engine_{args.quantization}"
                if not os.path.exists(engine_path):
                    print(f"Skipping TensorRT-LLM TRT: {engine_path} not found")
                    continue
                model = setup_trtllm_trt(args.quantization)
                result = measure_inference(model, prompt, f"TensorRT-LLM TRT ({args.quantization.upper()})")
            
            results.append(result)
            
        except Exception as e:
            print(f"Error with {backend}: {e}")
            continue
    
    # Summary comparison
    print(f"\n{'='*70}")
    print("PERFORMANCE COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'Backend':<20} {'TTFT (s)':<12} {'TPS':<12} {'Tokens':<8} {'Total (s)':<10}")
    print(f"{'-'*70}")
    
    for result in results:
        print(f"{result['backend']:<20} {result['ttft']:<12.3f} {result['tps']:<12.2f} {result['num_tokens']:<8} {result['total_time']:<10.3f}")
    
    # Find fastest
    if results:
        fastest_tps = max(results, key=lambda x: x['tps'])
        fastest_ttft = min(results, key=lambda x: x['ttft'])
        
        print(f"\nFastest TPS: {fastest_tps['backend']} ({fastest_tps['tps']:.2f} tokens/sec)")
        print(f"Fastest TTFT: {fastest_ttft['backend']} ({fastest_ttft['ttft']:.3f}s)")
        
        if len(results) > 1:
            speedup_tps = fastest_tps['tps'] / min(result['tps'] for result in results)
            speedup_ttft = max(result['ttft'] for result in results) / fastest_ttft['ttft']
            print(f"TPS Speedup: {speedup_tps:.2f}x")
            print(f"TTFT Speedup: {speedup_ttft:.2f}x")

if __name__ == "__main__":
    main()