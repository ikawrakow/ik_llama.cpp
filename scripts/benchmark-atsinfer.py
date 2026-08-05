#!/usr/bin/env python3
"""
ATSInfer Reproducible Benchmark Harness
Runs llama.cpp in baseline, static ATSInfer, and dynamic ATSInfer modes,
capturing system context, throughput metrics, and VRAM utilization.
"""

import sys
import os
import json
import time
import subprocess
import argparse
from datetime import datetime

def get_system_context():
    ctx = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "os": sys.platform,
        "python_version": sys.version,
    }
    
    # Try git commit hash
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        ctx["git_commit"] = commit
    except Exception:
        ctx["git_commit"] = "unknown"

    # Try nvidia-smi for GPU info
    try:
        gpu_info = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], stderr=subprocess.DEVNULL).decode().strip()
        ctx["gpu"] = gpu_info
    except Exception:
        ctx["gpu"] = "CPU / Unknown GPU"

    return ctx

def check_atsinfer_support(executable_path):
    """Return True if the binary's CLI parser knows the --atsinfer flags."""
    try:
        proc = subprocess.run([executable_path, "--help"], stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, timeout=60)
    except Exception as e:
        print(f"WARNING: could not probe '{executable_path}' for ATSInfer support: {e}")
        return False
    return "--atsinfer" in proc.stdout

def run_benchmark(executable_path, model_path, prompt, n_predict, vram_budget_mb, mode):
    cmd = [
        executable_path,
        "-m", model_path,
        "-p", prompt,
        "-n", str(n_predict),
    ]

    # NOTE: --atsinfer-vram-budget is expressed in MiB (see common/common.cpp),
    # so pass vram_budget_mb straight through -- do not convert to bytes.
    if mode == "static_atsinfer":
        cmd.extend(["--atsinfer", "--atsinfer-vram-budget", str(vram_budget_mb)])
    elif mode == "dynamic_atsinfer":
        cmd.extend(["--atsinfer", "--atsinfer-vram-budget", str(vram_budget_mb), "--atsinfer-dynamic"])

    print(f"Executing: {' '.join(cmd)}")
    start_t = time.time()
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300)
        elapsed = time.time() - start_t
        output = proc.stdout + "\n" + proc.stderr
        
        # Parse throughput if output contains tokens per second
        decode_tok_per_sec = 0.0
        prefill_tok_per_sec = 0.0
        
        for line in output.splitlines():
            if "eval time =" in line and "ms /" in line:
                # e.g., llama_print_timings: eval time = 1200.00 ms / 50 runs ( 24.00 ms per token, 41.67 tokens per second)
                if "tokens per second" in line:
                    try:
                        parts = line.split("(")
                        if len(parts) > 1:
                            sub = parts[1].split("tokens per second")[0].strip()
                            decode_tok_per_sec = float(sub)
                    except Exception:
                        pass
            elif "prompt eval time =" in line and "ms /" in line:
                if "tokens per second" in line:
                    try:
                        parts = line.split("(")
                        if len(parts) > 1:
                            sub = parts[1].split("tokens per second")[0].strip()
                            prefill_tok_per_sec = float(sub)
                    except Exception:
                        pass

        return {
            "mode": mode,
            "success": proc.returncode == 0,
            "total_latency_sec": elapsed,
            "decode_tokens_per_sec": decode_tok_per_sec,
            "prefill_tokens_per_sec": prefill_tok_per_sec,
            "raw_output_snippet": output[-500:] if len(output) > 500 else output
        }
    except Exception as e:
        return {
            "mode": mode,
            "success": False,
            "error": str(e)
        }

def main():
    parser = argparse.ArgumentParser(description="ATSInfer Reproducible Benchmark")
    parser.add_argument("--exe", default="./build/bin/Release/llama-cli.exe", help="Path to llama-cli executable")
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--prompt", default="Explain quantum computing in simple terms.", help="Benchmark prompt")
    parser.add_argument("--n-predict", type=int, default=128, help="Number of tokens to generate")
    parser.add_argument("--vram-budget-mb", type=int, default=4096, help="VRAM budget in MB")
    parser.add_argument("--output", default="atsinfer_benchmark_results.json", help="Output JSON path")

    args = parser.parse_args()

    atsinfer_supported = check_atsinfer_support(args.exe)
    if not atsinfer_supported:
        print("WARNING: binary does not expose --atsinfer flags; running baseline only.")

    sys_context = get_system_context()
    results = {
        "system_context": sys_context,
        "benchmark_config": {
            "model": args.model,
            "n_predict": args.n_predict,
            "vram_budget_mb": args.vram_budget_mb,
            "atsinfer_supported": atsinfer_supported
        },
        "runs": []
    }

    modes = ["baseline", "static_atsinfer", "dynamic_atsinfer"] if atsinfer_supported else ["baseline"]
    for mode in modes:
        print(f"\n--- Running Mode: {mode} ---")
        run_res = run_benchmark(args.exe, args.model, args.prompt, args.n_predict, args.vram_budget_mb, mode)
        results["runs"].append(run_res)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nBenchmark completed. Results written to {args.output}")

if __name__ == "__main__":
    main()
