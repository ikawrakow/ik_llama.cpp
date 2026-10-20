# Token generation performance troubleshooting

## Verifying that the model is running on the GPU with CUDA
Make sure you compiled llama with the correct env variables according to [this guide](/docs/build.md#cuda), so that llama accepts the `-ngl N` (or `--n-gpu-layers N`) flag. When running llama, you may configure `N` to be very large, and llama will offload the maximum possible number of layers to the GPU, even if it's less than the number you configured. For example:
```shell
./llama-cli -m "path/to/model.gguf" -ngl 200000 -p "Please sir, may I have some "
```

When running llama, before it starts the inference work, it will output diagnostic information that shows whether cuBLAS is offloading work to the GPU. Look for these lines:
```shell
llama_model_load_internal: [cublas] offloading 60 layers to GPU
llama_model_load_internal: [cublas] offloading output layer to GPU
llama_model_load_internal: [cublas] total VRAM used: 17223 MB
... rest of inference
```

If you see these lines, then the GPU is being used.

## Verifying that the CPU is not oversaturated
llama accepts a `-t N` (or `--threads N`) parameter. It's extremely important that this parameter is not too large. If your token generation is extremely slow, try setting this number to 1. If this significantly improves your token generation speed, then your CPU is being oversaturated and you need to explicitly set this parameter to the number of the physical CPU cores on your machine (even if you utilize a GPU). If in doubt, start with 1 and double the amount until you hit a performance bottleneck, then scale the number down.

## Thermal throttling on laptops (external tools)

On thermally constrained laptops, CPU inference can start at 25–30 t/s and
drop to 15–18 t/s after 1–2 minutes when the package reaches 95–100 °C.
This is thermal throttling, not an `ik_llama.cpp` bug. The engine does not
change CPU frequency; use the OS tools below.

Linux:

```bash
# show the current limits
cpupower frequency-info

# cap the P-cores used by --cpu-affinity (example: 0-7) at 2.6 GHz
sudo cpupower -c 0-7 frequency-set -u 2600MHz

# restore the hardware maximum (see frequency-info)
sudo cpupower -c 0-7 frequency-set -u 4600MHz
```

The same can be done per CPU via
`/sys/devices/system/cpu/cpuN/cpufreq/scaling_max_freq`. The cap is
system-wide: it also affects other applications, and a crash or `SIGKILL`
leaves it in place until reboot or reset.

Measure before/after with `llama-bench -p 512 -n 512` and watch
`turbostat --show PkgWatt,PkgTmp,Bzy_MHz`. Use `--cpu-affinity` so the cap
and the worker threads target the same cores.

Windows: set "Maximum processor state" to 99% in the power plan (disables
the aggressive turbo), or use ThrottleStop/UXTU. macOS: there is no
reliable user-space CPU frequency cap on Apple Silicon; limit threads or
context instead.

For automation, wrap these commands in a script or a systemd unit; the
engine does not manage CPU frequency.

# Example of runtime flags effect on inference speed benchmark
These runs were tested on the following machine:
GPU: A6000 (48GB VRAM)
CPU: 7 physical cores
RAM: 32GB

Model: `TheBloke_Wizard-Vicuna-30B-Uncensored-GGML/Wizard-Vicuna-30B-Uncensored.q4_0.gguf` (30B parameters, 4bit quantization, GGML)

Run command: `./llama-cli -m "path/to/model.gguf" -p "An extremely detailed description of the 10 best ethnic dishes will follow, with recipes: " -n 1000 [additional benchmark flags]`

Result:

| command | tokens/second (higher is better) |
| - | - |
| -ngl 2000000 | N/A (less than 0.1) |
| -t 7 | 1.7 |
| -t 1 -ngl 2000000 | 5.5 |
| -t 7 -ngl 2000000 | 8.7 |
| -t 4 -ngl 2000000 | 9.1 |
