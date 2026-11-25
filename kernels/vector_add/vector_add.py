import time
import torch
from torch.utils.cpp_extension import load
from pathlib import Path
from statistics import median

torch.set_grad_enabled(False)


lib = load(
    name="vector_add",
    sources=["vector_add.cu"],
    extra_cuda_cflags=[
        "-O3",
        "-lineinfo",
    ]
)

def test_correctness():
    for n in (1, 31, 32, 33, 255, 256, 257, 1_000_003):
        a = torch.rand(n, device="cuda", dtype=torch.float32)
        b = torch.rand(n, device="cuda", dtype=torch.float32)
        out = torch.full_like(a, float("nan"))
        
        lib.vector_add_out(a, b, out)
        torch.testing.assert_close(out, torch.add(a, b), rtol=0, atol=0)
        print(f"PASS N={n}")

def benchmark(n, warmup=20, repeats=100, rounds=5):
    a = torch.rand(n, device="cuda", dtype=torch.float32)
    b = torch.randn_like(a)
    out = torch.empty_like(a)
    
    for _ in range(warmup):
        lib.vector_add_out(a, b, out)
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples_us = []
        
    for _ in range(rounds):
        start.record()
        for _ in range(repeats):
            lib.vector_add_out(a, b, out)
        end.record()
        end.synchronize()
        
        samples_us.append(start.elapsed_time(end) * 1000 / repeats)
        
    print(f"N={n}, median={median(samples_us):.3f} us per launch")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_correctness()
    benchmark(257)
    benchmark(1_000_003)