import torch
from torch.utils.cpp_extension import load

lib = load(
    name="scale",
    sources=["scale.cu"],
    extra_cuda_cflags=[
        "-O3",
        "-lineinfo",
    ],
)


def test_correctness():
    for n in (1, 31, 32, 33, 255, 256, 257, 1_000_003):
        x = torch.rand(n, device="cuda", dtype=torch.float32)
        y = torch.full_like(x, float("nan"))
        scale = 2.0
        
        lib.scale_cuda(x, y, scale)
        torch.testing.assert_close(y, x * scale, rtol=0, atol=0)
        print(f"PASS N={n}")

if __name__ == "__main__":
    torch.manual_seed(0)
    test_correctness()
