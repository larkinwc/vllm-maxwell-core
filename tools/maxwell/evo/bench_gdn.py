#!/usr/bin/env python3
"""E31 recon: isolate the FLA Triton GDN packed-decode kernel and measure
runtime + effective state bandwidth on one die.

Shapes = Qwen3.5-9B at TP=4: K=V=128, H=4, HV=8. State per (seq, v-head)
is V*K*4B = 64 KiB fp32; per call the kernel reads + writes it once, so
min traffic = B*HV*2*64KiB. Reports GB/s vs the 73 GB/s ceiling.
"""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.expanduser(
    "~/repos/ml-maxwell/vllm-maxwell-core"))

from vllm.model_executor.layers.fla.ops.fused_recurrent import (  # noqa: E402
    fused_recurrent_gated_delta_rule_packed_decode,
)

K = V = 128
H, HV = 4, 8
REPS = 200


def bench(batch):
    torch.manual_seed(0)
    dev = "cuda"
    ntok = batch
    mixed = torch.randn(ntok, 2 * H * K + HV * V, dtype=torch.float16,
                        device=dev) * 0.3
    a = torch.randn(ntok, HV, dtype=torch.float32, device=dev) * 0.1
    b = torch.randn(ntok, HV, dtype=torch.float32, device=dev) * 0.1
    A_log = torch.randn(HV, dtype=torch.float32, device=dev) * 0.1
    dt_bias = torch.randn(HV, dtype=torch.float32, device=dev) * 0.1
    # paged state: slots 1..ntok (0 = NULL)
    state = torch.randn(ntok + 1, HV, V, K, dtype=torch.float32,
                        device=dev) * 0.05
    idx = torch.arange(1, ntok + 1, dtype=torch.int32, device=dev)
    out = torch.empty(ntok, 1, HV, V, dtype=torch.float16, device=dev)

    def call():
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed, a=a, b=b, A_log=A_log, dt_bias=dt_bias,
            scale=K ** -0.5, initial_state=state, out=out,
            ssm_state_indices=idx, use_qk_l2norm_in_kernel=True,
        )

    for _ in range(10):
        call()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(REPS):
        call()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / REPS
    traffic = batch * HV * V * K * 4 * 2  # state read+write
    print(f"GDN triton decode b={batch}: {dt * 1e3:7.3f} ms  "
          f"{traffic / dt / 1e9:6.1f} GB/s state-BW  "
          f"(x24 layers = {dt * 24 * 1e3:6.2f} ms/step)")
    return out


def main():
    for batch in (1, 8, 32):
        bench(batch)
    print("GDN_BENCH_DONE")


if __name__ == "__main__":
    main()
