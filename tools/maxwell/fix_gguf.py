"""Rewrite a llama.cpp Qwen3.5 GGUF into HF-native layout so that vLLM
(which expects HF conventions) produces correct output on Maxwell (sm_50).

Background
----------
llama.cpp's Qwen3.5 conversion (conversion/qwen.py:
_LinearAttentionVReorderBase + Qwen3NextModel.modify_tensors) applies three
transforms that vLLM's GGUF loader does not invert. Loading such a GGUF in
vLLM therefore yields gibberish. This script inverts all three:

  1. RMSNorm weights stored as (w + 1)  -> subtract 1
     (all *norm.weight except linear_attn.norm.weight / ssm_norm)
  2. ssm_a stores -exp(A_log)           -> A_log = log(-x)
  3. GDN V-heads reordered grouped->tiled (num_k_heads != num_v_heads)
     invert tiled->grouped for: in_proj_qkv (V rows), in_proj_z (rows),
     in_proj_a/b (rows), A_log/dt_bias (elems), conv1d (V channels),
     out_proj (input columns).

Quantized GDN tensors are reordered losslessly at the byte-row level (GGUF
quantizes along the input dim, so each output row is an independent block
group). out_proj needs an input-dim (column) reorder, which breaks Q8_0
blocks, so it is dequantized -> reordered -> requantized (Q8_0).

Two output variants are useful:
  * default            : quantized in_proj_qkv/in_proj_z preserved.
  * --f16-inproj       : dequantize in_proj_qkv/in_proj_z to F16. This is a
                         workaround for a vLLM merged-GGUF quantized loader
                         issue on the fused in_proj_qkvz path; the F16 route
                         goes through the well-tested unquantized loader.
                         NOTE: with the sm_50 `__dp4a` dequant fix in
                         quantization/gguf.py, the plain quantized variant is
                         expected to work too; keep --f16-inproj as a fallback.

Usage:
    python fix_gguf.py SRC.gguf DST.gguf [--f16-inproj]
"""
import sys
import numpy as np
import gguf
from gguf import GGUFReader, GGUFWriter, GGUFValueType, GGMLQuantizationType as QT

SRC = sys.argv[1]
DST = sys.argv[2]
F16_INPROJ = "--f16-inproj" in sys.argv[3:]

# Qwen3.5-9B GDN dims (from HF config Qwen/Qwen3.5-9B)
HK = 128          # linear_key_head_dim
HV = 128          # linear_value_head_dim
NK = 16           # linear_num_key_heads
NV = 32           # linear_num_value_heads
VPK = NV // NK    # 2  (value heads per key head)
KEYDIM = HK * NK  # 2048
VALDIM = HV * NV  # 4096


def tiled_to_grouped_index(n, nk, vpk, hd):
    """Index array that maps a tiled layout back to grouped layout.

    llama.cpp stores V-heads tiled as reshape(vpk, nk, hd).transpose(1,0,2);
    inverting means indexing with reshape(vpk,nk,hd).transpose(1,0,2).ravel().
    """
    return np.arange(n).reshape(vpk, nk, hd).transpose(1, 0, 2).reshape(n)


# Per-channel V permutation (length VALDIM) and per-head permutation (length NV).
V_ROW_PERM = tiled_to_grouped_index(VALDIM, NK, VPK, HV)
CONV_V_PERM = V_ROW_PERM
HEAD_PERM = tiled_to_grouped_index(NV, NK, VPK, 1)

reader = GGUFReader(SRC)
arch = None
for key, field in reader.fields.items():
    if key == "general.architecture":
        arch = field.contents()
writer = GGUFWriter(DST, arch if isinstance(arch, str) else "qwen35")

# ---- copy all KV metadata verbatim (except auto-managed keys) ----
SKIP_KEYS = {
    "GGUF.version",
    "GGUF.tensor_count",
    "GGUF.kv_count",
    # Already set via GGUFWriter(arch=...); re-adding here would trigger a
    # "Duplicated key name 'general.architecture'" warning.
    "general.architecture",
}
for key, field in reader.fields.items():
    if key in SKIP_KEYS:
        continue
    t0 = field.types[0]
    if t0 == GGUFValueType.ARRAY:
        sub = field.types[1]
        val = field.contents()
        if sub == GGUFValueType.STRING:
            val = [v if isinstance(v, str) else str(v) for v in val]
        writer.add_key_value(key, val, GGUFValueType.ARRAY, sub_type=sub)
    else:
        writer.add_key_value(key, field.contents(), t0)

n_transformed = 0
for t in reader.tensors:
    name = t.name
    data = t.data            # numpy view (byte rows for quantized tensors)
    qtype = t.tensor_type

    new_data = data
    raw_dtype = qtype

    # ---- (1) RMSNorm (w+1): subtract 1 (except ssm_norm = gated linear_attn.norm) ----
    if name.endswith("norm.weight") and not name.endswith("ssm_norm.weight"):
        new_data = (data.astype(np.float32) - 1.0).astype(np.float32)
        n_transformed += 1

    # ---- (2) ssm_a -> A_log = log(-x), then head reorder ----
    elif name.endswith(".ssm_a"):
        a = data.astype(np.float32)
        a = np.log(np.clip(-a, 1e-9, None))
        new_data = a[HEAD_PERM].astype(np.float32)
        n_transformed += 1

    # ---- ssm_dt.bias -> head reorder ----
    elif name.endswith(".ssm_dt.bias"):
        new_data = data.astype(np.float32)[HEAD_PERM].astype(np.float32)
        n_transformed += 1

    # ---- ssm_alpha / ssm_beta (in_proj_a/b): F32 [out=32, in], reorder rows ----
    elif name.endswith(".ssm_alpha.weight") or name.endswith(".ssm_beta.weight"):
        d = data.astype(np.float32)            # [32, hidden]
        new_data = d[HEAD_PERM].astype(np.float32)
        n_transformed += 1

    # ---- conv1d: F32 [conv_dim, kernel]; reorder V channels only ----
    elif name.endswith(".ssm_conv1d.weight"):
        d = data.astype(np.float32)            # [8192, 4]
        qk = KEYDIM * 2                         # 4096 (q+k channels)
        v = d[qk:]
        v = v[CONV_V_PERM]
        new_data = np.concatenate([d[:qk], v], axis=0).astype(np.float32)
        n_transformed += 1

    # ---- in_proj_qkv (attn_qkv): quantized [out=8192 rows, bytes]; reorder V rows ----
    elif name.endswith(".attn_qkv.weight"):
        qk = KEYDIM * 2                         # 4096
        if F16_INPROJ:
            w = gguf.dequantize(data, qtype).astype(np.float32)   # [8192, in]
            v = w[qk:][V_ROW_PERM]
            w = np.concatenate([w[:qk], v], axis=0).astype(np.float16)
            new_data = w
            raw_dtype = QT.F16
        else:
            v = data[qk:][V_ROW_PERM]
            new_data = np.concatenate([data[:qk], v], axis=0)
        n_transformed += 1

    # ---- in_proj_z (attn_gate): quantized [out=4096 rows, bytes]; reorder all rows ----
    elif name.endswith(".attn_gate.weight"):
        if F16_INPROJ:
            w = gguf.dequantize(data, qtype).astype(np.float32)   # [4096, in]
            new_data = w[V_ROW_PERM].astype(np.float16)
            raw_dtype = QT.F16
        else:
            new_data = data[V_ROW_PERM]
        n_transformed += 1

    # ---- out_proj (ssm_out): Q8_0, input-dim (column) reorder -> dequant/requant ----
    elif name.endswith(".ssm_out.weight"):
        # ne = [in=4096, out=4096]; dequantize -> [out, in], reorder columns (in)
        w = gguf.dequantize(data, qtype).astype(np.float32)   # [out, in]
        w = w[:, V_ROW_PERM]
        new_data = gguf.quantize(np.ascontiguousarray(w), QT.Q8_0)
        raw_dtype = QT.Q8_0
        n_transformed += 1

    new_data = np.ascontiguousarray(new_data)
    writer.add_tensor(name, new_data, raw_dtype=raw_dtype)

print(f"transformed {n_transformed} tensors; writing {DST} ...", flush=True)
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
print("DONE", flush=True)
