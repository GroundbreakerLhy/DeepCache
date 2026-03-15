"""
Export HuggingFace transformer models to ONNX format for the DeepCache pipeline.

Uses optimum to export decoder-only LLMs as static-shape ONNX models
compatible with TVM 0.11.1 relay compilation.

Small models (<2GB ONNX) → flat file:  ~/onnx_zoo/<name>.onnx
Large models (>=2GB ONNX) → subdir:    ~/onnx_zoo/<name>/model.onnx
                                                          model.onnx_data
The compile_all_onnx pipeline scans both layouts automatically.

Models exported:
  Step 1 - GPT-2 variants: gpt2-medium, opt-125m, opt-350m  (already done)
  Step 3 - Modern LLMs with RoPE/RMSNorm/GQA:
    - Qwen/Qwen2.5-0.5B
    - Qwen/Qwen3.5-0.6B
"""

import os
import shutil
import subprocess
import sys

TARGET_DIR = os.path.expanduser("~/onnx_zoo/")
OPSET = 14  # scaled_dot_product_attention requires opset >= 14

# (hf_model_id, output_name, task)
# output_name: if ends with .onnx → flat file; otherwise → subdir
MODELS = [
    # Step 1 (already done, kept for reference)
    # ("openai-community/gpt2-medium", "gpt2-medium.onnx", "text-generation"),
    # ("facebook/opt-125m",            "opt-125m.onnx",    "text-generation"),
    # ("facebook/opt-350m",            "opt-350m.onnx",    "text-generation"),

    # Step 3: Modern LLMs - will use subdir layout due to size
    ("Qwen/Qwen2.5-0.5B",                         "qwen2.5-0.5b",          "text-generation"),
    ("Qwen/Qwen3.5-0.8B",                         "qwen3.5-0.8b",          "text-generation"),
]


def export_model(hf_id: str, out_name: str, task: str) -> bool:
    is_flat = out_name.endswith(".onnx")
    target_path = os.path.join(TARGET_DIR, out_name)

    # Check already done
    if is_flat and os.path.exists(target_path):
        print(f"[Skip] {out_name} already exists")
        return True
    if not is_flat:
        final_onnx = os.path.join(target_path, "model.onnx")
        if os.path.exists(final_onnx):
            print(f"[Skip] {out_name}/model.onnx already exists")
            return True

    tmp_dir = f"/tmp/onnx_export_{out_name.replace('.onnx', '')}"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    print(f"\n[Export] {hf_id} -> {out_name}")

    cmd = [
        "optimum-cli", "export", "onnx",
        "--model", hf_id,
        "--task", task,
        "--monolith",
        "--no-dynamic-axes",
        "--batch_size", "1",
        "--sequence_length", "128",
        "--opset", str(OPSET),
        tmp_dir,
    ]

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"[FAILED] Export failed for {hf_id}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return False

    # Find the main ONNX file
    candidates = ["model.onnx", "decoder_model.onnx", "decoder_model_merged.onnx"]
    src_onnx = None
    for candidate in candidates:
        p = os.path.join(tmp_dir, candidate)
        if os.path.exists(p):
            src_onnx = p
            break

    if src_onnx is None:
        print(f"[FAILED] Could not find ONNX output. Found: {os.listdir(tmp_dir)}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return False

    # Check for external data file
    data_file = src_onnx + "_data"
    has_external = os.path.exists(data_file)

    if is_flat and has_external:
        # Model too large for flat layout, force subdir
        print(f"  [Info] External data detected, switching to subdir layout")
        is_flat = False
        out_name = out_name.replace(".onnx", "")
        target_path = os.path.join(TARGET_DIR, out_name)

    if is_flat:
        shutil.move(src_onnx, target_path)
        size_mb = os.path.getsize(target_path) / 1024 / 1024
        print(f"[OK] {out_name} ({size_mb:.0f} MB)")
    else:
        # Subdir layout: move all files from tmp_dir to target_path
        if os.path.exists(target_path):
            shutil.rmtree(target_path)
        os.makedirs(target_path)
        for fname in os.listdir(tmp_dir):
            if fname.endswith(".onnx") or fname.endswith(".onnx_data"):
                shutil.move(os.path.join(tmp_dir, fname),
                            os.path.join(target_path, fname))
        total_mb = sum(
            os.path.getsize(os.path.join(target_path, f)) / 1024 / 1024
            for f in os.listdir(target_path)
        )
        print(f"[OK] {out_name}/ ({total_mb:.0f} MB total)")

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return True


def verify_tvm_import(onnx_path: str) -> bool:
    """Quick check that TVM relay can parse the model."""
    import onnx as onnx_lib
    import tvm
    import tvm.relay as relay

    print(f"  Verifying TVM import for {os.path.basename(onnx_path)}...")
    model = onnx_lib.load(onnx_path)
    graph = model.graph

    initializer_names = {i.name for i in graph.initializer}
    shape_dict = {}
    for inp in graph.input:
        if inp.name not in initializer_names:
            dtype = inp.type.tensor_type.elem_type
            is_nlp = (dtype == 7)  # INT64
            dims = []
            for d in inp.type.tensor_type.shape.dim:
                v = d.dim_value
                if v > 0:
                    dims.append(v)
                elif is_nlp and len(dims) == 0:
                    dims.append(1)
                elif is_nlp and len(dims) == 1:
                    dims.append(128)
                else:
                    dims.append(1)
            shape_dict[inp.name] = tuple(dims)

    mod, _ = relay.frontend.from_onnx(model, shape_dict)
    print(f"  [TVM OK] Relay import succeeded. Inputs: {shape_dict}")
    return True


if __name__ == "__main__":
    os.makedirs(TARGET_DIR, exist_ok=True)

    # Parse --verify flag
    do_verify = "--verify" in sys.argv

    success_list = []
    fail_list = []

    for hf_id, filename, task in MODELS:
        ok = export_model(hf_id, filename, task)
        if ok:
            success_list.append(filename)
            if do_verify:
                onnx_path = os.path.join(TARGET_DIR, filename)
                verify_tvm_import(onnx_path)
        else:
            fail_list.append(filename)

    print("\n=== Export Summary ===")
    for f in success_list:
        print(f"  OK  : {f}")
    for f in fail_list:
        print(f"  FAIL: {f}")

    print(f"\nDisk usage:")
    os.system(f"du -sh {TARGET_DIR}")
