import onnx
import tvm
import tvm.relay as relay
import numpy as np
import os
import sys

model_path = './vgg16-7.onnx'

def get_input_info(onnx_model):
    """Extract all inputs (not weights) from ONNX model and return shape_dict."""
    graph = onnx_model.graph
    initializer_names = set(i.name for i in graph.initializer)
    shape_dict = {}

    for inp in graph.input:
        if inp.name not in initializer_names:
            # ONNX TensorProto.DataType: 1=FLOAT, 7=INT64
            input_dtype = inp.type.tensor_type.elem_type
            is_nlp = (input_dtype == 7)

            shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            resolved = []
            for s in shape:
                if s > 0:
                    resolved.append(s)
                elif is_nlp and len(resolved) == 0:
                    resolved.append(1)    # batch_size = 1
                elif is_nlp and len(resolved) == 1:
                    resolved.append(128)  # seq_len = 128
                else:
                    resolved.append(1)
            shape_dict[inp.name] = tuple(resolved)

    if not shape_dict:
        raise ValueError("Could not find input tensor in ONNX model")

    return shape_dict

def compile():
    if not os.path.exists(model_path):
        print(f"ERROR: Model {model_path} not found")
        return False

    print(f"Loading {model_path}...")
    onnx_model = onnx.load(model_path)

    shape_dict = get_input_info(onnx_model)
    print(f"Detected inputs: {shape_dict}")

    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

    print(f"Compiling with TVM (opt_level=3)...")
    
    target = tvm.target.Target("llvm -mtriple=x86_64-linux-gnu -mcpu=skylake")
    
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)

    build_dir = "build"
    if not os.path.exists(build_dir):
        os.makedirs(build_dir)

    # Export artifacts
    lib_path = os.path.join(build_dir, "model.so")
    lib.export_library(lib_path)
    print(f"Exported library to {lib_path}")
    
    graph_json_path = os.path.join(build_dir, "graph_c.json")
    with open(graph_json_path, "w") as f:
        f.write(lib.get_graph_json())
    
    params_path = os.path.join(build_dir, "model.params")
    with open(params_path, "wb") as f:
        f.write(relay.save_param_dict(lib.get_params()))
        
    print("Compilation finished successfully.")
    return True

if __name__ == "__main__":
    success = compile()
    sys.exit(0 if success else 1)