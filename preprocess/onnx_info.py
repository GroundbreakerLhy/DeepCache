import os
import onnx
import json
import subprocess
import sys
import shutil


class cd:
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


def cmd(commandline, under_dir=""):
    if len(under_dir) == 0:
        under_dir = project_dir
    with cd(under_dir):
        # print(commandline)
        status, output = subprocess.getstatusoutput(commandline)
        return status, output


def run(prog_path, under_dir=""):
    if len(under_dir) == 0:
        under_dir = project_dir
    with cd(under_dir):
        proc = subprocess.Popen(
            prog_path, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = proc.communicate()
        return stdout, stderr


project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(project_dir)


def get_input_name(onnx_path: str):
    model = onnx.load(onnx_path)
    output = [node for node in model.graph.output]

    input_all = model.graph.input
    input_initializer = set(node.name for node in model.graph.initializer)

    net_feed_input = []
    for v in input_all:
        if v.name not in input_initializer:
            net_feed_input.append(v)

    if not net_feed_input:
        return None, None, None, None, False

    # 选择最大的输入（元素数量最多）作为主要输入
    # 这对于多输入模型（如 BERT）很重要
    def get_total_elements(inp):
        shape = [d.dim_value if d.dim_value > 0 else 1 for d in inp.type.tensor_type.shape.dim]
        total = 1
        for s in shape:
            total *= s
        return total

    main_input = max(net_feed_input, key=get_total_elements)

    # 获取主要输入的数据类型
    input_dtype = main_input.type.tensor_type.elem_type
    # ONNX TensorProto.DataType: 1=FLOAT, 7=INT64, 8=UINT8
    # 只支持 FLOAT 和 INT64
    if input_dtype == 8:  # UINT8 - not supported by TVM
        print(f"[Skip] Model uses UINT8 input, not supported by TVM")
        return None, None, None, None, False

    is_nlp_model = input_dtype == 7  # INT64 indicates NLP/LLM model

    # 获取输入维度（主要输入，用于生成 demo_static.c）
    dim_list = []
    for dim_value in main_input.type.tensor_type.shape.dim:
        val = dim_value.dim_value
        # NLP 模型的动态维度使用更合理的默认值
        if val <= 0:
            if is_nlp_model and len(dim_list) == 0:
                dim_list.append(1)  # batch_size = 1
            elif is_nlp_model and len(dim_list) == 1:
                dim_list.append(128)  # seq_len = 128
            else:
                dim_list.append(1)
        else:
            dim_list.append(val)

    # 获取输出维度
    dim_list2 = []
    if output:
        for dim_value in output[0].type.tensor_type.shape.dim:
            val = dim_value.dim_value
            dim_list2.append(1 if val <= 0 else val)
    else:
        # 有些模型没有显式Output定义，给个默认值
        dim_list2 = [1, 1000]

    return (
        main_input.name,
        dim_list,
        output[0].name if output else "output",
        dim_list2,
        is_nlp_model,
    )


def compile_all_onnx(recompile=False):
    onnx_dir = os.path.expanduser("~/onnx_zoo/") 
    target_base = "./compiled_models/tvm/"
    
    # [关键] 定位生成函数的脚本
    gen_funcs_script = os.path.join(project_dir, "preprocess/generate_funcs.py")
    
    if not os.path.exists(target_base):
        os.makedirs(target_base)

    files = os.listdir(onnx_dir)
    files.sort()
    for f in files:
        if f.endswith(".onnx") and "mnist" not in f:
            model_name = os.path.splitext(f)[0]
            f_path = os.path.join(os.path.abspath(onnx_dir), f)
            model_dir = os.path.join(target_base, model_name)

            # Skip already compiled models if recompile=False
            if not recompile and os.path.exists(model_dir):
                exe_path = os.path.join(model_dir, "build/demo_static")
                if os.path.exists(exe_path):
                    print(f"[Skip Compile] {model_name} (already compiled)")
                    continue

            # 解析模型信息
            input_name, input_shape, output_name, output_shape, is_nlp_model = get_input_name(f_path)
            if input_name is None:
                continue

            # 1. 准备目录：复制模板
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)

            shutil.copytree("./template/", model_dir)

            run("make clean", model_dir)

            # 2. 修改 build.py
            script_path = os.path.join(model_dir, "build.py")
            if not os.path.exists(script_path):
                script_path = os.path.join(model_dir, "build_model.py")

            with open(script_path, "r") as fr:
                txt = fr.read()

            # 替换模型路径
            if "vgg16-7.onnx" in txt:
                txt = txt.replace(
                    "model_path = './vgg16-7.onnx'", f"model_path = '{f_path}'"
                )
            elif "alexnet.onnx" in txt:
                txt = txt.replace(
                    "model_path = './alexnet.onnx'", f"model_path = '{f_path}'"
                )
            else:
                print(
                    f"Warning: Could not find model_path placeholder in {script_path}"
                )

            with open(script_path, "w") as fw:
                fw.write(txt)

            # 3. 修改 demo_static.c
            main_path = os.path.join(model_dir, "demo_static.c")
            with open(main_path, "r") as fr:
                txt = fr.read()

            # 替换输入名
            txt = txt.replace(
                'tvm_runtime_set_input(handle, "data_0", &input);',
                f'tvm_runtime_set_input(handle, "{input_name}", &input);',
            )

            # 替换维度
            # 查找: input.ndim = 4;
            txt = txt.replace("input.ndim = 4;", f"input.ndim = {len(input_shape)};")

            # 替换 Shape 数组
            # 查找: int64_t shape[4] = {1, 3, 224, 224};
            shape_str = ", ".join(map(str, input_shape))
            txt = txt.replace(
                "int64_t shape[4] = {1, 3, 224, 224};",
                f"int64_t shape[{len(input_shape)}] = {{{shape_str}}};",
            )

            with open(main_path, "w") as fw:
                fw.write(txt)

            # 3.5 如果是 NLP 模型，修改 Makefile 添加 -DNLP_MODEL 标志
            if is_nlp_model:
                makefile_path = os.path.join(model_dir, "Makefile")
                with open(makefile_path, "r") as fr:
                    makefile_txt = fr.read()
                # NLP dense 算子执行快，需要更多推理次数来采集足够的 cache trace
                makefile_txt = makefile_txt.replace(
                    "CFLAGS = -O3 -no-pie -g",
                    "CFLAGS = -O3 -no-pie -g -DNLP_MODEL -DNUM_RUNS=500"
                )
                with open(makefile_path, "w") as fw:
                    fw.write(makefile_txt)
                print(f"[NLP Model] {model_name} - added -DNLP_MODEL -DNUM_RUNS=500")

            # 4. 执行编译
            out, err = run("make", model_dir)

            # 5. 验证与生成汇编文件
            exe_path = os.path.join(model_dir, "build/demo_static")
            so_path = os.path.join(model_dir, "build/model.so")
            
            if os.path.exists(exe_path):
                # 5.1 验证运行
                target_input = os.path.abspath(os.path.join(project_dir, "experiment_llc/cat.bin"))
                if not os.path.exists(target_input):
                    os.makedirs(os.path.dirname(target_input), exist_ok=True)
                    subprocess.run(f"dd if=/dev/urandom of={target_input} bs=1M count=1 status=none", shell=True)

                out, err = run(f"./build/demo_static {target_input}", model_dir)
                
                # 5.2 生成汇编切片
                if os.path.exists(so_path) and os.path.exists(gen_funcs_script):
                    funcs_output_dir = os.path.join(model_dir, "build/model.so_funcs")
                    cmd(f"python3 {gen_funcs_script} {so_path} {funcs_output_dir}")

                print(f"SUCCESS: {model_name}")

            else:
                print(f"FAILED: {model_name}")
                print(">>> Error Log:")
                if err: print(err.decode())
                print("------------------------------------------------")


def get_labels_new():
    """
    生成训练标签，包括所有模型
    """
    overall_labels = {}
    exe_dir = "./compiled_models/tvm/"
    files = os.listdir(exe_dir)
    files.sort()

    count = 0

    for d in files:
        if d.startswith(".") or "template" in d:
            continue

        d_path = os.path.join(exe_dir, d)
        if os.path.isdir(d_path):
            json_path = os.path.join(d_path, "build/graph_c.json")

            # Skip models that failed to compile
            if not os.path.exists(json_path):
                print(f"[Skip Label] {d} (compilation failed or incomplete)")
                continue

            with open(json_path) as f:
                tmp_dict = json.load(f)
            nodes = tmp_dict["nodes"]
            # TVM Graph JSON 中 attrs["shape"] 的结构通常是 [type_idx, [shape_list]]
            if "attrs" in tmp_dict and "shape" in tmp_dict["attrs"]:
                shapes = tmp_dict["attrs"]["shape"][1]

            for i in range(len(nodes)):
                node_name = nodes[i]["name"]
                shape_label = None

                if "tvmgen" in nodes[i]["name"] and "conv" in nodes[i]["name"]:
                    # find the weights index
                    input_ids = nodes[i]["inputs"]
                    for id in input_ids:
                        id = id[0]
                        if len(shapes[id]) == 6:
                            shape_label = shapes[id]
                            break
                elif "tvmgen" in nodes[i]["name"] and "dense" in nodes[i]["name"]:
                    # find the weights index
                    input_ids = nodes[i]["inputs"]
                    for id in input_ids:
                        id = id[0]
                        if len(shapes[id]) == 3:
                            shape_label = shapes[id]
                elif "tvmgen" in nodes[i]["name"]:
                    shape_label = shapes[i]

                if shape_label:
                    func_name = node_name
                    if "attrs" in nodes[i] and "func_name" in nodes[i]["attrs"]:
                        func_name = nodes[i]["attrs"]["func_name"]

                    unique_name = "{}+{}".format(d, node_name)
                    overall_labels[unique_name] = (func_name, shape_label)
                    count += 1

    print(f"\nLabel Generation Summary:")
    print(f"  - Total operators found: {count}")

    return overall_labels


if __name__ == "__main__":
    # 1. 编译所有模型 (包括 Victim，用于攻击测试)
    print("=== Step 1: Compiling Models ===")
    compile_all_onnx(recompile=False)

    # 2. 生成标签 (自动排除 Victim，用于训练)
    print("\n=== Step 2: Generating Labels ===")
    labels = get_labels_new()

    output_file = "attr_labels_tvm.json"
    with open(output_file, "w") as f:
        json.dump(labels, f, sort_keys=True, indent=2)

    print(f"Done. Labels saved to {output_file}")
