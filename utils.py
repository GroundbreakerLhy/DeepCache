import os
import re
import sys
import json
import subprocess
import numpy as np

from tqdm import tqdm
import pin_logger

from encoder.Encoder import get_loop_num_with_log


def list_to_json(dict_obj: list, output_path: str):
    j = json.dumps(dict_obj, sort_keys=True, indent=4)
    with open(output_path, "w") as f:
        f.write(j)


def dict_to_json(dict_obj: dict, output_path: str):
    j = json.dumps(dict_obj, sort_keys=True, indent=4)
    with open(output_path, "w") as f:
        f.write(j)


def json_to_list(json_path: str):
    if not os.path.exists(json_path):
        return list()
    with open(json_path, "r") as f:
        j_txt = f.read()
        list_obj = json.loads(s=j_txt)
        return list_obj


def json_to_dict(json_path: str):
    if not os.path.exists(json_path):
        return dict()
    with open(json_path, "r") as f:
        j_txt = f.read()
        dict_obj = json.loads(s=j_txt)
        return dict_obj


def get_func_range(func_asm_path: str):
    start_addr = ""
    end_addr = ""
    with open(func_asm_path, "r") as f:
        asm_txt = f.read()
        lines = asm_txt.split("\n")
        for line in lines:
            if line.startswith(";"):
                continue
            start_addr = line.split(":")[0]
            break
        lines.reverse()
        for line in lines:
            if line.startswith(";") or len(line) < 1:
                continue
            end_addr = line.split(":")[0]
            break
    return start_addr.lower(), end_addr.lower()


def generate_trace_for_all(models_dir="./compiled_models/tvm/"):
    # TVM
    def log_or_not(name_list, idx, size):
        if idx >= len(name_list) - 2:
            return False, ""

        name = name_list[idx].split(".")[1]
        if (
            "_compute_" in name
            and "conv2d" not in name
            and "dense" not in name
            and "max_pool" not in name
        ):
            return True, name_list[idx][:-4]

        if "_compute_" in name and "max_pool" in name and size > 0x50:
            return True, name_list[idx][:-4]

        if name.startswith("sub_") and not name_list[idx + 1].split(".")[1].startswith(
            "sub_"
        ):  # conv2d and dense
            label_idx = idx - 1
            while label_idx > 0:
                if not name_list[label_idx].split(".")[1].startswith("sub_"):
                    break
                label_idx -= 1
            return True, name_list[label_idx][:-4]
        return False, ""

    # Start the persistent attacker server
    pin_logger.start_server()

    models = os.listdir(models_dir)
    models.sort()

    model_bar = tqdm(models, desc="Total Progress")
    for model_name in model_bar:
        model_bar.set_description(f"Processing {model_name}")
        
        model_path = os.path.join(models_dir, model_name)
        if not os.path.isdir(model_path):
            continue

        # Locate build artifacts
        build_dir = os.path.join(model_path, "build")
        binary_path = os.path.join(build_dir, "demo_static")
        funcs_dir_path = os.path.join(build_dir, "demo_static_funcs")

        if not os.path.exists(binary_path) or not os.path.exists(funcs_dir_path):
            if os.path.isfile(model_path) and os.path.exists(model_path + "_funcs"):
                binary_path = model_path
                funcs_dir_path = model_path + "_funcs"
                dnn_input = "./examples/resnet18_tvm_O3/cat.bin"
            else:
                continue
        else:
            dnn_input = os.path.join(build_dir, "cat.bin")
            if not os.path.exists(dnn_input):
                dnn_input = "./examples/resnet18_tvm_O3/cat.bin"

        funcs = os.listdir(funcs_dir_path)
        funcs.sort()

        # Output trace dir keyed by model name
        trace_dir = "./cache_dataset/cache_dataset_llc_tvm/{}/".format(model_name)
        if not os.path.exists(trace_dir):
            os.makedirs("./cache_dataset/cache_dataset_llc_tvm/", exist_ok=True)

            status, output = pin_logger.cmd("mkdir -p {}".format(trace_dir))

        # Prepare list of ops to trace first
        ops_to_trace = []
        
        for idx in range(len(funcs)):
            func_path = os.path.join(funcs_dir_path, funcs[idx])
            start_addr, end_addr = get_func_range(func_path)
            if (int(end_addr, 16) - int(start_addr, 16)) < 0x50:
                continue

            should_log, label = log_or_not(
                funcs, idx, size=(int(end_addr, 16) - int(start_addr, 16))
            )

            if should_log:
                dnn_exe = binary_path
                ops_to_trace.append({
                    "dnn_exe": dnn_exe,
                    "dnn_input": dnn_input,
                    "start_addr": start_addr,
                    "end_addr": end_addr,
                    "label": label
                })

        # Execute: For each op, run 1 -> save, run 2 -> save
        for op in tqdm(ops_to_trace, desc=f"Ops in {model_name}", leave=False):
            for i in range(2):
                run_dir = os.path.join("./cache_dataset/cache_dataset_llc_tvm/", f"{model_name}-loop_{i + 1}")
                os.makedirs(run_dir, exist_ok=True)

                log_filename = "{}-{}-{}.log".format(op["label"], op["start_addr"], op["end_addr"])
                log_path = os.path.join(run_dir, log_filename)

                pin_logger.llc_log_server(
                    op["dnn_exe"],
                    op["dnn_input"],
                    op["start_addr"],
                    op["end_addr"],
                    clear=True,
                    log_path=log_path,
                )
    
    pin_logger.stop_server()


def generate_trace_for_glow(funcs_dir="./compiled_models/glow/"):
    def log_or_not(name_list, idx):
        if idx >= len(name_list):
            return False, ""

        name = name_list[idx].split(".")[1]
        if "libjit" in name and ("conv" in name or "fc" in name or "pool" in name):
            return True, name_list[idx][:-4]

        return False, ""

    files = os.listdir(funcs_dir)
    files.sort()
    for f in files:
        if ".lst" not in f and ".asm" not in f and f.endswith(".out"):
            f_path = os.path.join(funcs_dir, f)
            if not os.path.isfile(f_path):
                continue

            funcs_path = f_path + "_funcs"
            funcs = os.listdir(funcs_path)
            funcs.sort()

            trace_dir = "./cache_dataset/cache_dataset_glow/{}/".format(f)
            if not os.path.exists(trace_dir):
                status, output = pin_logger.cmd("mkdir {}".format(trace_dir))
            for idx in range(len(funcs)):
                func_path = os.path.join(funcs_path, funcs[idx])
                start_addr, end_addr = get_func_range(func_path)
                if (int(end_addr, 16) - int(start_addr, 16)) < 0x50:
                    continue

                should_log, label = log_or_not(funcs, idx)
                if should_log:
                    dnn_exe = f_path
                    dnn_input = "./examples/resnet18_tvm_O3/cat.bin"
                    log_path = os.path.join(
                        trace_dir, "{}-{}-{}.log".format(label, start_addr, end_addr)
                    )
                    # print(dnn_exe, dnn_input, start_addr, end_addr, log_path)
                    pin_logger.trace_log(
                        dnn_exe, dnn_input, start_addr, end_addr, log_path
                    )
                    # input("continue?")


def obfs_trace_example_glow(funcs_dir="./compiled_models/glow/"):
    def log_or_not(name_list, idx):
        if idx >= len(name_list):
            return False, ""

        name = name_list[idx].split(".")[1]
        if "libjit" in name and ("conv" in name or "fc" in name or "pool" in name):
            return True, name_list[idx][:-4]

        return False, ""

    files = os.listdir(funcs_dir)
    files.sort()
    for f in files:
        if ".lst" not in f and ".asm" not in f and f.endswith(".out"):
            f_path = os.path.join(funcs_dir, f)
            if not os.path.isfile(f_path):
                continue

            funcs_path = f_path + "_funcs"
            funcs = os.listdir(funcs_path)
            funcs.sort()

            trace_dir = "./mitigate/obfs_traces/{}/".format(f)
            if not os.path.exists(trace_dir):
                status, output = pin_logger.cmd("mkdir {}".format(trace_dir))
            for idx in range(len(funcs)):
                func_path = os.path.join(funcs_path, funcs[idx])
                start_addr, end_addr = get_func_range(func_path)
                if (int(end_addr, 16) - int(start_addr, 16)) < 0x50:
                    continue

                should_log, label = log_or_not(funcs, idx)
                if should_log:
                    dnn_exe = f_path
                    dnn_input = "./examples/resnet18_tvm_O3/cat.bin"
                    log_path = os.path.join(
                        trace_dir, "{}-{}-{}.log".format(label, start_addr, end_addr)
                    )
                    # print(dnn_exe, dnn_input, start_addr, end_addr, log_path)
                    pin_logger.obfus_log(
                        dnn_exe, dnn_input, start_addr, end_addr, log_path, 0x0
                    )


# ============================


def check_topk_labels(predict_labels, attr_labels, topk=1):
    """
    Check the database matching results according to the similarity of embeddings
    """
    with open(predict_labels, "r") as f:
        j_txt = f.read()
        pre_labels = json.loads(s=j_txt)
    with open(attr_labels, "r") as f:
        j_txt = f.read()
        attr_labels = json.loads(s=j_txt)

    correct_count = 0
    all_count = 0
    for name, output_list in pre_labels:
        # if "vgg16-7+tvmgen_default_fused_nn_contrib_conv2d_NCHWc_add_nn_relu_2" not in name:
        #     continue

        name = name.replace("_compute_", "")
        target = attr_labels[name]
        # print("target", target)
        output_list = output_list[:topk]

        exist = False
        for pre_name in output_list:
            pre_name = pre_name.replace("_compute_", "")
            predict = attr_labels[pre_name]
            # print("predict", predict)
            if target == predict:
                exist = True
                break
        if exist:
            correct_count += 1
        else:
            print(name, output_list)
            # input("debug")
        all_count += 1
    print("{}/{}".format(correct_count, all_count))


def get_loop_factor(trace_dir, model_name, layer_name, compiler="tvm"):
    # debuf
    # print("Identifying loop...", model_name, layer_name)
    dir_path = os.path.join(trace_dir, model_name)
    files = os.listdir(dir_path)
    for f in files:
        if f.endswith(".log") and layer_name + "-0x" in f:
            json_f = f.replace(".log", ".json")
            if json_f in files:
                json_f = os.path.join(dir_path, json_f)
                with open(json_f, "r") as dict_f:
                    seg_dict = json.load(dict_f)
                    seg_ids = json.loads(seg_dict["seg_ids"])
                    if len(seg_ids[0]) > 0:
                        tmp = [-1]
                        for ids in seg_ids:
                            if ids[0] > 0:
                                tmp = ids
                                break
                        return seg_dict["trace_len"], tmp[0]
                    else:  # empty
                        return seg_dict["trace_len"], -1
            else:
                # generate the json record, requiring training a small model
                log_path = os.path.join(dir_path, f)
                json_f = os.path.join(dir_path, json_f)
                factor_dict = testTraceSeg.get_loop_num_with_log(
                    log_path, prefix=model_name, compiler=compiler
                )
                print(json_f)
                # input("debug: {}".format(factor_dict))
                with open(json_f, "w") as dict_f:
                    json.dump(factor_dict, dict_f, indent=2)

                    seg_ids = json.loads(factor_dict["seg_ids"])
                    if len(seg_ids[0]) > 0:
                        tmp = [-1]
                        for ids in seg_ids:
                            if ids[0] > 0:
                                tmp = ids
                                break
                        return factor_dict["trace_len"], tmp[0]
                    else:  # empty
                        return factor_dict["trace_len"], -1


def get_log_path(trace_dir, model_name, layer_name, compiler):
    dir_path = os.path.join(trace_dir, model_name)
    if compiler == "glow" and ".out" not in model_name:
        dir_path = dir_path + ".out"
    if not os.path.exists(dir_path):
        return None
    files = os.listdir(dir_path)

    # First try to match the original format with memory address
    for f in files:
        if f.endswith(".log") and layer_name + "-0x" in f:
            f_path = os.path.join(dir_path, f)
            f_path = os.path.abspath(f_path)
            return f_path

    # If not found, try to match the new format without memory address
    for f in files:
        if f.endswith(".log") and layer_name + "_" in f:
            # Ensure it's an exact match, not a partial match
            if layer_name in f:
                f_path = os.path.join(dir_path, f)
                f_path = os.path.abspath(f_path)
                return f_path

    return None


# def get_log_path_llc(trace_dir, model_name, layer_name, compiler):
#     dir_path = os.path.join(trace_dir, model_name)
#     if compiler == 'glow':
#         dir_path = dir_path + '.out'
#     else:
#         dir_path = dir_path + '-loop'
#     files = os.listdir(dir_path)
#     for f in files:
#         if f.endswith(".log") and layer_name+"-0x" in f:
#             f_path = os.path.join(dir_path, f)
#             f_path = os.path.abspath(f_path)
#             return f_path


# def normalize(x, power=2):
#     norm = x.pow(power).sum(1, keepdim=True).pow(1./power)
#     out = x.div(norm)
from numpy.linalg import norm


def eq_factor(factor1, factor2):
    if (
        factor1[0] > 300
        and (abs(factor1[0] - factor2[0]) / min(factor1[0], factor2[0])) > 0.15
    ):  #  or abs(factor1[0]-factor2[0]) > 2000:
        return False
    # if (factor1[1] == -1 or factor2[1] == -1) and factor1[1] != factor2[1]:
    #     return False
    if factor1[1] == 1:
        factor1[1] = factor2[1]

    factor1 = np.array(factor1)
    factor2 = np.array(factor2)
    # print(factor1)
    # print(factor2)
    cos_sim = np.dot(factor1, factor2) / (norm(factor1) * norm(factor2))
    # print("debug cos_sim: {}".format(cos_sim))
    if cos_sim > 0.95:
        return True
    else:
        return False


def loop_factor_LLC(factor1, factor2):
    factor1 = np.array(factor1)
    factor2 = np.array(factor2)
    # print(factor1)
    # print(factor2)
    cos_sim = np.dot(factor1, factor2) / (norm(factor1) * norm(factor2))
    # print("debug cos_sim: {}".format(cos_sim))
    return cos_sim


def check_loop_factor(predict_labels, attr_labels, trace_dir, new_labels, topk=8):

    with open(predict_labels, "r") as f:
        j_txt = f.read()
        pre_labels = json.loads(s=j_txt)
    with open(attr_labels, "r") as f:
        j_txt = f.read()
        attr_labels = json.loads(s=j_txt)

    correct_count = 0
    all_count = 0
    # for each candidata, we extract the loop factor
    new_labels_list = []
    for name, output_list in pre_labels:
        model_name, layer_name = name.split("+")
        target_factor = get_loop_factor(trace_dir, model_name, layer_name)
        target_attr = attr_labels[name.replace("_compute_", "")][1]  # [0] -> func_name
        # input("debug: {} {}, continue?".format(target_factor, target_attr))
        print("target debug: {} {}".format(target_factor, target_attr))

        output_list = output_list[:topk]
        exist = False
        new_output_list = []
        for idx in range(len(output_list)):
            pre_name = output_list[idx]
            pre_model_name, pre_layer_name = pre_name.split("+")
            print("Top {}".format(idx))
            pre_factor = get_loop_factor(trace_dir, pre_model_name, pre_layer_name)
            # pre_attr = attr_labels[pre_name.replace("_compute_", "")][1]
            pre_attrs = get_attr_list(attr_labels, pre_name.replace("_compute_", ""))
            print("pre debug: {} {}".format(pre_factor, pre_attrs))
            new_output_list.append((pre_name, pre_factor, pre_attrs))

            if eq_factor(target_factor, pre_factor) and target_attr in pre_attrs:
                exist = True
                print("Success: match")  # debug
                break
            elif eq_factor(target_factor, pre_factor):
                exist = False
                print("Failed: not match")  # debug
                input("debug")
                break
        if exist:
            correct_count += 1
        else:
            print(name, output_list)
            # input("debug")
        all_count += 1
        new_labels_list.append([(name, target_factor, target_attr), new_output_list])
    with open(new_labels, "w") as f:
        json.dump(new_labels_list, f, indent=2)
    print("{}/{}".format(correct_count, all_count))


def check_loop_factor_glow(predict_labels, attr_labels, trace_dir, new_labels, topk=8):

    with open(predict_labels, "r") as f:
        j_txt = f.read()
        pre_labels = json.loads(s=j_txt)
    with open(attr_labels, "r") as f:
        j_txt = f.read()
        attr_labels = json.loads(s=j_txt)

    correct_count = 0
    all_count = 0
    # for each candidata, we extract the loop factor
    new_labels_list = []
    for name, output_list in pre_labels:
        model_name, layer_name = name.split("+")
        target_factor = get_loop_factor(
            trace_dir, model_name + ".out", layer_name, compiler="glow"
        )
        target_attr = attr_labels[name]
        # input("debug: {} {}, continue?".format(target_factor, target_attr))
        print("target debug: {} {}".format(target_factor, target_attr))

        output_list = output_list[:topk]
        exist = False
        new_output_list = []
        for idx in range(len(output_list)):
            # print("Top", idx)
            pre_name = output_list[idx]
            pre_model_name, pre_layer_name = pre_name.split("+")
            pre_factor = get_loop_factor(
                trace_dir, pre_model_name + ".out", pre_layer_name, compiler="glow"
            )
            pre_attr = attr_labels[pre_name]
            # print("pre debug: {} {}".format(pre_factor, pre_attr))
            new_output_list.append((pre_name, pre_factor, pre_attr))

            if eq_factor(target_factor, pre_factor) and target_attr == pre_attr:
                exist = True
                print("Success: match")  # debug
                break
            elif eq_factor(target_factor, pre_factor):
                exist = False
                print("Failed: not match")  # debug
                # input("debug")
                break
        if exist:
            correct_count += 1
        else:
            print(name, output_list)
            # input("debug")
        all_count += 1
        new_labels_list.append([(name, target_factor, target_attr), new_output_list])
        # input("debug")
    with open(new_labels, "w") as f:
        json.dump(new_labels_list, f, indent=2)
    print("{}/{}".format(correct_count, all_count))


def get_attr_list(attr_labels: dict, func_name):
    """
    Just noticed that operators (with different dimensions and layouts) may reuse the same binary function
    So for a single binary func, we should retrun a list of all possible attributes
    """
    # slow, could be refactored
    attr_list = []
    func_part = func_name.split('+', 1)[1] if '+' in func_name else func_name
    for name, attr in attr_labels.items():
        if func_name == name or func_part == attr[0]:  # attr[0] -> func_name
            if attr[1] not in attr_list:
                attr_list.append(attr[1])
    return attr_list


def check_loop_factor_red(
    predict_labels,
    attr_labels,
    trace_dir,
    new_labels,
    compiler="tvm",
    LLC=False,
    topk=100,
):
    """ground truth in attr_labels"""
    with open(predict_labels, "r") as f:
        j_txt = f.read()
        pre_labels = json.loads(s=j_txt)
    with open(attr_labels, "r") as f:
        j_txt = f.read()
        attr_labels = json.loads(s=j_txt)

    correct_count = 0
    all_count = 0
    # for each candidata, we extract the loop factor
    new_labels_list = []
    for name, output_list in pre_labels:
        model_name, layer_name = name.split("+")
        # target_factor = get_loop_factor(trace_dir, model_name, layer_name)
        log_path = get_log_path(trace_dir, model_name, layer_name, compiler)
        target_factor = get_loop_num_with_log(log_path, prefix=model_name)

        # name_key = name
        # if '-loop' in name:
        #     name_key = name.replace('-loop', '')
        name_key = dnn_name_filter(name)

        if compiler == "glow":
            target_attr_list = [attr_labels.get(name_key)] if attr_labels.get(name_key) else []
        else:
            lookup_key = name_key.replace("_compute", "") if name_key.endswith("_compute") else name_key.replace("_compute_", "")
            target_attr_entry = attr_labels.get(lookup_key)
            if target_attr_entry:
                target_attr_list = [target_attr_entry[1]]
            else:
                target_attr_list = get_attr_list(attr_labels, lookup_key)

        if not target_attr_list:
            print(f"Warning: could not find attributes for {name} (key: {name_key})")
            continue
        print("target debug: {} {} {}".format(name, target_factor, target_attr_list))

        output_list = output_list[:topk]
        exist = False
        new_output_list = []
        for idx in range(min(len(output_list), 35)):
            pre_name = output_list[idx][0]  # pre_name = output_list[idx]
            pre_sim_val = output_list[idx][1]
            pre_model_name, pre_layer_name = pre_name.split("+")
            print("Top {}".format(idx + 1))

            # pre_factor = get_loop_factor(trace_dir, pre_model_name, pre_layer_name)
            log_path = get_log_path(trace_dir, pre_model_name, pre_layer_name, compiler)
            pre_factor = get_loop_num_with_log(log_path, prefix=pre_model_name)

            pre_name_key = dnn_name_filter(pre_name)

            if compiler == "glow":
                pre_attrs = attr_labels[pre_name_key]
            else:
                pre_attrs = get_attr_list(
                    attr_labels, pre_name_key.replace("_compute_", "")
                )

            print("pre debug: {} {} {}".format(pre_name, pre_factor, pre_attrs))
            new_output_list.append((pre_name, pre_sim_val, pre_factor, pre_attrs))

            eq_flag = eq_factor(target_factor, pre_factor)
            if eq_flag:
                if (compiler == "tvm" and any(ta in pre_attrs for ta in target_attr_list)) or (
                    compiler == "glow" and any(ta == pre_attrs for ta in target_attr_list)
                ):
                    exist = True
                    print("Success: match")  # debug
                    break
                elif any(attr_fuzzy_match(ta, pre_attrs, compiler) for ta in target_attr_list):
                    exist = True
                    print("Success: match")  # debug
                    break
                else:
                    exist = False
                    print("Failed: not match")  # debug
                    break
        if exist:
            correct_count += 1
        all_count += 1
        new_labels_list.append([(name, target_factor, target_attr_list), new_output_list])
    with open(new_labels, "w") as f:
        json.dump(new_labels_list, f, indent=2)
    print("{}/{}".format(correct_count, all_count))


def check_loop_factor_red_LLC(
    predict_labels,
    attr_labels,
    trace_dir,
    new_labels,
    compiler="tvm",
    topk=30,
    len_thre=0.15,
):
    """ground truth in attr_labels"""
    with open(predict_labels, "r") as f:
        j_txt = f.read()
        pre_labels = json.loads(s=j_txt)
    with open(attr_labels, "r") as f:
        j_txt = f.read()
        attr_labels = json.loads(s=j_txt)

    correct_count = 0
    all_count = 0
    # for each candidata, we extract the loop factor
    new_labels_list = []
    for name, output_list in pre_labels:
        model_name, layer_name = name.split("+")
        # target_factor = get_loop_factor(trace_dir, model_name, layer_name)
        log_path = get_log_path(trace_dir, model_name, layer_name, compiler)
        if log_path is None:
            print(f"Warning: log file not found for {name}")
            continue
        target_len = get_log_length(log_path)
        target_factor = get_loop_num_with_log(log_path, prefix=model_name)

        name_key = dnn_name_filter(name)

        if compiler == "glow":
            target_attr_list = [attr_labels.get(name_key)] if attr_labels.get(name_key) else []
        else:
            lookup_key = name_key.replace("_compute", "") if name_key.endswith("_compute") else name_key.replace("_compute_", "")
            target_attr_entry = attr_labels.get(lookup_key)
            if target_attr_entry:
                target_attr_list = [target_attr_entry[1]]
            else:
                # Fallback: multiple graph nodes may share one binary function
                target_attr_list = get_attr_list(attr_labels, lookup_key)

        if not target_attr_list:
            print(f"Warning: could not find attributes for {name} (key: {name_key})")
            continue
        # print("target debug: {} {} {}".format(name, target_factor, target_attr_list))

        # calculate the final score
        output_list = output_list[:topk]
        new_output_list = []
        for idx in range(len(output_list)):
            item = output_list[idx]
            pre_name = item[0]
            pre_sim_val = item[1]
            pre_model_name, pre_layer_name = pre_name.split("+")

            pre_log_path = get_log_path(
                trace_dir, pre_model_name, pre_layer_name, compiler
            )
            pre_len = get_log_length(pre_log_path)

            # if (
            #     abs(pre_len - target_len) / target_len
            # ) > len_thre:  # if (abs(pre_len - target_len) / min(pre_len, target_len)) > len_thre:
            #     continue

            pre_name_key = dnn_name_filter(pre_name)
            if compiler == "glow":
                pre_attrs = attr_labels[pre_name_key]
            else:
                # pre_attr = attr_labels[pre_name.replace("_compute_", "")][1]
                # Remove _compute suffix (handle both _compute_ and _compute)
                lookup_key = pre_name_key
                if lookup_key.endswith("_compute_"):
                    lookup_key = lookup_key[:-9]  # remove "_compute_"
                elif lookup_key.endswith("_compute"):
                    lookup_key = lookup_key[:-8]  # remove "_compute"
                pre_attrs = get_attr_list(attr_labels, lookup_key)
            pre_factor = get_loop_num_with_log(pre_log_path, prefix=pre_model_name)

            # print("pre debug: {} {} {}".format(pre_name, pre_factor, pre_attrs))
            new_output_list.append(
                [
                    pre_name,
                    pre_sim_val,
                    pre_len,
                    pre_sim_val + 5 * loop_factor_LLC(target_factor, pre_factor),
                    pre_factor,
                    pre_attrs,
                ]
            )

            if len(new_output_list) >= 5:
                break

        new_output_list.sort(key=lambda x: x[3])
        new_output_list.reverse()

        # check if match
        print(f"{model_name}")
        print(f"  {layer_name}")
        print(f"    Target attr: {target_attr_list}")

        exist = False
        if len(new_output_list) > 0:
            for rank, candidate in enumerate(new_output_list, 1):
                cand_name, cand_sim, cand_len, cand_score, cand_factor, cand_attrs = candidate
                cand_model, cand_layer = cand_name.split("+")
                loop_sim = loop_factor_LLC(target_factor, cand_factor)
                print(f"    Top{rank}: model={cand_model}, layer={cand_layer}")
                print(f"          embedding_sim={cand_sim:.4f}, loop_factor_sim={loop_sim:.4f}, score={cand_score:.4f}")
                print(f"          attr={cand_attrs}")

                if (compiler == "tvm" and any(ta in cand_attrs for ta in target_attr_list)) or (
                    compiler == "glow" and any(ta == cand_attrs for ta in target_attr_list)
                ):
                    exist = True
                    print(f"    Match: {candidate[0]} (exact, rank {rank})\n")
                    break
                elif any(attr_fuzzy_match(ta, cand_attrs, compiler) for ta in target_attr_list):
                    exist = True
                    print(f"    Match: {candidate[0]} (fuzzy, rank {rank})")
                    break
            if not exist:
                print(f"    Result: FAILED\n")
        else:
            print(f"    Result: FAILED - No candidates\n")

        if exist:
            correct_count += 1
        all_count += 1
        new_labels_list.append([(name, target_factor, target_attr_list), new_output_list])
    with open(new_labels, "w") as f:
        json.dump(new_labels_list, f, indent=2)
    print("Top 5 accuracy: {}/{}".format(correct_count, all_count))


def attr_fuzzy_match(target_attr, pre_attrs, compiler=""):
    import math

    if compiler == "tvm":
        for pre in pre_attrs:
            if len(target_attr) != len(pre):
                continue
            match_flag = True
            for i in range(len(target_attr)):
                threshold = math.ceil(target_attr[i] * 0.1)

                if abs(target_attr[i] - pre[i]) > threshold:
                    match_flag = False
                    break
            if match_flag:
                return True

    elif compiler == "glow":
        if len(target_attr) != len(pre_attrs):
            return False
        for i in range(len(target_attr)):
            threshold = math.ceil(target_attr[i] * 0.1)

            if abs(target_attr[i] - pre_attrs[i]) > threshold:
                return False
        return True

    return False


def check_topk_attrs(
    predict_labels, attr_labels, trace_dir, compiler="glow", topk=30, len_thre=0.15
):
    """ground truth in attr_labels"""
    with open(predict_labels, "r") as f:
        j_txt = f.read()
        pre_labels = json.loads(s=j_txt)
    with open(attr_labels, "r") as f:
        j_txt = f.read()
        attr_labels = json.loads(s=j_txt)

    overall_count = 0
    success_count = 0
    for name, output_list in pre_labels:
        model_name, layer_name = name.split("+")
        # if '-loop' in name:
        #     name = name.replace('-loop', '')
        name = dnn_name_filter(name)

        if compiler == "glow":
            target_attr = attr_labels[name]
        else:
            target_attr = attr_labels[name.replace("_compute_", "")][
                1
            ]  # [0] -> func_name
        # input("debug: {} {}, continue?".format(target_factor, target_attr))

        # debug the log length
        # model_name, layer_name = name.split("+")
        log_path = get_log_path(trace_dir, model_name, layer_name, compiler)
        target_len = get_log_length(log_path)
        print("\n\ntarget debug: {} {} {}".format(name, target_len, target_attr))

        overall_count += 1
        output_list = output_list[:topk]
        exist = False

        for idx in range(min(len(output_list), 35)):
            pre_name = output_list[idx][0]  # pre_name = output_list[idx]
            pre_model_name, pre_layer_name = pre_name.split("+")
            # if '-loop' in pre_name:
            #     pre_name = pre_name.replace('-loop', '')
            pre_name = dnn_name_filter(pre_name)

            pre_sim_val = output_list[idx][1]
            # pre_model_name, pre_layer_name = pre_name.split("+")
            pre_log_path = get_log_path(
                trace_dir, pre_model_name, pre_layer_name, compiler
            )
            pre_len = get_log_length(pre_log_path)

            if compiler == "glow":
                pre_attrs = attr_labels[pre_name]
            else:
                # pre_attr = attr_labels[pre_name.replace("_compute_", "")][1]
                pre_attrs = get_attr_list(
                    attr_labels, pre_name.replace("_compute_", "")
                )

            # len_diff =  (abs(pre_len - target_len) / min(pre_len, target_len))
            if target_len == 0:
                continue
            len_diff = abs(pre_len - target_len) / target_len

            if len_diff <= len_thre and (
                (pre_attrs == target_attr) or (target_attr in pre_attrs)
            ):
                print("Top {}".format(idx + 1))
                print(
                    "pre debug: {} {} {} {}".format(
                        pre_name, pre_sim_val, pre_len, pre_attrs
                    )
                )
                success_count += 1
                break
            elif len_diff <= len_thre:  # else:
                # pass
                print("Top {}".format(idx + 1))  # debug
                print(
                    "pre debug: {} {} {} {}".format(
                        pre_name, pre_sim_val, pre_len, pre_attrs
                    )
                )
    print("succeess / overall: {} / {}".format(success_count, overall_count))


def dnn_name_filter(dnn_exe_name: str):
    name_key = dnn_exe_name

    # for tvm
    mat = re.search("-loop_\d+", dnn_exe_name)
    if mat:
        name_key = name_key.replace(mat.group(), "")
    elif "-loop" in dnn_exe_name:
        name_key = name_key.replace("-loop", "")

    # for glow
    mat = re.search("\.out(_\d+)?", dnn_exe_name)
    if mat:
        name_key = dnn_exe_name.replace(mat.group(), "")
    return name_key


def get_log_length(log_path: str):
    status, output = subprocess.getstatusoutput("wc -l {}".format(log_path))
    l = output.split(" ")[0]
    return int(l)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        obfs_trace_example_glow()
        exit(0)

    if sys.argv[1] == "oram":
        check_topk_attrs(
            "./embedding/distinct_labels_tvm_oram.json",
            "./attr_labels_tvm.json",
            "./mitigate/oram_traces/",
            compiler="tvm",
            topk=10,
            len_thre=0.15,
        )

    if sys.argv[1] == "obfs":
        check_topk_attrs(
            "./embedding/distinct_labels_glow_cache.json",
            "./attr_labels_glow.json",
            "./cache_dataset/cache_dataset_glow",
            compiler="glow",
            topk=35,
            len_thre=0.1,
        )

    # check_topk_attrs("./embedding/distinct_labels_tvm_cache.json",
    #                  "./attr_labels_tvm.json",
    #                  "./cache_dataset/cache_dataset_tvm/",
    #                   compiler="tvm",
    #                   topk=30, len_thre=0.15)

    # check_topk_attrs("./embedding/distinct_labels_glow_cache.json",
    #                  "./attr_labels_glow.json",
    #                   "./cache_dataset/cache_dataset_glow",
    #                   compiler="glow",
    #                   topk=35, len_thre=0.1)

    # === LLC ===
    # check_topk_attrs("./embedding/distinct_labels_tvm_cache_llc.json",
    #                  "./attr_labels_tvm.json",
    #                   "./cache_dataset/LLC_dataset_tvm",
    #                   compiler="tvm",
    #                   topk=20)
    # check_topk_attrs("./embedding/distinct_labels_glow_cache_llc.json",
    #                  "./attr_labels_glow.json",
    #                   "./cache_dataset/LLC_dataset_glow",
    #                   compiler="glow",
    #                   topk=20)
    exit(0)

    check_topk_attrs(
        "./embedding/distinct_labels_glow_cache.json", "./attr_labels_glow.json"
    )
    exit(0)

    # print(eq_factor([74823, 4815], [70554, 1]))
    # print(eq_factor([74823, 4815], [70554, -1]))

    generate_trace_for_glow()
    # generate_trace_for_all()
    exit(0)

    # check_loop_factor("./embedding/distinct_labels.json", "./attr_labels_tvm.json",
    #                   trace_dir="./cache_dataset/cache_dataset_tvm/", new_labels="./labels.json", topk=8)
    check_loop_factor_glow(
        "./embedding/distinct_labels_glow.json",
        "./attr_labels_glow.json",
        trace_dir="./cache_dataset/cache_dataset_glow/",
        new_labels="./labels_glow.json",
        topk=100,
    )
