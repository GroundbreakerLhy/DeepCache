#!/usr/bin/python3
import os
import re
import sys
import time
import signal
import subprocess

import numpy as np
from scipy.stats import entropy
import statistics


class cd:
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


def cmd(commandline, debug_flag=False):
    with cd(project_dir):
        if debug_flag:
            print(commandline)
        status, output = subprocess.getstatusoutput(commandline)
        # print(output)
        return status, output


def run(prog_path):
    with cd(project_dir):
        # print(prog_path)
        proc = subprocess.Popen(prog_path, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate()  # stderr: summary, stdout:  each statement
        return stdout, stderr


project_dir = './'

dnn_exe = "/home/monkbai/Downloads/tmp/resnet18-v1-7"
cat_img = "/home/monkbai/Downloads/tmp/cat.bin"
# pp_exe = os.path.abspath("./L1-capture-signal")
pp_exe = os.path.abspath("./L1i-capture-signal")



# ======= generate dataset =======


def get_func_range(func_asm_path: str):
    start_addr = ''
    end_addr = ''
    with open(func_asm_path, 'r') as f:
        asm_txt = f.read()
        lines = asm_txt.split('\n')
        for line in lines:
            if line.startswith(';'):
                continue
            start_addr = line.split(':')[0]
            break
        lines.reverse()
        for line in lines:
            if line.startswith(';') or len(line) < 1:
                continue
            end_addr = line.split(':')[0]
            break
    return start_addr.lower(), end_addr.lower()


def get_func_range_ret(func_asm_path: str):
    """ find the function end with the `ret` instruction """
    # print(func_asm_path)
    start_addr = ''
    end_addr = ''
    with open(func_asm_path, 'r') as f:
        asm_txt = f.read()
        
        if ' ret' not in asm_txt:
            return get_func_range(func_asm_path)
        
        lines = asm_txt.split('\n')
        for line in lines:
            if line.startswith(';'):
                continue
            start_addr = line.split(':')[0]
            break
        lines.reverse()
        for line in lines:
            if line.startswith(';') or len(line) < 1:
                continue
            if 'ret' not in line:
                continue
            end_addr = line.split(':')[0]
            break
    return start_addr.lower(), end_addr.lower()
            

def generate_trace_for_all(dnn_exe_dir="/home/monkbai/Downloads/tmp/experiment/TVM-0.12", 
                           trace_dir="/home/monkbai/Downloads/tmp/experiment/cache_dataset/cache_dataset_tvm/", 
                           tmp_log_dir="./tmp_log/",
                           compiler="tvm",
                           order_only=False):
    
    files = os.listdir(dnn_exe_dir)
    files.sort()
    for f in files:
        if ".lst" not in f and ".asm" not in f and ".txt" not in f and ".log" not in f:
            # if "alexnet" not in f:  # "resnet18-v1-7" not in f:
            #     continue  # debug

            # print(f)
            dnn_exe_path = os.path.join(dnn_exe_dir, f)
            if not os.path.isfile(dnn_exe_path):
                continue

            funcs_path = dnn_exe_path + '_funcs/'

            cur_trace_dir = os.path.join(trace_dir, f)
            # print(cur_trace_dir)
            # generate_trace_for_tvm(dnn_exe_path, funcs_path, cur_trace_dir, os.path.join(tmp_log_dir, f), compiler=compiler)
            # generate_trace_for_tvm_simple(dnn_exe_path, funcs_path, cur_trace_dir, os.path.join(tmp_log_dir, f), compiler=compiler)  # deprecated

            if order_only:
                get_dnn_exe_funcs_order(dnn_exe_path, funcs_path, cur_trace_dir, compiler=compiler) 
            else:
                generate_trace_for_tvm_LLC(dnn_exe_path, funcs_path, cur_trace_dir, compiler=compiler)



def timeout_run(cmd: str, timeout=5):
    # print(cmd)
    with subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, preexec_fn=os.setsid) as process:
        try:
            time.sleep(3)
            output, unused_err = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            print("timeout", end=' ')
            # os.kill(process.pid, signal.SIGKILL)
            os.killpg(process.pid, signal.SIGKILL)
            # process.kill()
            # output, unused_err = process.communicate()
            # raise TimeoutExpired(process.args, timeout, output=output)
            return -1, ""
        return unused_err, output

# ==============================================================
#                    for LLC attack
# ==============================================================

from colorama import Fore, Back, Style
from datetime import datetime


def test_filter(llc_trace_path: str):
    """ can the filter_trace_entropy be reused for LLC traces? """
    filter_trace_entropy_LLC(llc_trace_path)


def filter_trace_entropy_LLC(log_path):
    def entropy1(labels, base=None):
        for i in range(len(labels)):
            if labels[i] > 0:
                labels[i] = 1
        value,counts = np.unique(labels, return_counts=True)
        print("counts", counts)
        ent = entropy(counts, base=base)
        print(ent)
        return ent

    def entropy2(labels, base=None):
        c = 0
        for i in range(len(labels)):
            if labels[i] > 0:
                labels[i] = 1
                c += 1
        value,counts = np.unique(labels, return_counts=True)
        # print("value {} counts {}".format(value, counts))
        return c

    def get_avg_threshold(log: list):
        count = 0
        number = 0

        step = 30
        for i in range(0, len(log), step):
            tmp_log = log[i:i+step]
            np_arr = np.array(tmp_log)
            arr = np_arr.flatten()
            entro = entropy2(arr)

            number += entro
            count += 1
        return number / count


    threshold = 1  # 100
    new_log = []
    with open(log_path, 'r') as f:
        start_flag = False
        while True:
            line = f.readline().strip()
            if not line:
                break
            
            nums = line.split(" ")
            int_line = []
            for n in nums:
                int_line.append(int(n))
            new_log.append(int_line)
    assert len(new_log) > 0
    old_size = len(new_log)

    if old_size > 550000:  # the actual trace is longer than sample limit 
        print("no need to filter, size: {}".format(old_size))
        return old_size

    threshold = get_avg_threshold(new_log)
    print("threshold", int(threshold), end=' ')

    # see the entropy
    start_idx = -1
    end_idx = -1
    for idx in range(990, len(new_log), 30):
        if idx + 30 >= len(new_log):
            break
        np_arr = np.array(new_log[idx:idx+30])
        arr = np_arr.flatten()
        entro = entropy2(arr)  # entro = entropy1(arr)
        if entro > threshold + 10:  # entro > 0.27:
            start_idx = idx
            print("start idx:", start_idx, entro, end=' ')
            break
    for idx in range(len(new_log)-1, 0, -30):
        if idx - 30 < 0:
            break
        np_arr = np.array(new_log[idx-30:idx])
        arr = np_arr.flatten()
        entro = entropy2(arr)  # entro = entropy1(arr)
        if entro > threshold + 10:  # entro > 0.27:
            end_idx = idx
            print("end idx:", end_idx, entro, end=' ')
            break
    # print("start idx: {}, end idx: {}".format(start_idx, end_idx))
    
    if start_idx == -1 or end_idx == -1:
        print("no much info")
        return 0  # no valuable info logged 
    new_log = new_log[start_idx:end_idx]
    new_log_txt = []
    for l in new_log:
        new_l = [str(v) for v in l]
        new_l = ' '.join(new_l) + ' '
        new_log_txt.append(new_l)
    new_log_txt = '\n'.join(new_log_txt)
    
    # new_log_path = log_path[:log_path.rfind('.')] + '-filtered.log'
    new_log_path = log_path
    with open(new_log_path, 'w') as f:
        f.write(new_log_txt)
    print("size before: {}, size after filter: {}".format(old_size, len(new_log)))
    sys.stdout.flush()
    return len(new_log)


# def dnn_exe_profile():
#     """ for each DNN exe, we need to profile to get the function order """
#     return


def log_or_not_glow(name_list, idx):
    if idx >= len(name_list):
        return False, ""
    
    name = name_list[idx].split(".")[1]
    if "libjit" in name and ("conv" in name or "fc" in name or "pool" in name): 
        return True, name_list[idx][:-4]

    return False, ""

def log_or_not_tvm(name_list, idx, size):
    if idx >= len(name_list)-2:
        return False, ""
    
    name = name_list[idx].split(".")[1]
    if '_compute_' in  name and "conv2d" not in name and "dense" not in name and "pool" not in name: # like softmax? but we skip them
        return True, name_list[idx][:-4]
    
    # if '_compute_' in  name and "max_pool" in name and size > 0x50:  # max_pool
    #     return True, name_list[idx][:-4]
    
    if name.startswith("sub_") and not name_list[idx+1].split(".")[1].startswith("sub_"):  # conv2d and dense
        label_idx = idx - 1
        while label_idx > 0:
            if not name_list[label_idx].split(".")[1].startswith("sub_"):
                break
            label_idx -= 1
        # if "pool" in name_list[label_idx][:-4]:  # 2nd pool also need to be logged 
        #     return False, name_list[label_idx][:-4]
        return True, name_list[label_idx][:-4]
    elif name.startswith("sub_") and name_list[idx-1].split(".")[1].startswith("tvmgen"):  # handle pool operator
        label_idx = idx - 1
        
        if "pool" in name_list[label_idx].split(".")[1]:
            return True, name_list[label_idx][:-4]
    return False, ""


def get_dnn_exe_funcs_order(dnn_exe_path, 
                                  funcs_path, 
                                  trace_dir,
                                #   tmp_log_dir,
                                  compiler="tvm"):
    global cat_img, pp_exe

    # pp_exe = "/home/monkbai/Documents/DeepCache/i7-9700k/PRIME-SCOPE/primescope_demo/app"
    cat_img = "/home/monkbai/Documents/DeepCache/ONNX_Zoo_Loop/DNN_exe_HP/cat.bin"

    dnn_exe = dnn_exe_path
    dnn_exe_name = os.path.basename(dnn_exe_path)
    print(dnn_exe_name)

    # move log check to outer scope

    funcs_path = os.path.abspath(funcs_path)
    funcs = os.listdir(funcs_path)
    funcs.sort()

    # ======= Step 1 =======
    # get to know which functions should be logged
    # function name, start_addr, end_addr
    # leverages Pin tool
    log_funcs_dict = {}
    for idx in range(len(funcs)):
        func_path = os.path.join(funcs_path, funcs[idx])
        start_addr, end_addr = get_func_range_ret(func_path)
        # print("start_addr {}, end_addr {}".format(start_addr, end_addr))
        if (int(end_addr, 16) - int(start_addr, 16)) < 0x50:
            continue  # not a real operator
        
        if compiler == "tvm":
            should_log, label = log_or_not_tvm(funcs, idx, size=(int(end_addr, 16) - int(start_addr, 16)))
        elif compiler == 'glow':
            should_log, label = log_or_not_glow(funcs, idx)
        if should_log:
            if not("conv" in label or "dense" in label or "pool" in label or "fc" in label):
                continue
            log_funcs_dict[start_addr] = [label, start_addr, end_addr]
    # print(log_funcs_dict)

    # ======================
    # prepare addresses file for pin tool
    addrs_file = os.path.abspath("addrs_file.txt")
    with open(addrs_file, 'w') as f:
        for key, value in log_funcs_dict.items():
            s_addr = value[1]
            f.write(s_addr + '\n')
    
    # ======================
    # run the pin tool
    pin_home = '/home/monkbai/Downloads/pin-3.24-98612-g6bd5931f2-gcc-linux/'  # cpu 4
    mypintool_dir = '/home/monkbai/Downloads/pin-3.24-98612-g6bd5931f2-gcc-linux/source/tools/MyPinTool/'  # cpu4
    log_path = os.path.abspath("order.log")
    trace_log_cmd = pin_home + "pin -t " + \
                    mypintool_dir + "obj-intel64/FunCallTrace.so -o {} -addrs_file {} -- {} {}".format(log_path, addrs_file, dnn_exe, cat_img)
    status, output = subprocess.getstatusoutput("rm {}".format(log_path))
    # print(trace_log_cmd)  # for debug
    # status, output = subprocess.getstatusoutput(trace_log_cmd)
    status, output = timeout_run(trace_log_cmd, timeout=10)
    
    # ======================
    # analyze the pin trace --> function order 
    funcs_order_list = []
    funcs_order_set = set()
    with open(log_path, 'r') as f:
        lines = f.readlines()
        for idx in range(len(lines)):
            l = lines[idx]
            addr = int(l, 16)
            addr = hex(addr).lower()
            if addr not in funcs_order_set:
                funcs_order_set.add(addr)
                funcs_order_list.append(log_funcs_dict[addr])

    assert len(funcs_order_list) == len(log_funcs_dict)

    # ======================
    # store the function orders in a file, and return
    order_path = os.path.basename(dnn_exe) + '-order.txt'
    order_path = os.path.join(os.path.dirname(dnn_exe), order_path)
    with open(order_path, 'w') as f:
        for name, start_addr, end_addr, in funcs_order_list:
            f.write("{} {} {}\n".format(name, start_addr, end_addr))


def generate_trace_for_tvm_LLC(dnn_exe_path, 
                                  funcs_path, 
                                  trace_dir,
                                #   tmp_log_dir,
                                  compiler="tvm"):
    global cat_img, pp_exe

    # if 'vgg16_7' not in dnn_exe_path:
    #     return

    pp_exe = "/home/monkbai/Documents/DeepCache/i7-9700k/PRIME-SCOPE/primescope_demo/app"
    cat_img = "/home/monkbai/Documents/DeepCache/ONNX_Zoo_Loop/DNN_exe_HP/cat.bin"

    dnn_exe = dnn_exe_path
    dnn_exe_name = os.path.basename(dnn_exe_path)

    # ======================
    # trace directory
    if not os.path.exists(trace_dir):
        # print("mkdir {}".format(trace_dir))
        status = os.system("mkdir {}".format(trace_dir))
    else:
        return

    # ======= Step 2 =======
    # run the LLC attack, traces for all functions should be generated under the same directory
    # "{}-{}-{}.log".format(func_name, start_addr, end_addr)
    now = datetime.now()
    print('\n' + str(now))
    order_file = dnn_exe + '-order.txt'
    print("sudo cset shield --exec {} {} {} {}".format(pp_exe, dnn_exe, cat_img, order_file))
    status, output = subprocess.getstatusoutput("sudo cset shield --exec {} {} {} {}".format(pp_exe, dnn_exe, cat_img, order_file))


    # ======================
    # handle each trace
    with open(order_file, 'r') as f:
        lines = f.readlines()
        for l in lines:
            func_name, start_addr, end_addr = l.split(' ')
            func_name =func_name.strip()
            start_addr = start_addr.strip()
            end_addr = end_addr.strip()
            tmp_log_path = "{}-{}-{}.log".format(func_name, start_addr, end_addr)        
            log_path = os.path.join(trace_dir, "{}-{}-{}.log".format(func_name, start_addr, end_addr))  
            status, output = subprocess.getstatusoutput('mv {} {}'.format(tmp_log_path, log_path))
            
            print(tmp_log_path)
            filter_trace_entropy_LLC(log_path)
    
    time.sleep(5)
    
    # # ======================
    # funcs_path = os.path.abspath(funcs_path)
    # funcs = os.listdir(funcs_path)
    # funcs.sort()


    
    # # if not os.path.exists(tmp_log_dir):
    # #     status = os.system("mkdir {}".format(tmp_log_dir))
    
    # for idx in range(len(funcs)):
    #     func_path = os.path.join(funcs_path, funcs[idx])
    #     start_addr, end_addr = get_func_range_ret(func_path)
    #     # print("start_addr {}, end_addr {}".format(start_addr, end_addr))
    #     if (int(end_addr, 16) - int(start_addr, 16)) < 0x50:
    #         continue  # not a real operator
        
    #     if compiler == "tvm":
    #         should_log, label = log_or_not_tvm(funcs, idx, size=(int(end_addr, 16) - int(start_addr, 16)))
    #     elif compiler == 'glow':
    #         should_log, label = log_or_not_glow(funcs, idx)
    #     if should_log:
    #         if not("conv" in label or "dense" in label or "pool" in label or "fc" in label):
    #             continue

    #         log_path = os.path.join(trace_dir, "{}-{}-{}.log".format(label, start_addr, end_addr))
    #         # print(dnn_exe, cat_img, start_addr, end_addr, pp_exe, log_path)
    #         if os.path.exists(log_path):
    #             status, output = subprocess.getstatusoutput("wc {}".format(log_path))
    #             output = output.strip()
    #             length = int(output.split(' ')[0])
                
    #             if length > 1000: # or 'vgg' in dnn_exe:
    #                 # print("exists")
    #                 continue
    #             else:
    #                 print('\n', dnn_exe_name, label)
    #                 print("old length: {}".format(length))
    #         else:
    #             print('\n', dnn_exe_name, label)
    #         # input("continue?")

    #         now = datetime.now()
    #         print(now)
    #         print("sudo cset shield --exec {} {} {} {} {}".format(pp_exe, dnn_exe, cat_img, start_addr, end_addr))
    #         status, output = subprocess.getstatusoutput("rm output.log")
    #         status, output = subprocess.getstatusoutput("sudo cset shield --exec {} {} {} {} {}".format(pp_exe, dnn_exe, cat_img, start_addr, end_addr))
    #         # status, output = timeout_run("sudo cset shield --exec ./app {} {} {} {}".format(dnn_exe, cat_img, start_addr, end_addr))
    #         # if status != None:
    #         #     print(Fore.RED, "FAILED")
    #         #     print(output)
            
    #         status = os.system("cp {} {}".format("output.log", "{}-{}-{}-{}.log".format(os.path.basename(dnn_exe), label, start_addr, end_addr)))  # for debug use
    #         status = os.system("mv {} {}".format("output.log", log_path)) 
    #         filter_trace_entropy_LLC(log_path)

    #         time.sleep(5)

if __name__ == '__main__':
    # for root, dirs, files in os.walk("./LLC_dataset/LLC_dataset_tvm"):
    #     for filename in files:
    #         file_path = os.path.join(root, filename)
    #         status, output = subprocess.getstatusoutput('wc -l {}'.format(file_path))
    #         length = int(output.split(' ')[0])
    #         if length >= 200000:
    #             print(length, file_path)
    #             filter_trace_entropy_LLC(file_path)
    # exit(0)

    # ==============================================================
    #                    for LLC attack
    # ==============================================================
    # test_filter("/home/monkbai/Documents/DeepCache/i7-9700k/PRIME-SCOPE/primescope_demo/output.log")
    # test_filter("inception-v2-3-loop-0173.tvmgen_default_fused_nn_contrib_conv2d_NCHWc_multiply_add_nn_relu_1_compute_-0x624ea0-0x625c97.log")
    # exit(0)

    # generate_trace_for_all(dnn_exe_dir="/home/monkbai/Documents/DeepCache/ONNX_Zoo_Loop/DNN_exe_HP/TVM-0.12/", 
    #                        trace_dir="/home/monkbai/Documents/DeepCache/i7-9700k/experiment_llc/LLC_dataset/LLC_dataset_tvm/", 
    #                        tmp_log_dir="./tmp_log/",
    #                        compiler="tvm",
    #                       )  # order_only=True)
    
    # exit(0)

    generate_trace_for_all(dnn_exe_dir="/home/monkbai/Documents/DeepCache/ONNX_Zoo_Loop/DNN_exe_HP/Glow-2023/", 
                           trace_dir="/home/monkbai/Documents/DeepCache/i7-9700k/experiment_llc/LLC_dataset/LLC_dataset_glow/", 
                           tmp_log_dir="./tmp_log/",
                           compiler="glow",
                           )  # order_only=True)
    
    exit(0)
    
