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


def wrapper(dnn_exe, in_data, func_start, func_end,
            pp_exe, log_path, lazy=True, init_samples=-1):
    """ The number of samples should not be overly large, pp may not initialized when target function finished  
        also should not be too small
        Ideal: PP started, but not fully finished
    """
    if lazy and os.path.exists(log_path):
        return -1

    # 1. try to find a reasonable sample number
    if init_samples == -1:
        min_samples = 2000   # 2k
        max_samples = 1000000 # 1m

        cur_samples = 200000  # 200k
    else:
        cur_samples = init_samples
        min_samples = max(2000, init_samples-5000)
        max_samples = min(1000000, init_samples+5000)
    prev_samples = cur_samples
    print("samples", end=' ')
    while min_samples < max_samples:
        cur_samples = int(cur_samples)
        print(cur_samples, end=" ")
        
        status, output = cmd(f"python3 ./single_log.py {dnn_exe} {in_data} {func_start} {func_end} {pp_exe} {log_path} {cur_samples}")  # , debug_flag=True)
        sys.stdout.flush()
        if status != 0:
            # the funciton range could be wrong
            # clean the logged trace
            print("return val", status)
            with open(log_path, 'w') as f:
                f.write("")
            return -1, cur_samples
        if "Prime+Probe finished" in output:
            # longer
            min_samples = cur_samples
            prev_samples = cur_samples
            cur_samples = (min_samples + max_samples) / 2
            cur_samples -= cur_samples % 100
        else:
            # shorter
            max_samples = cur_samples
            prev_samples = cur_samples
            cur_samples = (min_samples + max_samples) / 2
            cur_samples -= cur_samples % 100
        if prev_samples == cur_samples:
            break
    print()
    cur_samples = int(cur_samples)
    try_count = 30
    while try_count > 0 and not ("Prime+Probe started" in output and "Prime+Probe finished" not in output):
        if "Prime+Probe finished" in output:
            cur_samples += 100
        elif "Prime+Probe started" not in output:
            cur_samples -= 100
        print(cur_samples, end=" ")
        status, output = cmd(f"sudo python3 ./single_log.py {dnn_exe} {in_data} {func_start} {func_end} {pp_exe} {log_path} {cur_samples}", debug_flag=try_count==1)
        try_count -= 1
    if try_count == 0:
        started = "Prime+Probe started" in output
        finished = "Prime+Probe finished" in output
        print("no ideal case, started {}, finished {}".format(started, finished))
        print(output)
        sys.stdout.flush()
        # input("continue")
        with open(log_path, 'w') as f:
            f.close()
            return -1, cur_samples

    # if not os.path.exists(log_path) and cur_samples == 5000:
    #     # the trace is too short to log  # No that is wrong
    #     print("[too short to log]", end=" ")
    #     return -1

    # 2. remove redundant part in the trace log
    filter_size = filter_trace_entropy(log_path)
    return filter_size, cur_samples


def filter_trace(log_path):
    threshold = 1  # 100
    new_log = []
    with open(log_path, 'r') as f:
        start_flag = False
        while True:
            line = f.readline().strip()
            if not line:
                break
            
            nums = line.split(" ")
            hit_count = 0
            for n in nums:
                if int(n) < threshold:
                    hit_count += 1
            if not start_flag and (hit_count > 60 or hit_count < 5):
                continue
            elif not start_flag:
                start_flag = True
            new_log.append(line)
    assert len(new_log) > 0
    # remove end part
    end_idx = -1
    for idx in range(len(new_log)-1, 0, -1):
        line = new_log[idx]

        nums = line.split(" ")
        hit_count = 0
        for n in nums:
            if int(n) < threshold:
                hit_count += 1
        if (hit_count > 60 or hit_count < 5):
            continue
        else:
            end_idx = idx
            break
    new_log = new_log[:end_idx]
    new_log_txt = '\n'.join(new_log)
    with open(log_path, 'w') as f:
        f.write(new_log_txt)
    return len(new_log)


def filter_trace_entropy(log_path):
    def entropy1(labels, base=None):
        value,counts = np.unique(labels, return_counts=True)
        return entropy(counts, base=base)

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
    
    # see the entropy
    start_idx = -1
    end_idx = -1
    for idx in range(30, len(new_log), 30):
        if idx + 30 >= len(new_log):
            break
        np_arr = np.array(new_log[idx:idx+30])
        arr = np_arr.flatten()
        entro = entropy1(arr)
        if entro > 0.27:
            start_idx = idx
            print(start_idx, entro)
            break
    for idx in range(len(new_log)-1, 0, -30):
        if idx - 30 < 0:
            break
        np_arr = np.array(new_log[idx-30:idx])
        arr = np_arr.flatten()
        entro = entropy1(arr)
        if entro > 0.27:
            end_idx = idx
            print(end_idx, entro)
            break
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
    with open(log_path, 'w') as f:
        f.write(new_log_txt)
    print("size after filter:", len(new_log))
    sys.stdout.flush()
    return len(new_log)


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


def generate_trace_for_tvm(dnn_exe_path, 
                           funcs_path="/home/monkbai/Downloads/tmp/experiment/resnet18-v1-7_funcs/", 
                           trace_dir = "/home/monkbai/Downloads/tmp/experiment/cache_dataset/resnet18-v1-7/",
                           tmp_log_dir="",
                           compiler="tvm"):
    global cat_img, pp_exe

    dnn_exe = dnn_exe_path
    dnn_exe_name = os.path.basename(dnn_exe_path)

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
        if '_compute_' in  name and "conv2d" not in name and "dense" not in name and "max_pool" not in name: 
            return True, name_list[idx][:-4]
        
        if '_compute_' in  name and "max_pool" in name and size > 0x50:  # max_pool
            return True, name_list[idx][:-4]
        
        if name.startswith("sub_") and not name_list[idx+1].split(".")[1].startswith("sub_"):  # conv2d and dense
            label_idx = idx - 1
            while label_idx > 0:
                if not name_list[label_idx].split(".")[1].startswith("sub_"):
                    break
                label_idx -= 1
            return True, name_list[label_idx][:-4]
        return False, ""

    def safe_average(log_list):
        # ignore outliers when calculating average (mean)
        # in worst case, return median
        length_list = [v[1] for v in log_list if v[1] > 0]
        if len(length_list) == 0:
            # al failed
            return 0.0
        med = statistics.median(length_list)
        d = [abs(l - med) for l in length_list]  # distance to median
        mdev = statistics.median(d)
        if mdev == 0:
            mdev = 1
        s = [l/mdev for l in d]
        
        threshold = 50
        count = 0
        sum = 0
        for idx in range(len(length_list)):
            if s[idx] < threshold:
                sum += length_list[idx]
                count += 1
        # patch
        if count == 2:
            sum += 1  # choose the larger one
        return sum / count

    funcs_path = os.path.abspath(funcs_path)
    funcs = os.listdir(funcs_path)
    funcs.sort()

    if not os.path.exists(trace_dir):
        # print("mkdir {}".format(trace_dir))
        status = os.system("mkdir {}".format(trace_dir))
    if not os.path.exists(tmp_log_dir):
        status = os.system("mkdir {}".format(tmp_log_dir))
    
    for idx in range(len(funcs)):
        func_path = os.path.join(funcs_path, funcs[idx])
        start_addr, end_addr = get_func_range(func_path)
        if (int(end_addr, 16) - int(start_addr, 16)) < 0x50:
            continue  # not a real operator
        
        if compiler == "tvm":
            should_log, label = log_or_not_tvm(funcs, idx, size=(int(end_addr, 16) - int(start_addr, 16)))
        elif compiler == 'glow':
            should_log, label = log_or_not_glow(funcs, idx)
        if should_log:
            if not("conv" in label or "dense" in label or "pool" in label or "fc" in label):
                continue

            log_path = os.path.join(trace_dir, "{}-{}-{}-.log".format(label, start_addr, end_addr))
            # print(dnn_exe, cat_img, start_addr, end_addr, pp_exe, log_path)
            print(dnn_exe_name, label)
            if os.path.exists(log_path):
                print("exisit")
                continue
            
            failed_case = False
            length_avg = 0.0
            while length_avg < 128:
                # repeat several times to find an average length
                length_dict = {}
                init_samples = -1
                for i in range(5):
                    print("Tmp try", i)
                    tmp_log_path = os.path.join(tmp_log_dir, "{}-{}-{}-{}.log".format(label, start_addr, end_addr, i))
                    length, init_samples = wrapper(dnn_exe, cat_img, start_addr, end_addr, pp_exe, tmp_log_path, lazy=False)  # , init_samples=init_samples)
                    length_dict[tmp_log_path] = length

                    if length == -1:
                        # failed, due to wrong function range
                        failed_case = True
                        break
            
                if failed_case:
                    break

                tmp_trace_list = list(length_dict.items())
                length_avg = safe_average(tmp_trace_list)
                # print(tmp_trace_list)
                print("average length", length_avg)
                break
            
            if failed_case:
                continue

            if length_avg < 128:
                print("Failed to log trace: the trace is too short!\n")
                sys.stdout.flush()
                continue
            else:
                # find a close one 
                candidate_idx = min(range(len(tmp_trace_list)), key = lambda i: abs(tmp_trace_list[i][1] - length_avg))
                candidate = tmp_trace_list[candidate_idx]
                print(start_addr, end_addr)
                # input("continue?")
                print(candidate_idx)
                print("Candidate's length", candidate[1], '\n')
                sys.stdout.flush()
                status = os.system("cp {} {}".format(candidate[0], log_path))
            

def generate_trace_for_all(dnn_exe_dir="/home/monkbai/Downloads/tmp/experiment/TVM-0.12", 
                           trace_dir="/home/monkbai/Downloads/tmp/experiment/cache_dataset/cache_dataset_tvm/", 
                           tmp_log_dir="./tmp_log/",
                           compiler="TVM"):
    
    files = os.listdir(dnn_exe_dir)
    files.sort()
    for f in files:
        if ".lst" not in f and ".asm" not in f:
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

            generate_trace_for_tvm_LLC(dnn_exe_path, funcs_path, cur_trace_dir, compiler="tvm")


# ======= a simpler version =======
# =======     deprecated    =======
def generate_trace_for_tvm_simple(dnn_exe_path, 
                                  funcs_path, 
                                  trace_dir,
                                  tmp_log_dir,
                                  compiler="tvm"):
    global cat_img, pp_exe

    dnn_exe = dnn_exe_path
    dnn_exe_name = os.path.basename(dnn_exe_path)

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
        if '_compute_' in  name and "conv2d" not in name and "dense" not in name and "max_pool" not in name: 
            return True, name_list[idx][:-4]
        
        if '_compute_' in  name and "max_pool" in name and size > 0x50:  # max_pool
            return True, name_list[idx][:-4]
        
        if name.startswith("sub_") and not name_list[idx+1].split(".")[1].startswith("sub_"):  # conv2d and dense
            label_idx = idx - 1
            while label_idx > 0:
                if not name_list[label_idx].split(".")[1].startswith("sub_"):
                    break
                label_idx -= 1
            return True, name_list[label_idx][:-4]
        return False, ""

    funcs_path = os.path.abspath(funcs_path)
    funcs = os.listdir(funcs_path)
    funcs.sort()

    if not os.path.exists(trace_dir):
        # print("mkdir {}".format(trace_dir))
        status = os.system("mkdir {}".format(trace_dir))
    if not os.path.exists(tmp_log_dir):
        status = os.system("mkdir {}".format(tmp_log_dir))
    
    for idx in range(len(funcs)):
        func_path = os.path.join(funcs_path, funcs[idx])
        start_addr, end_addr = get_func_range(func_path)
        if (int(end_addr, 16) - int(start_addr, 16)) < 0x50:
            continue  # not a real operator
        
        if compiler == "tvm":
            should_log, label = log_or_not_tvm(funcs, idx, size=(int(end_addr, 16) - int(start_addr, 16)))
        elif compiler == 'glow':
            should_log, label = log_or_not_glow(funcs, idx)
        if should_log:
            if not("conv" in label or "dense" in label or "pool" in label or "fc" in label):
                continue

            log_path = os.path.join(trace_dir, "{}-{}-{}.log".format(label, start_addr, end_addr))
            # print(dnn_exe, cat_img, start_addr, end_addr, pp_exe, log_path)
            if os.path.exists(log_path):
                # print("exisit")
                continue
            print(dnn_exe_name, label)

            # status, output = subprocess.getstatusoutput("./app {} {} {} {}".format(dnn_exe, cat_img, start_addr, end_addr))
            status, output = timeout_run("./app {} {} {} {}".format(dnn_exe, cat_img, start_addr, end_addr))
            while status != None:
                print("Retry")
                time.sleep(3)
                status, output = timeout_run("./app {} {} {} {}".format(dnn_exe, cat_img, start_addr, end_addr))
            output = str(output)
            match = re.search("count number: \d+", output)
            print(match.group())
            status = os.system("mv {} {}".format("output.log", log_path)) 


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

    if old_size > 199999:  # the actual trace is longer than sample limit 
        print("no need to filter, size: {}".format(old_size))
        return old_size

    threshold = get_avg_threshold(new_log)
    print("threshold", threshold)

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
            print(start_idx, entro, end=' ')
            break
    for idx in range(len(new_log)-1, 0, -30):
        if idx - 30 < 0:
            break
        np_arr = np.array(new_log[idx-30:idx])
        arr = np_arr.flatten()
        entro = entropy2(arr)  # entro = entropy1(arr)
        if entro > threshold + 10:  # entro > 0.27:
            end_idx = idx
            print(end_idx, entro)
            break
    print("start idx: {}, end idx: {}".format(start_idx, end_idx))
    
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


def generate_trace_for_tvm_LLC(dnn_exe_path, 
                                  funcs_path, 
                                  trace_dir,
                                #   tmp_log_dir,
                                  compiler="tvm"):
    global cat_img, pp_exe

    pp_exe = "/home/monkbai/Documents/DeepCache/i7-9700k/PRIME-SCOPE/primescope_demo/app"
    cat_img = "/home/monkbai/Documents/DeepCache/ONNX_Zoo_Loop/DNN_exe_HP/cat.bin"

    dnn_exe = dnn_exe_path
    dnn_exe_name = os.path.basename(dnn_exe_path)

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

    funcs_path = os.path.abspath(funcs_path)
    funcs = os.listdir(funcs_path)
    funcs.sort()

    if not os.path.exists(trace_dir):
        # print("mkdir {}".format(trace_dir))
        status = os.system("mkdir {}".format(trace_dir))
    # if not os.path.exists(tmp_log_dir):
    #     status = os.system("mkdir {}".format(tmp_log_dir))
    
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

            log_path = os.path.join(trace_dir, "{}-{}-{}.log".format(label, start_addr, end_addr))
            # print(dnn_exe, cat_img, start_addr, end_addr, pp_exe, log_path)
            if os.path.exists(log_path):
                status, output = subprocess.getstatusoutput("wc {}".format(log_path))
                output = output.strip()
                length = int(output.split(' ')[0])
                
                if length > 1000: # or 'vgg' in dnn_exe:
                    # print("exists")
                    continue
                else:
                    print('\n', dnn_exe_name, label)
                    print("old length: {}".format(length))
            else:
                print('\n', dnn_exe_name, label)
            # input("continue?")

            now = datetime.now()
            print(now)
            print("sudo cset shield --exec {} {} {} {} {}".format(pp_exe, dnn_exe, cat_img, start_addr, end_addr))
            status, output = subprocess.getstatusoutput("rm output.log")
            status, output = subprocess.getstatusoutput("sudo cset shield --exec {} {} {} {} {}".format(pp_exe, dnn_exe, cat_img, start_addr, end_addr))
            # status, output = timeout_run("sudo cset shield --exec ./app {} {} {} {}".format(dnn_exe, cat_img, start_addr, end_addr))
            # if status != None:
            #     print(Fore.RED, "FAILED")
            #     print(output)
            
            status = os.system("cp {} {}".format("output.log", "{}-{}-{}-{}.log".format(os.path.basename(dnn_exe), label, start_addr, end_addr)))  # for debug use
            status = os.system("mv {} {}".format("output.log", log_path)) 
            filter_trace_entropy_LLC(log_path)

            time.sleep(5)

if __name__ == '__main__':
    # ==============================================================
    #                    for LLC attack
    # ==============================================================
    # test_filter("/home/monkbai/Documents/DeepCache/i7-9700k/PRIME-SCOPE/primescope_demo/output.log")
    # test_filter("inception-v2-3-loop-0173.tvmgen_default_fused_nn_contrib_conv2d_NCHWc_multiply_add_nn_relu_1_compute_-0x624ea0-0x625c97.log")
    # exit(0)
    generate_trace_for_all(dnn_exe_dir="/home/monkbai/Documents/DeepCache/ONNX_Zoo_Loop/DNN_exe_HP/TVM-0.12/", 
                           trace_dir="/home/monkbai/Documents/DeepCache/i7-9700k/experiment_llc/LLC_dataset/LLC_dataset_tvm/", 
                           tmp_log_dir="./tmp_log/",
                           compiler="tvm")
    
    exit(0)
    

    # status, output = timeout_run("./app /home/monkbai/Downloads/DNN_exes/TVM-0.12/densenet-3 /home/monkbai/Downloads/tmp/cat.bin 0x457850 0x4580ef")
    # # status, output = timeout_run("sleep 3")
    # print(status, output)
    # exit(0)


    # generate_trace_for_tvm(dnn_exe_path="/home/monkbai/Downloads/tmp/experiment/TVM-0.12/resnet18-v1-7",
    #                        funcs_path="/home/monkbai/Downloads/tmp/experiment/resnet18-v1-7_funcs/", 
    #                        trace_dir = "/home/monkbai/Downloads/tmp/experiment/cache_i_dataset/resnet18-v1-7/",
    #                        tmp_log_dir="",
    #                        compiler="tvm")
    # exit(0)  # test
    






    # generate_trace_for_all(dnn_exe_dir="/home/monkbai/Downloads/DNN_exes/TVM-0.12", 
    #                        trace_dir="/home/monkbai/Downloads/experiment_llc/cache_dataset/cache_dataset_tvm/",
    #                        tmp_log_dir="./tmp_log_tvm/",
    #                        compiler="tvm")
    # exit(0)
    
    generate_trace_for_all(dnn_exe_dir="/home/monkbai/Downloads/DNN_exes/Glow-2023", 
                           trace_dir="/home/monkbai/Downloads/experiment_llc/cache_dataset/cache_dataset_glow/",
                           compiler="glow")
    exit(0)

    # # filter_trace_entropy("/home/monkbai/Downloads/tmp/experiment/cache_dataset/resnet18-v1-7/0053.tvmgen_default_fused_nn_contrib_conv2d_NCHWc_add_1_compute_-0x409d30-0x40bd8c-.log")
    # filter_trace_entropy("/home/monkbai/Downloads/tmp/experiment/cache_dataset/cache_dataset_tvm/bvlcalexnet-3/0022.tvmgen_default_fused_layout_transform_compute_-0x402010-0x4021e9-.log")
    # exit(0)
    
    # generate_trace_for_tvm(funcs_path="/home/monkbai/Downloads/tmp/experiment/resnet18-v1-7_funcs/")
    generate_trace_for_all(dnn_exe_dir="/home/monkbai/Downloads/tmp/experiment/TVM-0.12", 
                           trace_dir="/home/monkbai/Downloads/tmp/experiment/cache_dataset/cache_dataset_tvm/")
    exit(0)
    
    # Test
    status, output = cmd("sudo python3 ./single_log.py /home/monkbai/Downloads/tmp/resnet18-v1-7 /home/monkbai/Downloads/tmp/cat.bin 0x409d30 0x40bd8c ./L1-capture-signal ./L1-log.txt 100000")
    print(status, output)
