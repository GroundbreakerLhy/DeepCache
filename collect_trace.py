import os
import sys
import time
import signal
import subprocess
import mmap
import struct
import json
from tqdm import tqdm
import numpy as np
from scipy.stats import entropy

dnn_input_vision = "/home/groundbreaker/DeepCache/experiment_llc/cat.bin"
dnn_input_nlp = "/home/groundbreaker/DeepCache/experiment_llc/nlp_input.bin"

# NLP models that need int64 input
NLP_MODELS = {"bertsquad", "gpt2", "roberta", "t5-encoder"}

# ============ Shared Memory Protocol ============
SHM_NAME = "/llc_signal"
SHM_SIZE = 4096


class SharedMemoryController:
    def __init__(self):
        self.shm_fd = None
        self.shm_map = None

    def open(self):
        """Open shared memory (attacker should create it first)"""
        shm_path = f"/dev/shm{SHM_NAME}"

        self.shm_fd = os.open(shm_path, os.O_RDWR)
        self.shm_map = mmap.mmap(self.shm_fd, SHM_SIZE)
        return True

    def get_status(self):
        """Read status from shared memory"""
        if self.shm_map is None:
            return -1
        self.shm_map.seek(0)
        return struct.unpack("i", self.shm_map.read(4))[0]

    def set_status(self, status):
        """Write status to shared memory"""
        if self.shm_map is None:
            return
        self.shm_map.seek(0)
        self.shm_map.write(struct.pack("i", status))
        self.shm_map.flush()

    def wait_for_status(self, expected_status, timeout=10):
        """Wait until status becomes expected_status"""
        start = time.time()
        last_print = start
        while self.get_status() != expected_status:
            elapsed = time.time() - start
            if elapsed > timeout:
                return False

            if time.time() - last_print > 50:
                print(f"[Wait] status={self.get_status()}, waiting for {expected_status}, {int(elapsed)}s elapsed")
                last_print = time.time()
            time.sleep(0.01)
        return True

    def close(self):
        if self.shm_map:
            self.shm_map.close()
        if self.shm_fd:
            os.close(self.shm_fd)


# ============ Spy (Attacker) Process Manager ============
class SpyController:
    def __init__(self, output_file="/tmp/llc_trace_buffer.log"):
        self.process = None
        self.output_file = output_file
        self.shm = SharedMemoryController()
        self.fn_shm_map = None

    def start(self):
        """Start persistent llc_attacker"""
        attacker_bin = os.path.abspath("./sca_tools/llc_attacker")

        for name in ["/llc_signal", "/llc_filename"]:
            shm_path = f"/dev/shm{name}"
            if os.path.exists(shm_path):
                os.remove(shm_path)

        # Start attacker (needs sudo for /proc/self/pagemap physical address access)
        cmd = ["sudo", "taskset", "-c", "0", attacker_bin, self.output_file]
        # Use DEVNULL instead of PIPE to prevent blocking when buffer fills
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )

        time.sleep(2)
        if not self.shm.open():
            self.stop()
            raise RuntimeError("Failed to open shared memory")

        # Wait for attacker to finish initialization (status == 2)
        print("[Spy] Waiting for attacker to be ready (status=2)...")
        if not self.shm.wait_for_status(2, timeout=180):
            print(f"[Spy] Timeout! Current status: {self.shm.get_status()}")
            self.stop()
            raise RuntimeError("Attacker failed to initialize (timeout waiting for status=2)")
        print("[Spy] Attacker ready!")

        # 写入文件名到 /llc_filename
        fn_fd = os.open("/dev/shm/llc_filename", os.O_RDWR)
        self.fn_shm_map = mmap.mmap(fn_fd, 1024)
        self.fn_shm_map.seek(0)
        self.fn_shm_map.write(self.output_file.encode("utf-8") + b"\x00")
        self.fn_shm_map.flush()

    def stop(self):
        """Stop the attacker process"""
        if self.process:
            self.shm.set_status(0)  # Signal exit
            time.sleep(0.5)
            # Use sudo to kill since attacker runs as root
            try:
                pgid = os.getpgid(self.process.pid)
                subprocess.run(["sudo", "kill", "-9", f"-{pgid}"], check=False)
            except Exception as e:
                print(f"[Spy] Warning: failed to kill attacker: {e}")
            self.process.wait(timeout=2)
            self.process = None
        if self.fn_shm_map:
            self.fn_shm_map.close()
        self.shm.close()


# ============ Utility Functions (from utils.py) ============
def extract_symbol_name(filename):
    """Extract symbol name from filename like '0018.symbol_name.txt'"""
    name = filename.replace(".txt", "")
    parts = name.split(".", 1)
    if len(parts) == 2:
        return parts[1]
    return name


def get_func_range(func_asm_path):
    """Extract function start and end addresses from assembly file"""
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


def log_or_not(name_list, idx, size):
    if idx >= len(name_list) - 2:
        return False, ""

    name = name_list[idx].split(".")[1]

    # 只采集 _compute_ 结尾的函数（实际计算函数，排除入口函数）
    if not name.endswith("_compute_"):
        return False, ""

    # 只采集 conv2d, dense, pool 相关算子
    if "conv2d" in name or "dense" in name or "pool" in name:
        if size > 0x50:
            return True, name_list[idx][:-4]

    return False, ""


# ============ Filter Function (from data_gen_paral.py) ============
def filter_trace_entropy_LLC(log_path):
    """Clean up LLC trace by removing low-entropy head/tail"""

    def entropy2(labels):
        c = 0
        for i in range(len(labels)):
            if labels[i] > 0:
                labels[i] = 1
                c += 1
        return c

    def get_avg_threshold(log):
        count, number = 0, 0
        step = 30
        for i in range(0, len(log), step):
            tmp_log = log[i : i + step]
            np_arr = np.array(tmp_log)
            arr = np_arr.flatten()
            entro = entropy2(arr.copy())
            number += entro
            count += 1
        return number / count if count > 0 else 1

    new_log = []
    if not os.path.exists(log_path):
        return 0

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                break
            nums = line.split(" ")
            int_line = [int(n) for n in nums if n]
            new_log.append(int_line)

    if new_log:
        new_log.pop()

    if len(new_log) == 0:
        return 0

    old_size = len(new_log)
    if old_size > 550000:
        return old_size

    threshold = get_avg_threshold(new_log)

    # Dynamically adjust start search based on trace length
    # Original used 990 for large traces, but we may have smaller traces
    search_start = min(30, old_size // 10)  # Start from 10% of trace or line 30

    # Find start
    start_idx = -1
    for idx in range(search_start, len(new_log), 30):
        if idx + 30 >= len(new_log):
            break
        np_arr = np.array(new_log[idx : idx + 30])
        arr = np_arr.flatten()
        entro = entropy2(arr.copy())
        if entro > threshold + 10:
            start_idx = idx
            break

    # Find end
    end_idx = -1
    for idx in range(len(new_log) - 1, 0, -30):
        if idx - 30 < 0:
            break
        np_arr = np.array(new_log[idx - 30 : idx])
        arr = np_arr.flatten()
        entro = entropy2(arr.copy())
        if entro > threshold + 10:
            end_idx = idx
            break

    if start_idx == -1 or end_idx == -1:
        print("[Filter] No valuable info (start_idx={}, end_idx={})".format(start_idx, end_idx))
        return 0

    new_log = new_log[start_idx:end_idx]
    new_log_txt = "\n".join([" ".join([str(v) for v in l]) for l in new_log])

    with open(log_path, "w") as f:
        f.write(new_log_txt)

    return len(new_log)


# ============ Main Collection Logic ============
def collect_operator_trace(binary_path, dnn_input, symbol_name):
    gdb_script = os.path.abspath("gdb.py")
    cmd = [
        "sudo",
        f"TRACE_SYMBOL={symbol_name}",
        f"TRACE_INPUT={dnn_input}",
        "taskset",
        "-c",
        "4",
        "gdb",
        "-q",
        "-nx",
        "-x",
        gdb_script,
        "--args",
        "build/demo_static",
    ]
    cwd = os.path.dirname(binary_path).replace("/build", "")

    subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
    )


def generate_trace_for_all(models_dir="compiled_models/tvm/", only_models=None):
    """Main entry: process all models (or only specified ones)"""
    models = sorted(
        [
            m
            for m in os.listdir(models_dir)
            if os.path.isdir(os.path.join(models_dir, m))
        ]
    )

    # Start persistent spy
    spy = SpyController("/tmp/llc_trace_buffer.log")
    spy.start()

    for model_name in models:
        if only_models is not None and model_name not in only_models:
            print(f"[Skip] {model_name}")
            continue
        print(f"[Model] Processing {model_name}")
        model_path = os.path.join(models_dir, model_name)
        build_dir = os.path.join(model_path, "build")
        binary_path = os.path.join(build_dir, "demo_static")
        funcs_dir_path = os.path.join(build_dir, "model.so_funcs")

        funcs = sorted(os.listdir(funcs_dir_path))

        # Prepare ops list
        ops_to_trace = []
        for idx in range(len(funcs)):
            func_path = os.path.join(funcs_dir_path, funcs[idx])
            start_addr, end_addr = get_func_range(func_path)
            size = int(end_addr, 16) - int(start_addr, 16)
            if size < 0x50:
                continue

            should_log, label = log_or_not(funcs, idx, size)
            if should_log:
                symbol_name = extract_symbol_name(funcs[idx])
                ops_to_trace.append(
                    {"symbol": symbol_name, "label": label, "idx": f"{idx:04d}"}
                )

        if len(ops_to_trace) == 0:
            continue

        # Output directory
        output_dir = f"./cache_dataset/cache_dataset_llc_tvm/{model_name}/"
        os.makedirs(output_dir, exist_ok=True)

        # Select input file based on model type
        model_input = dnn_input_nlp if model_name in NLP_MODELS else dnn_input_vision

        for idx, op in enumerate(ops_to_trace):
            print(f"[Operator {idx+1}/{len(ops_to_trace)}] {op['label']}")
            log_filename = f"{op['label']}.log"
            final_log_path = os.path.join(output_dir, log_filename)

            # Clean buffer file
            if os.path.exists("/tmp/llc_trace_buffer.log"):
                os.remove("/tmp/llc_trace_buffer.log")

            # Run gdb extraction
            # gdb sets status=3/2 during execution, then status=4 at end
            # attacker receives status=4, saves file, resets buffer, sets status=2
            collect_operator_trace(binary_path, model_input, op["symbol"])

            # Wait for status=2 (attacker saves file and resets after receiving status=4)
            if spy.shm.wait_for_status(2, timeout=120):
                # Small delay to ensure file is fully written
                time.sleep(0.2)
                if os.path.exists("/tmp/llc_trace_buffer.log"):
                    subprocess.run(
                        ["mv", "/tmp/llc_trace_buffer.log", final_log_path],
                        check=True,
                    )
                    # Filter low-entropy samples
                    filter_trace_entropy_LLC(final_log_path)
                else:
                    print("[Warning] status=2 but file not found, skipping")
            else:
                print(
                    f"[Warning] Timeout waiting for status=2 (status={spy.shm.get_status()})"
                )
                exit(1)

    spy.stop()


# ============ Entry Point ============
if __name__ == "__main__":
    # Only collect traces for these models (set to None to collect all)
    ONLY_MODELS = ["gpt2", "roberta", "t5-encoder"]

    subprocess.run(["sudo", "bash", "./sca_tools/reduce_noise.sh", "start"])
    if ONLY_MODELS is None:
        subprocess.run(["rm", "-rf", "./cache_dataset/cache_dataset_llc_tvm/"])
        os.mkdir("./cache_dataset/cache_dataset_llc_tvm/")
    else:
        os.makedirs("./cache_dataset/cache_dataset_llc_tvm/", exist_ok=True)
        for m in ONLY_MODELS:
            d = f"./cache_dataset/cache_dataset_llc_tvm/{m}"
            if os.path.exists(d):
                subprocess.run(["sudo", "rm", "-rf", d])
    generate_trace_for_all("compiled_models/tvm/", only_models=ONLY_MODELS)
    subprocess.run(["sudo", "bash", "./sca_tools/reduce_noise.sh", "stop"])
