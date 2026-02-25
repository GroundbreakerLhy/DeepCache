#!/usr/bin/python3
from subprocess import Popen, PIPE, STDOUT

import os
import time
import subprocess
import ctypes
import mmap
import struct

import config


class cd:
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


def cmd(commandline):
    with cd(project_dir):
        status, output = subprocess.getstatusoutput(commandline)
        # print(output)
        return status, output


def run(prog_path):
    with cd(project_dir):
        # print(prog_path)
        proc = subprocess.Popen(
            prog_path, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = proc.communicate()  # stderr: summary, stdout:  each statement
        return stdout, stderr


project_dir = "./"

pin_home = config.pin_home

mypintool_dir = config.mypintool_dir

trace_log_cmd = (
    pin_home
    + "pin -t "
    + mypintool_dir
    + "obj-intel64/TraceLogger.so -o {} -func_start {} -func_end {} -- {} {}"
)

kernel_log_cmd = (
    pin_home
    + "pin -t "
    + mypintool_dir
    + "obj-intel64/TraceLogger_kernel.so -o {} -func_start {} -func_end {} -- {} {}"
)

oram_log_cmd = (
    pin_home
    + "pin -t "
    + mypintool_dir
    + "obj-intel64/ORAMTrace.so -o {} -func_start {} -func_end {} -- {} {}"
)

write_log_cmd = (
    pin_home
    + "pin -t "
    + mypintool_dir
    + "obj-intel64/WriteLogger.so -o {} -func_start {} -func_end {} -- {} {}"
)

ciphertext_log_cmd = (
    pin_home
    + "pin -t "
    + mypintool_dir
    + "obj-intel64/CiphertextLogger.so -o {} -func_start {} -func_end {} -- {} {}"
)
inst_log_cmd = (
    pin_home
    + "pin -t "
    + mypintool_dir
    + "obj-intel64/InstLogger.so -o {} -func_start {} -func_end {} -- {} {}"
)

obfus_log_cmd = (
    pin_home
    + "pin -t "
    + mypintool_dir
    + "obj-intel64/ObfusSim.so -o {} -func_start {} -func_end {} -insert_point {} -- {} {}"
)

compile_tool_cmd = "make obj-intel64/{}.so TARGET=intel64 PIN_ROOT=" + pin_home
tools_list = [
    "TraceLogger",  # log all memory access address in a function (but only [12:7] bits)
    # 64 byte each cache line -> low 6 bit (block offset)
    # 64 l1 cache sets -> 12-7 bits (set index)
    # print every 100 memory accesses
    "TraceLogger_kernel",
    "ORAMTrace",
    "WriteLogger",
    "CiphertextLogger",
    "InstLogger",
    "ObfusSim",
]


def compile_all_tools():
    global project_dir
    for tool_name in tools_list:
        print("copying {} source code to MyPinTool dir...".format(tool_name))
        status, output = cmd("cp pin_tool/{}.cpp {}".format(tool_name, mypintool_dir))
        if status != 0:
            print(output)

    project_dir_backup = project_dir
    project_dir = mypintool_dir
    for tool_name in tools_list:
        print("compiling {}...".format(tool_name))
        status, output = cmd(compile_tool_cmd.format(tool_name))
        if status != 0:
            print(output)
    project_dir = project_dir_backup


def trace_log(dnn_exe, dnn_input, func_start, func_end, log_path):
    global project_dir
    project_dir_backup = project_dir
    project_dir = mypintool_dir
    # ------- set project_dir before instrumentation

    dnn_exe = os.path.abspath(dnn_exe)
    dnn_input = os.path.abspath(dnn_input)
    log_path = os.path.abspath(log_path)
    status, output = cmd(
        trace_log_cmd.format(log_path, func_start, func_end, dnn_exe, dnn_input)
    )

    # ------- end reset project_dir
    project_dir = project_dir_backup


def oram_log(dnn_exe, dnn_input, func_start, func_end, log_path):
    global project_dir
    project_dir_backup = project_dir
    project_dir = mypintool_dir
    # ------- set project_dir before instrumentation

    dnn_exe = os.path.abspath(dnn_exe)
    dnn_input = os.path.abspath(dnn_input)
    log_path = os.path.abspath(log_path)
    status, output = cmd(
        oram_log_cmd.format(log_path, func_start, func_end, dnn_exe, dnn_input)
    )

    # ------- end reset project_dir
    project_dir = project_dir_backup


def obfus_log(dnn_exe, dnn_input, func_start, func_end, log_path, insert_point):
    global project_dir
    project_dir_backup = project_dir
    project_dir = mypintool_dir
    # ------- set project_dir before instrumentation

    dnn_exe = os.path.abspath(dnn_exe)
    dnn_input = os.path.abspath(dnn_input)
    log_path = os.path.abspath(log_path)
    status, output = cmd(
        obfus_log_cmd.format(
            log_path, func_start, func_end, insert_point, dnn_exe, dnn_input
        )
    )
    print(output)
    # ------- end reset project_dir
    project_dir = project_dir_backup


def kernel_log(dnn_exe, dnn_input, func_start, func_end, log_path):
    global project_dir
    project_dir_backup = project_dir
    project_dir = mypintool_dir
    # ------- set project_dir before instrumentation

    dnn_exe = os.path.abspath(dnn_exe)
    dnn_input = os.path.abspath(dnn_input)
    log_path = os.path.abspath(log_path)
    status, output = cmd(
        kernel_log_cmd.format(log_path, func_start, func_end, dnn_exe, dnn_input)
    )

    # ------- end reset project_dir
    project_dir = project_dir_backup


def write_log(dnn_exe, dnn_input, func_start, func_end, log_path):
    status, output = cmd(
        write_log_cmd.format(log_path, func_start, func_end, dnn_exe, dnn_input)
    )


def inst_log(dnn_exe, dnn_input, func_start, func_end, log_path):
    global project_dir
    project_dir_backup = project_dir
    project_dir = mypintool_dir
    # ------- set project_dir before instrumentation

    dnn_exe = os.path.abspath(dnn_exe)
    dnn_input = os.path.abspath(dnn_input)
    log_path = os.path.abspath(log_path)

    status, output = cmd(
        inst_log_cmd.format(log_path, func_start, func_end, dnn_exe, dnn_input)
    )
    # ------- end reset project_dir
    project_dir = project_dir_backup


def ciphertext_log(dnn_exe, dnn_input, func_start, func_end, log_path):
    global project_dir
    project_dir_backup = project_dir
    project_dir = mypintool_dir
    # ------- set project_dir before instrumentation

    dnn_exe = os.path.abspath(dnn_exe)
    dnn_input = os.path.abspath(dnn_input)
    log_path = os.path.abspath(log_path)
    status, output = cmd(
        ciphertext_log_cmd.format(log_path, func_start, func_end, dnn_exe, dnn_input)
    )
    # ------- end reset project_dir
    project_dir = project_dir_backup


# Global variables for the persistent attacker server
server_proc = None
server_shm = None
SINGLE_TRACE_BUFFER = "/dev/shm/llc_trace_buffer.bin"
SHM_PATH = "/dev/shm/deepcache_sync"

import threading


def drain_thread(proc):
    while True:
        if proc.poll() is not None:
            break
        line = proc.stdout.readline()
        if not line:
            break
        print(f"[Attacker] {line.strip()}")


def cleanup_system():
    subprocess.run(["pkill", "-f", "llc_attacker"], stderr=subprocess.DEVNULL)


def start_server():
    global server_proc, server_shm
    if server_proc is not None:
        return

    cleanup_system()

    attacker_tool = os.path.abspath("./sca_tools/llc_attacker")

    # Initialize shared memory
    if os.path.exists(SHM_PATH):
        os.remove(SHM_PATH)

    # Create the file and set initial state to 0 (Wait)
    # Make it 4096 bytes to align with page size
    with open(SHM_PATH, "wb") as f:
        f.write(struct.pack("i", 0) * 1024)

    # Map it for the controller
    f_shm = open(SHM_PATH, "r+b")
    server_shm = mmap.mmap(f_shm.fileno(), 0)

    cmd_atk = [
        "taskset",
        "-c",
        str(config.attacker_core),
        attacker_tool,
        SINGLE_TRACE_BUFFER,
    ]
    print(f"[LLC] Launching Persistent Attacker: {' '.join(cmd_atk)}")

    # Start attacker
    server_proc = subprocess.Popen(
        cmd_atk, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    # Wait for initialization
    print("Waiting for attacker initialization...")
    while True:
        line = server_proc.stdout.readline()
        if not line:
            print("Attacker process ended unexpectedly.")
            # print(server_proc.stderr.read())
            server_proc = None
            return
        if "Failed" in line.strip():
            print(f"[Attacker Init] {line.strip()}")
        if "Monitoring ready" in line:
            break

    # Start drain thread to prevent pipe blocking
    t = threading.Thread(target=drain_thread, args=(server_proc,), daemon=True)
    t.start()


def stop_server():
    global server_proc, server_shm
    if server_proc is None:
        return

    if server_shm:
        server_shm.seek(0)
        server_shm.write(struct.pack("i", -1))
        server_shm.close()
        server_shm = None

    server_proc = None
    print("Attacker server stopped.")


import shutil


def clear_trace_buffer():
    if os.path.exists(SINGLE_TRACE_BUFFER):
        os.remove(SINGLE_TRACE_BUFFER)


def save_trace_buffer(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if os.path.exists(SINGLE_TRACE_BUFFER):
        shutil.move(SINGLE_TRACE_BUFFER, log_path)


def llc_log_server(
    dnn_exe, dnn_input, func_start, func_end, log_path=None, clear=False
):
    global server_shm, server_proc
    if server_shm is None:
        start_server()

    if clear:
        clear_trace_buffer()

    dnn_exe = os.path.abspath(dnn_exe)
    dnn_input = os.path.abspath(dnn_input)
    
    exe_dir = os.path.dirname(dnn_exe)
    if os.path.basename(exe_dir) == "build":
        work_dir = os.path.dirname(exe_dir)
    else:
        work_dir = exe_dir

    if not func_start.startswith("0x"):
        func_start = "0x" + func_start
    if not func_end.startswith("0x"):
        func_end = "0x" + func_end

    # Run Victim with Pin Tool (TraceLogger signals via shm)
    pin_cmd = [
        os.path.join(config.pin_home, "pin"),
        "-t",
        os.path.join(config.mypintool_dir, "obj-intel64/TraceLogger.so"),
        "-o", "/dev/null",
        "-func_start", func_start,
        "-func_end", func_end,
        "--",
        dnn_exe,
        dnn_input,
    ]
    victim_proc = subprocess.Popen(
        ["taskset", "-c", str(config.victim_core)] + pin_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=work_dir,
    )

    # Wait for victim to finish
    victim_proc.wait()

    # Wait for attacker to receive stop signal (2) and finish saving
    # Pin Tool sends signal 2 in Fini callback
    timeout = 5.0
    start = time.time()
    while time.time() - start < timeout:
        server_shm.seek(0)
        sig = struct.unpack("i", server_shm.read(4))[0]
        if sig == 2:
            break
        time.sleep(0.01)
    
    # Give attacker time to save data
    time.sleep(0.05)

    # Reset signal to 0 so attacker knows to wait for next start
    server_shm.seek(0)
    server_shm.write(struct.pack("i", 0))

    # Check if attacker died
    if server_proc.poll() is not None:
        print(f"Attacker died with return code: {server_proc.returncode}")

    # Save if path provided
    if log_path:
        save_trace_buffer(log_path)


if __name__ == "__main__":
    compile_all_tools()
    exit(0)

    dnn_exe = "./compiled_models/tvm/resnet18-v1-7"
    dnn_input = "./examples/resnet18_tvm_O3/cat.bin"
    func_start = "0x406f80"  # tvmgen_default_fused_nn_contrib_conv2d_NCHWc_add
    func_end = "0x409406"
    insert_point = "0x40877e"
    obfus_log(dnn_exe, dnn_input, func_start, func_end, "./tmp-1.log", insert_point)
    exit(0)

    # test
    # dnn_exe = "./examples/resnet18-glow2022/resnet18_v1_7.out"
    # func_start = "0x4047b0"  # libjit_conv2d_f2
    # func_end = "0x404e1f"

    dnn_exe = "./examples/resnet18_tvm_O3/resnet18_tvm_O3"
    func_start = "0x436630"  # tvmgen_default_fused_nn_contrib_conv2d_NCHWc_add_nn_relu_4_compute_
    func_end = "0x439abf"

    # dnn_exe = "./examples/resnet18_tvm_O0/resnet18_tvm_O0"
    # # func_start = "0x41F110"  # tvmgen_default_fused_nn_conv2d_4
    # # func_end = "0x425248"
    # # func_start = "0x433EF0"  # tvmgen_default_fused_nn_relu
    # # func_end = "0x4344DB"
    # func_start = "0x401A70"  # whole
    # func_end = "0x436065"
    dnn_input = "./examples/resnet18_tvm_O3/cat.bin"

    log_path = "./tmp.log"
    # write_log(dnn_exe, dnn_input, func_start, func_end, log_path)
    ciphertext_log(dnn_exe, dnn_input, func_start, func_end, log_path)

    # =======

    # dnn_exe = "./compiled_models/tvm/resnet18-v2-7"
    # dnn_input = "./examples/resnet18_tvm_O3/cat.bin"
    # func_start = '0x408880'  # tvmgen_default_fused_nn_contrib_conv2d_NCHWc
    # func_end = '0x409764'
    # inst_log(dnn_exe, dnn_input, func_start, func_end, "./tmp.log")

    dnn_exe = "./compiled_models/tvm/resnet18-v1-7"
    dnn_input = "./examples/resnet18_tvm_O3/cat.bin"
    func_start = "0x406f80"  # tvmgen_default_fused_nn_contrib_conv2d_NCHWc_add
    func_end = "0x409406"
    inst_log(dnn_exe, dnn_input, func_start, func_end, "./tmp-1.log")
