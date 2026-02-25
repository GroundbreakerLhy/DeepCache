#!/usr/bin/python3
import os
import sys
import time
import signal
import subprocess

# ======= Prime+Probe utilities =======


def start(cmd):
    print(cmd)
    return subprocess.Popen(
        cmd,
        shell= True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)


def start_simple(cmd):
    print(cmd)
    return subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)


def read(process, line_count=1, stop_word="Breakpoint 2", fail_word=""):
    # return process.stdout.readline().decode("utf-8").strip()
    lines = ''
    while line_count:
        line = process.stdout.readline()  # blocking attention
        if not line:
            break
        line = line.decode("utf-8")
        print(line.strip())
        lines += line
        line_count -= 1
        if line.startswith(stop_word):
            break
        if "The maximum position in output vector" in line:
            return ""
        if len(fail_word) > 0 and fail_word in line:
            return ""
    return lines.strip()


def write(process, message):
    print(message)
    process.stdin.write(f"{message.strip()}\n".encode("utf-8"))
    process.stdin.flush()


def terminate(process):
    process.stdin.close()
    process.terminate()
    process.wait(timeout=0.2)

def kill_p(process):
    process.kill()
    process.wait(timeout=2)


def start_attack(pp_exe="/home/monkbai/Downloads/tmp/Mastik/demo/L3-capturecount_signal", 
                 log_path="/home/monkbai/Downloads/tmp/Mastik/demo/L3-log.txt", 
                 samples=100000):
    """ prime+probe running on core 0
        victim dnn exe running on core 3 
        core 3 and 0 (virtual cores) are form differen physical cpu cores 
    """
    return start_simple(f"taskset -c 0 nice -n -10 {pp_exe} {samples} {log_path}",shell= True,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)  # run on cpu 0


def end_attack(proc):
    pp_id = proc.pid
    # os.killpg(proc.pid, signal.SIGTERM)
    # os.killpg(proc.pid, signal.SIGINT)
    
    os.killpg(proc.pid, signal.SIGUSR1)
    print(f"process ({proc.pid}) is killed.")
    
    # parrrent_proc = psutil.Process(pp_id)
    # print(parrrent_proc)
    # children = parrrent_proc.children(recursive=True)
    # print(children)
    # for child in children:
    #     if "L1-capture" in child.name():
    #         print("Child pid is {}".format(child.pid))
    #         os.killpg(child.pid, signal.SIGUSR1)
    #         print(f"process ({child.pid}) is killed.")


# ======= Prime+Probe & wrapper =======


def single_log(dnn_exe, in_data, func_start, func_end,
               pp_exe, log_path, samples=200000):
    # before calling this function, a cpuset shield (core 1 and 3) should be create
    
    # the victim process running on the cpu core 3
    process = start(f"taskset -c 3 nice -n -10 gdb --args {dnn_exe} {in_data}")
    # cset shield --> taskset --> nice --> gdb --> dnn_exe
    print(read(process, 50, stop_word="Reading symbols"))
    
    write(process, f"b *{func_start}")
    read(process)
    write(process, f"b *{func_end}")
    read(process)

    write(process, f"r")
    read(process, 50, stop_word="Breakpoint 1")
    time.sleep(0.5)
    
    # invoke prime+probe here
    pp_proc = start_attack(pp_exe, log_path, samples)
    ret_str = read(process, 50, stop_word="Breakpoint 2")  # time.sleep(0.8)

    # run the target function
    write(process, f"c")
    ret_str = read(process, 50, stop_word="Breakpoint 2", fail_word="Breakpoint 1")
    
    # terminate prime+probe here
    # time.sleep(3)
    end_attack(pp_proc)

    terminate(process)

    return ret_str


# ======= Prime+Scope & wrapper =======
# ========     deprecated    ==========

app_path = "/home/monkbai/Downloads/PRIME-SCOPE/primescope_demo/app"

def single_log_PS(dnn_exe, in_data, func_start, func_end, log_path):
    # the attacker process running on the cpu core 1
    # the victim process running on the cpu core 2
    # process = start_simple(f"nice -n -10 {app_path} {dnn_exe} {in_data} {func_start} {func_end}")
    err, out = timeout_run(f"{app_path} {dnn_exe} {in_data} {func_start} {func_end}")
    try_count = 1
    
    tmp_path = os.path.abspath('output.log')
    lines = 150000
    if os.path.exists(tmp_path):
        print("found output.log")
        status, output = subprocess.getstatusoutput("wc -l {}".format(tmp_path))
        lines = int(output.split(" ")[0])
    
    while (not os.path.exists(tmp_path) or lines == 150000) and try_count < 10:
        print("try_count", try_count)
        # run the attack again
        err, out = timeout_run(f"{app_path} {dnn_exe} {in_data} {func_start} {func_end}")
        try_count += 1

    sattus, output = subprocess.getstatusoutput("mv {} {}".format(tmp_path, log_path))

    return try_count

def timeout_run(cmd: str, tt=25):
    # print(cmd)
    with subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid) as process:
        try:
            time.sleep(3)
            output, unused_err = process.communicate(timeout=tt)
        except subprocess.TimeoutExpired:
            print("timeout", end=' ')
            # os.kill(process.pid, signal.SIGKILL)
            os.killpg(process.pid, signal.SIGKILL)
            # process.kill()
            # output, unused_err = process.communicate()
            # raise TimeoutExpired(process.args, timeout, output=output)
            return -1, ""
        return unused_err, output

# =====================================

def outdated_test():
    # if (len(sys.argv) == 8):
    #     dnn_exe = os.path.abspath(sys.argv[1])
    #     in_data = os.path.abspath(sys.argv[2])
    #     func_start = sys.argv[3]
    #     func_end = sys.argv[4]
    #     pp_exe = os.path.abspath(sys.argv[5])
    #     log_path = os.path.abspath(sys.argv[6])
    #     samples = int(sys.argv[7])
    #     # print("Args: <dnn_exe> {}, <in_data> {}, <func_start> {}, <func_end> {}, <pp_exe> {}, <log_path> {}, <samples>{}".format(dnn_exe, in_data, func_start, func_end, pp_exe, log_path, samples))
    #     ret_str = single_log(dnn_exe, in_data, func_start, func_end,
    #                          pp_exe, log_path, samples)
    #     if len(ret_str) == 0:
    #         exit(-1)
    # else:
    #     print("Args: <dnn_exe>, <in_data>, <func_start>, <func_end>, <pp_exe>, <log_path>, <samples>")


    # exit(0)


    # Test
    dnn_exe = "/home/monkbai/Downloads/tmp/resnet18-v1-7"
    cat_img = "/home/monkbai/Downloads/tmp/cat.bin"

    func_start = '0x409d30'
    func_end ='0x40bd8c'
    
    pp_exe="/home/monkbai/Downloads/tmp/Mastik/demo/L3-capturecount_signal" 
    log_path="/home/monkbai/Downloads/tmp/Mastik/demo/L3-log.txt"
    samples=100000
    ret_str = single_log(dnn_exe, cat_img, func_start, func_end, pp_exe, log_path, samples)
    exit(0)

    # the victim process running on the cpu core 3
    process = start(f"taskset -c 3 gdb --args {dnn_exe} {cat_img}")
    print(read(process, 15))
    
    write(process, f"b *{func_start}")
    read(process)
    write(process, f"b *{func_end}")
    read(process)

    write(process, f"r")
    read(process)
    
    
    # invoke prime+probe here
    pp_proc = start_attack()
    # time.sleep(1)

    # run the target function
    write(process, f"c")
    read(process, 10)
    
    # terminate prime+probe here
    # time.sleep(3)
    end_attack(pp_proc)

    terminate(process)


if __name__ == '__main__':
    dnn_exe = '/home/monkbai/Downloads/PRIME-SCOPE/primescope_demo/resnet18-v1-7'
    in_data = '/home/monkbai/Downloads/PRIME-SCOPE/primescope_demo/cat.bin'

    func_start = '0x41afd0'
    func_end = '0x41c60b'

    log_path = "./tmp_.log"
    single_log_PS(dnn_exe, in_data, func_start, func_end, log_path)
