import gdb
import os
import time
import mmap
import struct
import sys

# ============ Shared Memory Protocol ============
# Must match collect_trace.py
SHM_NAME = "/llc_signal"
SHM_SIZE = 4096


class SharedMemoryController:
    def __init__(self):
        self.shm_fd = None
        self.shm_map = None

    def open(self):
        """Open shared memory (attacker should create it first)"""
        max_retries = 30
        for attempt in range(max_retries):
            shm_path = f"/dev/shm{SHM_NAME}"
            if not os.path.exists(shm_path):
                if attempt < max_retries - 1:
                    time.sleep(0.1)
                    continue
                else:
                    print(f"Shared memory not found: {shm_path}")
                    return False

            self.shm_fd = os.open(shm_path, os.O_RDWR)
            self.shm_map = mmap.mmap(self.shm_fd, SHM_SIZE)
            return True
        return False

    def set_status(self, status):
        """Write status to shared memory"""
        if self.shm_map is None:
            return
        self.shm_map.seek(0)
        self.shm_map.write(struct.pack("i", status))
        self.shm_map.flush()

    def close(self):
        if self.shm_map:
            self.shm_map.close()
        if self.shm_fd:
            os.close(self.shm_fd)


def run_trace():
    symbol = os.environ.get("TRACE_SYMBOL")
    input_file = os.environ.get("TRACE_INPUT")

    if symbol is None or input_file is None:
        symbol = "tvmgen_default_fused_nn_contrib_conv2d_NCHWc_add_nn_relu_compute_"
        input_file = "/home/groundbreaker/DeepCache/experiment_llc/cat.bin"

    # Configure GDB
    gdb.execute("set confirm off")
    gdb.execute("set pagination off")
    gdb.execute("set breakpoint pending on")
    gdb.execute(f"set args {input_file}")

    # Set breakpoint at operator entry
    gdb.execute(f"break {symbol}")

    # Setup SHM
    shm = SharedMemoryController()
    shm.open()

    # Run to first breakpoint
    gdb.execute("run")

    # Check if process is valid
    if hasattr(gdb, "selected_inferior"):
        threads = gdb.selected_inferior().threads()
        if not threads:
            print("[GDB] Process exited before hitting breakpoint")
            shm.set_status(4)
            shm.close()
            sys.exit(0)

    # Precise sampling loop: only sample during operator execution
    # NLP dense 算子执行快，需要更多迭代来采集足够数据（demo_static 最多 500 次）
    NUM_ITERATIONS = 500
    completed = 0
    for _ in range(NUM_ITERATIONS):
        # Start sampling
        shm.set_status(3)

        # Small delay to ensure attacker starts sampling
        time.sleep(0.001)

        # Execute operator until it returns
        gdb.execute("finish")

        # Stop sampling
        shm.set_status(2)
        completed += 1

        # Check if process still running
        if hasattr(gdb, "selected_inferior"):
            threads = gdb.selected_inferior().threads()
            if not threads:
                print(f"[GDB] Process exited after {completed} iterations")
                break

        # Continue to next breakpoint (next inference's target operator)
        gdb.execute("continue")

        # Check again after continue
        if hasattr(gdb, "selected_inferior"):
            threads = gdb.selected_inferior().threads()
            if not threads:
                print(f"[GDB] Process exited, completed {completed} iterations")
                break

    print(f"[GDB] Finished {completed} iterations, signaling save...")

    # Signal attacker to save file and reset (status=4)
    shm.set_status(4)
    shm.close()

    # Kill process 
    if hasattr(gdb, "selected_inferior"):
        threads = gdb.selected_inferior().threads()
        if threads:
            gdb.execute("kill")
    gdb.execute("quit")


if __name__ == "__main__":
    run_trace()