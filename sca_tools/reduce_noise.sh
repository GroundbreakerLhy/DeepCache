#!/bin/bash

set -e

ACTION=${1:-start}

start_noise_reduction() {
    echo "启动降噪模式..."

    # 1. 禁用 CPU 频率缩放 - 固定到最高频率
    for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        if [ -f "$cpu" ]; then
            echo "performance" > "$cpu" 2>/dev/null || true
        fi
    done
    
    # 2. 禁用 Turbo Boost (减少频率波动)
    if [ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
        echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo
    fi
    
    # 3. 禁用超线程的兄弟核心
    for cpu in 1 3 5 7 9 11 13 15 16 17 18 19; do
        echo 0 > /sys/devices/system/cpu/cpu$cpu/online 2>/dev/null || true
    done

    # 3.5. 禁用硬件预取器
    modprobe msr 2>/dev/null || true
    for cpu in 0 2 4 6 8 10 12 14; do
        # MSR 0x1A4: Disable all prefetchers (L2 HW, L2 Adjacent, DCU, DCU IP)
        wrmsr -p $cpu 0x1A4 0xF 2>/dev/null || true
    done

    # 3.6. 将IRQ迁移离开CPU 0 (隔离测量核心)
    for irq in /proc/irq/*/smp_affinity; do
        echo 4 > $irq 2>/dev/null || true  # 0x4 = CPU 2
    done

    # 4. 设置进程调度优先级
    echo -1 > /proc/sys/kernel/sched_rt_runtime_us 2>/dev/null || true
    
    # 5. 禁用 ASLR (地址空间随机化)
    echo 0 > /proc/sys/kernel/randomize_va_space
    
    # 6. 隔离 CPU 核心
    if [ -d /sys/fs/cgroup/cpuset ]; then
        mkdir -p /sys/fs/cgroup/cpuset/sca_isolated 2>/dev/null || true
        # 为侧信道攻击隔离 CPU 0, 2, 4 (P-Cores)
        echo "0,2,4" > /sys/fs/cgroup/cpuset/sca_isolated/cpuset.cpus 2>/dev/null || true
        echo "0" > /sys/fs/cgroup/cpuset/sca_isolated/cpuset.mems 2>/dev/null || true
        echo 1 > /sys/fs/cgroup/cpuset/sca_isolated/cpuset.cpu_exclusive 2>/dev/null || true
    fi
    
    # 7. 停止不必要的系统服务
    SERVICES_TO_STOP=(
        "cron"
        "atd"
        "cups"
        "bluetooth"
        "avahi-daemon"
        "ModemManager"
        "NetworkManager-wait-online"
        "snapd"
        "packagekit"
        "unattended-upgrades"
    )
    
    for svc in "${SERVICES_TO_STOP[@]}"; do
        systemctl stop "$svc" 2>/dev/null || true
    done
    
    # 8. 清空页面缓存
    sync
    echo 3 > /proc/sys/vm/drop_caches
    
    # 9. 分配 Huge Pages 
    sysctl -w vm.nr_hugepages=1024 >/dev/null
    HP_COUNT=$(cat /proc/meminfo | grep HugePages_Free | awk '{print $2}')
    
    # 9. 禁用透明大页
    echo never > /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || true
    
    # 10. 设置内核定时器频率
    sysctl -w kernel.nmi_watchdog=0 2>/dev/null || true
}

stop_noise_reduction() {
    echo "停止降噪模式..."

    # 恢复 CPU 频率缩放
    for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        if [ -f "$cpu" ]; then
            echo "ondemand" > "$cpu" 2>/dev/null || echo "powersave" > "$cpu" 2>/dev/null || true
        fi
    done

    # 恢复 Turbo Boost
    if [ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
        echo 0 > /sys/devices/system/cpu/intel_pstate/no_turbo
    fi
    
    # 恢复超线程核心
    for cpu in 1 3 5 7 9 11 13 15 16 17 18 19; do
        echo 1 > /sys/devices/system/cpu/cpu$cpu/online 2>/dev/null || true
    done

    # 恢复硬件预取器
    if command -v wrmsr &> /dev/null; then
        modprobe msr 2>/dev/null || true
        echo "恢复硬件预取器..."
        for cpu in 0 2 4 6 8 10 12 14; do
            wrmsr -p $cpu 0x1A4 0x0 2>/dev/null || true
        done
    fi

    # 恢复 IRQ 亲和性 (所有CPU)
    echo "恢复中断分配..."
    for irq in /proc/irq/*/smp_affinity; do
        echo ff > $irq 2>/dev/null || true  # 所有CPU
    done
    
    # 恢复 ASLR
    echo 2 > /proc/sys/kernel/randomize_va_space
    
    # 恢复 NMI watchdog
    sysctl -w kernel.nmi_watchdog=1 2>/dev/null || true
    
    # 重启服务
    SERVICES_TO_START=(
        "cron"
        "NetworkManager-wait-online"
    )
    for svc in "${SERVICES_TO_START[@]}"; do
        systemctl start "$svc" 2>/dev/null || true
    done

    echo "=== 系统已恢复正常模式 ==="
}


case "$ACTION" in
    start)
        start_noise_reduction
        ;;
    stop)
        stop_noise_reduction
        ;;
    *)
        echo "用法: sudo $0 [start|stop]"
        echo "  start - 启用降噪模式"
        echo "  stop  - 恢复正常模式"
        exit 1
        ;;
esac
