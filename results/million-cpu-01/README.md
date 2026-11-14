# Million-vector CPU validation

Executed with commit bcf76dec07be9543eb86f391a55873c1a79572a2. CPU identified by /proc/cpuinfo as AMD EPYC 7763 64-Core Processor; execution was in a hosted environment, not a controlled dedicated-machine benchmark. OPENBLAS_NUM_THREADS=1. GPU unavailable.

Command:

```sh
OPENBLAS_NUM_THREADS=1 python sweep.py --sizes 1000000 --batches 32 --queries 64 --dim 384 --repeats 3 --warmup 1 --output validation-million-cpu-01
```

One million synthetic float32 document vectors, dimension 384, 64 queries, k=10. Both batch sizes passed score tolerance and 100% mean top-k set agreement. Median whole-workload times: batch 1 = 8237.670 ms, batch 32 = 1834.898 ms (4.49x relative throughput). Raw timings and environment are in measurements.json; download report.html to view locally.

This is a three-repeat validation trial, not a stable performance estimate or an AMD GPU result. P95 from three samples is particularly weak evidence. Runtime estimates are not measured peak memory. No CPU/GPU crossover was measured. Run more repetitions and independent trials on controlled hardware before making performance claims.
