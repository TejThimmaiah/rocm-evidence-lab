# Bastion Helix — ROCm Evidence Lab

**Measure batching gains. Check correctness. Publish reproducible evidence.**

A working developer-tool prototype for the cosine-scoring stage of vector retrieval. It compares one-query-at-a-time execution against batches, checks scores and top-k agreement, and generates a self-contained HTML report plus raw JSON. Built for an AMD Developer Community showcase; selection is not guaranteed.

## Current evidence

The bundled report is a real **CPU-only** run, not a simulation and not an AMD benchmark. Batch 32 measured approximately 1.94× the throughput of batch 1 on this run. Both passed score tolerance and top-k agreement checks. Timings vary by machine and system load. The code has a ROCm execution path, but **that path has not yet been executed or validated on an AMD GPU**. No claim of production readiness, novel algorithm, or AMD endorsement is made.

## Quick start (Python 3.10+)

```sh
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows instead: .venv\Scripts\activate
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 python evidence_lab.py --backend cpu --output results-cpu
python -m unittest -v
```

On Windows PowerShell set `$env:OPENBLAS_NUM_THREADS="1"` then run the Python command without the environment prefix. Open `results-cpu/report.html` in a browser. No web server, API account, paid model or network connection is needed after installation.

## Run on AMD

Use your own supported AMD machine or an authorized AMD Developer Cloud environment. Do not use a certification lab for unrelated workloads unless its terms permit it. Cloud access, quotas and charges depend on your account; this project does not provision compute.

1. Match your GPU, operating system and ROCm/PyTorch build against AMD's current compatibility documentation.
2. Install the matching ROCm PyTorch build using AMD's official instructions. Installing generic `torch` alone does not establish ROCm support.
3. Install NumPy and copy this folder into that environment.
4. Run:

```sh
python evidence_lab.py --backend rocm --docs 4096 --queries 64 --dim 384 --batch 32 --repeats 30 --warmup 5 --output results-rocm
```

The command refuses to silently fall back to CPU. It requires both a HIP-enabled PyTorch build and an available device. `torch.cuda` is the API name used by ROCm PyTorch too. Device synchronization surrounds wall-clock timing; input transfer and accuracy validation occur outside timed sections.

For larger experiments, increase `--docs` gradually and record every configuration. The tool applies conservative size limits. If memory is exhausted, reduce docs, queries or dimension. Do not change results or mark a failed quality check as passing.

## What is implemented

- Seeded, normalized float32 synthetic embeddings with a SHA-256 dataset fingerprint.
- NumPy CPU and optional PyTorch ROCm implementations of cosine scoring.
- Batch-1 baseline and configurable batched candidate, including partial final batches.
- Warmups, alternating execution order, individual samples, median and p95.
- Numerical tolerance check and top-k set agreement against CPU reference.
- Recommendation restricted to passing candidates; it can recommend batch 1.
- Portable HTML report, raw measurements, local notebook and tests.

## Methodology and limits

The experiment scores all queries against all documents in memory. It includes matrix multiplication, concatenation, Python dispatch and synchronization, but excludes generating embeddings, host-to-device transfer, top-k selection and queueing. P95 is across repeated whole-workload runs, not production request latency. Throughput is queries divided by total run time. Candidate and baseline use the same backend and dtype.

Correctness requires `allclose(atol=1e-4, rtol=1e-3)` and mean top-k set agreement ≥99%. Near-ties can change rankings; inspect failures rather than loosening tolerances to create a favorable result. Synthetic random vectors test numerical consistency, not whether search results are meaningful. This is not an end-to-end RAG benchmark or proof that batching always helps. Whole score matrices are materialized, so it is not a billion-document retrieval engine.

Two automated tests exercise corrupted rankings, remainder batches, deterministic datasets, sample counts and invalid k. A complete CPU run also passed. GPU execution remains unverified. Benchmark values are not signed or tamper-proof: dataset hashes support reproduction, not independent attestation.

## Next milestones

1. Run three independent ROCm experiments on the same GPU with 30 repetitions each; retain all runs, including regressions.
2. Sweep at least three query counts and batch sizes. Explain throughput versus waiting-time tradeoffs.
3. Extend the benchmark to a licensed public document corpus and a fixed open embedding model; report retrieval quality separately from numeric agreement.
4. Record a 90-second screen demo: environment → run → correctness → report → limitations.
5. Publish source, setup instructions, hardware/software versions and raw results in a repository you control.
6. Update the Showcase discussion with verified AMD results.

Showcase discussion: https://devcommunity.amd.com/t/bastion-helix-rocm-evidence-lab-correctness-first-vector-retrieval-benchmarking/1382

## Product direction

An evidence-producing performance companion for developers moving retrieval and AI services onto AMD. Subsequent milestones: real embedding inputs; chunked top-k to bound memory; paired repeated-trial confidence intervals; validated FP16/BF16 accuracy tradeoffs; profiler trace export; optional read-only MCP tools to inspect saved run results. These are roadmap items, not implemented features.

## Sources

- PyTorch HIP semantics: https://docs.pytorch.org/docs/stable/notes/hip.html
- Synchronization and timing: https://docs.pytorch.org/docs/stable/notes/cuda.html
- AMD PyTorch installation: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html
- AMD developer showcase: https://devcommunity.amd.com/c/general/showcase/19

Independent Bastion Helix project. AMD and ROCm names identify the target platform only.
