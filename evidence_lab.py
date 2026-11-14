"""Bastion Helix ROCm Evidence Lab. Local, reproducible retrieval microbenchmark."""
import argparse
import datetime as dt
import hashlib
import html
import json
import platform
import os
import statistics
import time
from pathlib import Path
import numpy as np


def topk(scores, k):
    return np.argsort(-scores, axis=1, kind='stable')[:, :k]


def quality(reference, candidate, k):
    # Validate one query at a time to bound sorting and comparison scratch space.
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError('Score matrices must have matching two-dimensional shapes')
    if not 1 <= k <= reference.shape[1]:
        raise ValueError('k must be between 1 and the document count')
    error, agreement, close = 0.0, 0.0, True
    for ref, got in zip(reference, candidate):
        a, b = topk(ref[None, :], k)[0], topk(got[None, :], k)[0]
        error = max(error, float(np.max(np.abs(ref-got))))
        agreement += len(set(a) & set(b)) / k
        close = close and bool(np.allclose(ref, got, atol=1e-4, rtol=1e-3))
    return {'max_abs_error': error, 'recall_at_k': agreement / len(reference),
            'scores_close': close}


def memory_plan(args):
    vectors = 4 * (args.docs + args.queries) * args.dim
    scores = 4 * args.docs * args.queries
    # Resident vectors, overlapping score matrices, one-row sort/validation,
    # bounded float64 generation scratch, and runtime headroom. An estimate,
    # not a guarantee: BLAS/PyTorch workspace and other processes can vary.
    host = vectors + 3 * scores + 40 * args.docs + 16 * min(4096, max(args.docs, args.queries)) * args.dim + 256 * 2**20
    device = vectors + 2 * scores + 256 * 2**20 if args.backend == 'rocm' else 0
    return {'estimated_host_mib': host / 2**20,
            'estimated_device_mib': device / 2**20,
            'corpus_mib': 4 * args.docs * args.dim / 2**20}


def normalized_vectors(rng, rows, dim):
    out = np.empty((rows, dim), dtype='float32')
    for start in range(0, rows, 4096):
        block = out[start:start+4096]
        block[:] = rng.normal(size=block.shape)
        block /= np.linalg.norm(block, axis=1, keepdims=True)
    return out


def run(args):
    if min(args.docs,args.queries,args.dim,args.repeats,args.batch,args.k) < 1:
        raise ValueError('All sizes, repeats and k must be positive')
    if args.k > args.docs or args.warmup < 0:
        raise ValueError('k must not exceed docs; warmup must be nonnegative')
    plan = memory_plan(args)
    budget = getattr(args, 'memory_budget_mib', 4096)
    if budget <= 0 or max(plan['estimated_host_mib'], plan['estimated_device_mib']) > budget:
        raise ValueError(f'Memory estimate exceeds per-memory-space budget {budget} MiB: {plan}')
    rng=np.random.default_rng(args.seed)
    docs=normalized_vectors(rng, args.docs, args.dim)
    queries=normalized_vectors(rng, args.queries, args.dim)
    digest=hashlib.sha256()
    digest.update(memoryview(docs).cast('B'))
    digest.update(memoryview(queries).cast('B'))
    fingerprint=digest.hexdigest()
    reference=queries@docs.T
    sync=lambda:None
    meta={'backend':args.backend,'python':platform.python_version(),'numpy':np.__version__,
          'platform':platform.platform(),'thread_environment':{key:os.environ.get(key) for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},'device':'CPU','dtype':'float32','gpu_verified':False}
    torch=None
    if args.backend=='rocm':
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError('Install a compatible ROCm PyTorch build using the AMD guide in README.') from exc
        if not torch.version.hip or not torch.cuda.is_available():
            raise RuntimeError('ROCm GPU required. No CPU fallback is permitted for --backend rocm.')
        torch.backends.cuda.matmul.allow_tf32=False
        device=torch.device('cuda:0')
        free_bytes, _ = torch.cuda.mem_get_info(device)
        if plan['estimated_device_mib'] * 2**20 > free_bytes:
            raise RuntimeError('Estimated device memory exceeds currently free GPU memory')
        d=torch.from_numpy(docs).to(device)
        q=torch.from_numpy(queries).to(device)
        sync=torch.cuda.synchronize
        meta.update(device=torch.cuda.get_device_name(0),torch=torch.__version__,hip=torch.version.hip,gpu_verified=True)
        def score(batch):
            with torch.inference_mode():
                return torch.cat([q[i:i+batch]@d.T for i in range(0,args.queries,batch)])
        def host(x):return x.cpu().numpy()
    else:
        def score(batch):
            return np.concatenate([queries[i:i+batch]@docs.T for i in range(0,args.queries,batch)])
        def host(x):return x
    batches=sorted(set([1,args.batch]))
    samples={b:[] for b in batches}
    for b in batches:
        for _ in range(args.warmup):score(b)
        sync()
    # Rotate order each round to reduce order/thermal bias.
    for r in range(args.repeats):
        for b in batches[r%len(batches):]+batches[:r%len(batches)]:
            sync()
            start=time.perf_counter()
            out=score(b)
            sync()
            samples[b].append((time.perf_counter()-start)*1000)
            del out
    results=[]
    for b in batches:
        out=host(score(b))
        check=quality(reference,out,args.k)
        del out
        median=statistics.median(samples[b])
        results.append({'batch':b,'median_ms':median,'p95_ms':float(np.percentile(samples[b],95)),
                        'queries_per_second':args.queries*1000/median,'samples_ms':samples[b],
                        **check,'accepted':check['scores_close'] and check['recall_at_k']>=0.99})
    base=results[0]['median_ms']
    for item in results:item['speedup_vs_batch1']=base/item['median_ms']
    eligible=[r for r in results if r['accepted']]
    best=min(eligible,key=lambda x:x['median_ms'])['batch'] if eligible else None
    return {'schema_version':2,'project':'Bastion Helix — ROCm Evidence Lab',
            'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'environment':meta,
            'memory_plan':plan,'config':vars(args),'dataset_sha256':fingerprint,'results':results,'recommended_batch':best,
            'measurement_scope':'In-memory cosine score matrix computation, concatenation and host dispatch; GPU synchronization included. Excludes embedding, transfer, top-k selection and request queueing.',
            'limitations':['Synthetic normalized vectors; no semantic relevance evaluation.',
                          'Batch throughput is not individual online request latency.',
                          'CPU and ROCm use different libraries; compare batching within each backend.',
                          'No energy, cost, end-to-end RAG, or model-quality claim.']}


def report(data, path):
    esc=lambda s:html.escape(str(s))
    env=data['environment']
    base=data['results'][0]['median_ms']
    rows=''.join(f"<tr><td>{r['batch']}</td><td>{r['median_ms']:.3f}</td><td>{r['p95_ms']:.3f}</td><td>{r['queries_per_second']:,.0f}</td><td>{r['recall_at_k']:.1%}</td><td>{r['max_abs_error']:.2e}</td><td>{'PASS' if r['accepted'] else 'FAIL'}</td></tr>" for r in data['results'])
    bars=''.join(f"<div class='barrow'><span>Batch {r['batch']}</span><div class='track'><div style='width:{100*r['median_ms']/max(x['median_ms'] for x in data['results']):.2f}%'></div></div><b>{r['median_ms']:.3f} ms</b></div>" for r in data['results'])
    badge='MEASURED AMD ROCm RUN' if env['gpu_verified'] else 'MEASURED CPU RUN · AMD GPU VALIDATION PENDING'
    best=next((r for r in data['results'] if r['batch']==data['recommended_batch']),None)
    speed=f"{best['speedup_vs_batch1']:.2f}×" if best else 'No pass'
    content="""<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>ROCm Evidence Lab | Bastion Helix</title><style>
    :root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#0a101c;color:#edf3fd;font:16px system-ui,sans-serif}main{max-width:1120px;margin:auto;padding:48px 26px}nav{color:#77e5c6;letter-spacing:.15em;font-size:13px}h1{font-size:clamp(36px,6vw,68px);line-height:1.05;max-width:850px;margin:30px 0 18px}p{color:#b8c7dd;line-height:1.7}.badge{display:inline-block;background:#233c43;color:#a4f0d5;padding:9px 13px;border-radius:5px;font-size:12px;margin-top:22px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:30px 0}.card,section{border:1px solid #29374b;border-radius:14px;background:#111c2e;padding:25px}.big{font-size:42px;font-weight:700;margin:12px 0}.muted{font-size:13px;color:#a1b3ca}section{margin:20px 0}h2{font-size:22px;margin-top:0}.barrow{display:flex;align-items:center;gap:16px;margin:25px 0}.barrow span{width:90px}.barrow b{min-width:115px;text-align:right}.track{flex:1;background:#253047;border-radius:6px;overflow:hidden}.track div{height:20px;background:linear-gradient(90deg,#38b9b1,#b7edd1)}table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:14px 9px;border-bottom:1px solid #29374b;text-align:left}.scroll{overflow:auto}li{color:#b8c7dd;margin:12px 0}code{overflow-wrap:anywhere;color:#a4f0d5}summary{cursor:pointer;padding:10px}pre{white-space:pre-wrap;overflow-wrap:anywhere;color:#b8c7dd}footer{font-size:13px;color:#93a6c2;padding:24px 0}@media(max-width:650px){.grid{grid-template-columns:1fr}.barrow{gap:8px;font-size:12px}main{padding:25px 16px}}
    </style><main><nav>BASTION HELIX / ENGINEERING EVIDENCE</nav><div class='badge'>BADGE</div><h1>Faster retrieval.<br>Show the evidence.</h1><p>A reproducible experiment in batching cosine similarity workloads.<br>Same vectors. Same precision. Accuracy checked before a recommendation.</p><div class='grid'><div class='card'><div class='muted'>FASTEST ACCEPTED RESULT</div><div class='big'>SPEED</div><div class='muted'>relative to batch 1 on this device</div></div><div class='card'><div class='muted'>RECOMMENDED BATCH</div><div class='big'>BEST</div><div class='muted'>lowest measured median among passing candidates</div></div><div class='card'><div class='muted'>EXECUTION DEVICE</div><h2>DEVICE</h2><div class='muted'>float32 · BACKEND</div></div></div><section><h2>Measured time to score the query set</h2><p>Lower is better. Raw repetitions are included below.</p>BARS</section><section><h2>Performance with a correctness gate</h2><div class='scroll'><table><thead><tr><th>Batch</th><th>Median ms</th><th>P95 ms</th><th>Queries/s</th><th>Recall@K</th><th>Max error</th><th>Check</th></tr></thead><tbody>ROWS</tbody></table></div></section><section><h2>What this experiment measures</h2><p>SCOPE</p><ul>LIMITS</ul><p>Score tolerance: atol 1e-4, rtol 1e-3. Top-k set agreement must be ≥99%. This checks numerical agreement, not semantic relevance.</p></section><section><h2>Reproduce, inspect, challenge</h2><p>Dataset SHA-256</p><code>HASH</code><details><summary>Environment, configuration and raw measurements</summary><pre>RAW</pre></details></section><footer>Independent Bastion Helix project. Not endorsed by AMD. No telemetry, API key or model download required. Generated DATE.</footer></main></html>"""
    values={'BADGE':badge,'SPEED':speed,'BEST':data['recommended_batch'],'DEVICE':env['device'],'BACKEND':env['backend'],'BARS':bars,'ROWS':rows,'SCOPE':esc(data['measurement_scope']),'LIMITS':''.join('<li>'+esc(s)+'</li>' for s in data['limitations']),'HASH':data['dataset_sha256'],'RAW':esc(json.dumps(data,indent=2)),'DATE':esc(data['created_utc'])}
    import re
    content=re.sub(r'\b('+'|'.join(values)+r')\b',lambda m:str(values[m.group()]),content)
    Path(path).write_text(content)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=['cpu','rocm'],default='cpu')
    for name,default in [('docs',4096),('queries',64),('dim',384),('batch',32),('repeats',15),('warmup',3),('seed',42),('k',10)]:
        p.add_argument('--'+name,type=int,default=default)
    p.add_argument('--memory-budget-mib',type=int,default=4096)
    p.add_argument('--output',default='results')
    args=p.parse_args()
    try:data=run(args)
    except (ValueError,RuntimeError,MemoryError) as exc:p.exit(2,str(exc)+'\n')
    dest=Path(args.output); dest.mkdir(parents=True,exist_ok=True)
    (dest/'measurements.json').write_text(json.dumps(data,indent=2))
    report(data,dest/'report.html')
    print(json.dumps({'report':str(dest/'report.html'),'backend':args.backend,'recommended_batch':data['recommended_batch']}))
    return 0 if data['recommended_batch'] is not None else 1

if __name__=='__main__':raise SystemExit(main())
