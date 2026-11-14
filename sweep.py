"""Run corpus/batch combinations in isolated processes; retain every outcome."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['cpu', 'rocm'], default='cpu')
    p.add_argument('--sizes', nargs='+', type=int, default=[4096, 32768, 131072, 1000000])
    p.add_argument('--batches', nargs='+', type=int, default=[1, 8, 32, 64])
    p.add_argument('--queries', type=int, default=64)
    p.add_argument('--dim', type=int, default=384)
    p.add_argument('--repeats', type=int, default=30)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--memory-budget-mib', type=int, default=4096)
    p.add_argument('--timeout', type=int, default=1800)
    p.add_argument('--output', default='sweep-results')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if min(a.sizes + a.batches + [a.queries, a.dim, a.repeats, a.memory_budget_mib, a.timeout]) < 1 or a.warmup < 0:
        p.error('sizes and limits must be positive; warmup must be nonnegative')
    root = Path(a.output)
    # Refuse overwrites so previous trials and failures remain available.
    root.mkdir(parents=True, exist_ok=False)
    outcomes = []
    for size in dict.fromkeys(a.sizes):
        for batch in dict.fromkeys(a.batches):
            dest = root / f'docs-{size}-batch-{batch}'
            cmd = [sys.executable, str(Path(__file__).with_name('evidence_lab.py')), '--backend', a.backend,
                   '--docs', str(size), '--batch', str(batch), '--output', str(dest)]
            for key in ['queries', 'dim', 'repeats', 'warmup', 'memory_budget_mib']:
                cmd += ['--' + key.replace('_', '-'), str(getattr(a, key))]
            entry = {'docs': size, 'batch': batch, 'command': cmd}
            if a.dry_run:
                from evidence_lab import memory_plan
                shape = argparse.Namespace(docs=size, queries=a.queries, dim=a.dim, backend=a.backend)
                plan = memory_plan(shape)
                entry.update(status='planned', memory_plan=plan,
                             within_budget=max(plan['estimated_host_mib'], plan['estimated_device_mib']) <= a.memory_budget_mib)
            else:
                dest.mkdir()
                with (dest/'run.log').open('w') as log:
                    try:
                        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=a.timeout)
                        entry.update(status='passed' if result.returncode == 0 else 'failed', returncode=result.returncode)
                    except subprocess.TimeoutExpired:
                        entry.update(status='timeout')
                entry['log'] = str(dest/'run.log')
            outcomes.append(entry)
            (root/'summary.json').write_text(json.dumps(outcomes, indent=2))
            print(f'{size} vectors / batch {batch}: {entry["status"]}', flush=True)
    return int(any(x['status'] in ('failed', 'timeout') for x in outcomes))


if __name__ == '__main__':
    raise SystemExit(main())
