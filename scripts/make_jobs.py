#!/usr/bin/env python3
"""Write one command per line for a parameter sweep (cartesian product), for slurm/array_cpu.sbatch or a plain loop.

    python scripts/make_jobs.py run_mpo.py --route direct split --n 64 --chi 128 256 --dt 0.05 0.025 > jobs/mpo.txt
    python scripts/make_jobs.py run_ed.py --L 12 14 16 18 20 22 24 --bc pbc obc --beta 0,0.5,1,2 > jobs/ed.txt

A value with commas is passed as one multi-valued option (--beta 0,0.5,1,2 -> --beta 0 0.5 1 2); use it only for
options that take several values (run_ed.py --L / --beta).  Flags without values (e.g. --profile) are copied to
every line.  --python sets the interpreter (default: python3).  Create jobs/ first: mkdir -p jobs logs results
"""
import itertools
import shlex
import sys


def main(argv):
    if not argv or argv[0].startswith('-'):
        print(__doc__)
        return 1
    script, rest = argv[0], argv[1:]
    if script.startswith('scripts/'):
        script = script[len('scripts/'):]
    if rest and not rest[0].startswith('--'):
        print(f'make_jobs: value {rest[0]!r} before the first option', file=sys.stderr)
        return 2
    py = 'python3'
    groups, flags, key = [], [], None
    for tok in rest:
        if tok.startswith('--'):
            key = tok
            if key == '--python':
                continue
            groups.append((key, []))
        elif key == '--python':
            py = tok
        else:
            groups[-1][1].append(tok)
    flags = [k for k, v in groups if not v]
    groups = [(k, v) for k, v in groups if v]
    for combo in itertools.product(*[v for _, v in groups]):
        parts = [py, f'scripts/{script}']
        for (k, _), val in zip(groups, combo):
            parts += [k] + val.split(',')
        parts += flags
        print(' '.join(shlex.quote(x) for x in parts))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
