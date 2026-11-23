# Contributing

Thanks for your interest in `energy-to-quality`.

## Scope

This repository is a **reproducible proof-of-protocol benchmark** for measuring `Joules_to_Q_min` (energy to reach a target quality).  
Please keep contributions aligned with that goal: honest measurement, reproducibility, and clear limitations.

## Good contributions

- Bug fixes in scripts or reporting
- Better portability (Linux/RAPL environments, GPU/NVML adapters, external meters)
- New tasks/models *with the same protocol discipline*
- Documentation improvements
- Reproducibility reports (include hardware, OS, load, raw JSON)

## Before opening a PR

1. Keep comparisons fair: same task/split/Q_min/oracle, warmup excluded, interleaved A/B when possible.
2. Do not claim isolated physical proof from package-level RAPL alone.
3. Include raw JSON artifacts for benchmark claims.
4. Run a quick syntax check:

```bash
python3 -m py_compile scripts/run_champion.py scripts/run_proof.py
```

## Reporting results

When sharing benchmark results, include:

- Hardware/CPU, OS, kernel, governor/turbo settings
- RAPL path and whether wraparound handling was used
- Background/load conditions
- Number of repeats, seeds/splits, Q_min
- Raw JSON output

## Code of conduct

Be technical, respectful, and evidence-based. Claims should be reproducible.
