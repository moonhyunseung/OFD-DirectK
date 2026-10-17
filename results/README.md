# Results

Only manifest-backed results belong here.

Recommended structure:

```text
results/<experiment_id>/<timestamp>/
  manifest.json
  raw/
  derived/
  summary.md
```

Each summary must separate measured values, derived values, interpretation, and the claim status affected. Negative/crossover-defining runs are retained when they constrain the hardware claim.