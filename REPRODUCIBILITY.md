# Reproducibility

Every empirical DirectK result must record:

- repository commit SHA;
- experiment ID and exact command/config;
- hardware model/count;
- driver/CUDA/PyTorch/Triton/compiler versions as applicable;
- model/checkpoint revision when used;
- dtype and accumulation policy;
- context/batch/GQA geometry;
- warmup/repetition protocol;
- raw output path;
- analysis script/path and SHA.

Recommended result layout:

```text
results/<experiment_id>/<timestamp>/
  manifest.json
  raw/
  derived/
  summary.md
```

Negative results are retained when they constrain the hardware claim boundary. No profiler screenshot or copied terminal value is a durable result without a manifest.