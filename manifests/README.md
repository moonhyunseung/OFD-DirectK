# Manifests

Every durable experiment should carry a machine-readable manifest containing at least:

- repository commit SHA;
- experiment ID;
- command/config;
- geometry (`L`, batch, GQA/query-head ratio, group size);
- dtype/accumulation policy;
- hardware and software stack;
- model/checkpoint revision if applicable;
- raw result paths;
- analysis script/path.

Manifests are part of the evidence record, not optional metadata.