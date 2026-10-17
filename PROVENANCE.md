# Provenance

## Research origin

`OFD-DirectK` is a focused branch of the broader OFD research line. It isolates the K-side packed execution claim so that algebra, implementation, measurement, and interpretation can be audited independently.

## Provenance rule

Every durable research artifact should state whether it is:

- **USER / RESEARCHER DERIVATION** — originated from the researcher's own mathematical or systems reasoning;
- **ASSISTANT SYNTHESIS** — organized, reformulated, or drafted by an AI assistant from supplied context;
- **MEASURED RESULT** — produced by executable code under a recorded manifest;
- **EXTERNAL FACT** — imported from documentation, papers, specifications, or source code and should carry a citation/reference.

Assistant-generated prose is not automatically evidence. A claim becomes part of the research record only when its proof, implementation, source, or measurement is independently inspectable.

## Historical separation

Results from the original `moonhyunseung/OFD` repository are predecessor evidence only unless explicitly imported with a source commit SHA and revalidated under this repository's semantic contract.

## Required metadata for imported evidence

- source repository;
- source commit SHA;
- source file/result path;
- semantic compatibility note;
- whether revalidation was performed.
