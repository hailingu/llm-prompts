# Data Engineering Best Practices

This document defines baseline standards for data engineering work in this repository.

## Scope

- Data ingestion and extraction
- Data cleaning and normalization
- Data quality validation
- Train/validation/test split strategy
- Feature-ready dataset production
- Reproducible pipeline execution

## Core Principles

1. Reproducibility first: every transformation step must be traceable and repeatable.
2. Data quality gates: validate schema, null ratio, value ranges, and uniqueness before downstream use.
3. Leakage prevention: enforce split strategy before feature engineering and modeling.
4. Contract-driven outputs: artifacts should have explicit schema and ownership.
5. Fail fast on critical data defects; do not silently patch unknown anomalies.

## Minimum Checklist

- [ ] Source data schema is documented
- [ ] Data quality checks are defined and executed
- [ ] Split method is justified (random/stratified/time/group)
- [ ] Output artifacts and paths are explicit
- [ ] Pipeline commands are reproducible in CI/local environments

## Related Standards

- `cheat-sheet.md`
- `feature-engineering-patterns.md`
- `model-monitoring-guide.md`
- `experimentation-design-guide.md`
