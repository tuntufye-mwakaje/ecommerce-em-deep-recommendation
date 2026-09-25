# Source Code

The `src/` directory contains reusable Python modules extracted from
the implemented customer analytics and clustering workflow.

## Structure

```text
src/
+-- __init__.py
+-- data/
¦   +-- __init__.py
¦   +-- loader.py
+-- preprocessing/
¦   +-- __init__.py
¦   +-- cleaning.py
¦   +-- scaling.py
+-- features/
¦   +-- __init__.py
¦   +-- customer_features.py
+-- clustering/
    +-- __init__.py
    +-- gmm.py
```

## Modules

- `data/loader.py` - CSV loading and file-existence validation
- `preprocessing/cleaning.py` - transaction cleaning and validation
- `preprocessing/scaling.py` - customer-feature standardization using `StandardScaler`
- `features/customer_features.py` - customer-level behavioral feature construction
- `clustering/gmm.py` - GMM model evaluation, final model fitting, cluster assignment, and cluster profiling
