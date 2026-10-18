# MLOps Week 2 - DVC for the IRIS pipeline

Data and model versioning with DVC, using a Google Cloud Storage bucket as the remote.

## Files
| File | Purpose |
|---|---|
| `train.py` | Trains a Decision Tree (Week 1 reference notebook code) on `data/iris.csv`; saves `artifacts/model.joblib` and `metrics.json` |
| `data/v1.dvc`, `data/v2.dvc` | `dvc import` records: source repo (IITMBSMLOps/ga_resources, branch week_1) and the exact commit of each data batch |
| `data/iris.csv.dvc` | DVC pointer to the training data version |
| `artifacts/model.joblib.dvc` | DVC pointer to the trained model version |
| `metrics.json` | Row count and test accuracy of the model at this version |
| `.dvc/config` | DVC config; default remote = `gs://21f2000602-dvc-week2/dvc-store` |
| `.dvcignore`, `data/.gitignore`, `artifacts/.gitignore` | Keep data and model files out of Git |

## Versions
| Tag | Data | Rows | Accuracy |
|---|---|---|---|
| v1.0 | v1 | 101 | 0.951 |
| v2.0 | v1 + v2 | 150 | 0.917 |

## Reproduce
    pip install dvc dvc-gs scikit-learn pandas joblib
    dvc pull                              # fetch data + model for the current commit
    git checkout v1.0 && dvc checkout     # switch to version 1
    git checkout week_2 && dvc checkout   # back to latest
