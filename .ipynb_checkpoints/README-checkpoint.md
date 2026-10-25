# MLOps Weekly Assignment — Roll No. 21f2000602

| Branch | Topic |
|---|---|
| `week_2` | Data & model versioning with DVC (GCS remote) |
| `week_3` | **Feast feature store for the IRIS pipeline** (this README) |

---

## Week 3 — Integrating the Feast Feature Store

### Goal
Training and inference must use the **same** feature definitions, so the model never receives
differently-computed features in production (*training–serving skew*). In this branch, the model
no longer reads the CSV for its inputs. Training pulls features from Feast's **offline store**, and
inference pulls them from Feast's **online store**. Both read one registered feature view.

```
iris_data_adapted_for_feast.csv   (DVC-imported from IITMBSMLOps/ga_resources @ week_3)
            │  converted to Parquet (UTC timestamps)
            ▼
   Feast feature repo  ──  iris_features.py : Entity `iris` + Source + FeatureView `iris_features`
      ┌──────────────┴────────────────┐
 OFFLINE store                         ONLINE store (SQLite)
 Parquet file  /  BigQuery (Task 6)    ◄── feast materialize (latest value per iris_id)
 get_historical_features()             get_online_features()
      │  point-in-time join                 │  lookup by iris_id
   TRAINING (Decision Tree)              INFERENCE (IrisPredictor)
```

### Environment
- **Vertex AI Workbench** instance `mlops-week3` (`e2-standard-2`, `us-central1-a`, idle shutdown 60 min), GCP project `arcane-sentinel-252018`
- Feast **0.67.0**, scikit-learn 1.9.1, NumPy 2.5.4, DVC with GCS remote `gs://21f2000602-dvc-week2/dvc-store`

### Dataset
45 rows = 3 iris plants (`iris_id` 1001–1003) × 15 daily measurements (2025-09-17 → 2025-10-01, UTC).
Plant 1001 is *versicolor*; plants 1002 and 1003 are *setosa*.

| Column | Feast role |
|---|---|
| `iris_id` | Entity join key |
| `event_timestamp` | Timestamp field: when the measurement was taken (used for point-in-time joins) |
| `created_timestamp` | Created timestamp: when the row was written (tie-breaker for duplicate events) |
| `sepal_length`, `sepal_width`, `petal_length`, `petal_width` | Features (Float64) |
| `species` | **Label**: passed through the entity dataframe, never served as a feature |

### Repository structure
| Path | Purpose |
|---|---|
| `week3_feast_iris.ipynb` | Main notebook. Runs Tasks 1–6 end to end, with outputs |
| `iris_feature_store/feature_repo/feature_store.yaml` | Task 1 config: project `iris_feature_store`, file registry, **file (Parquet) offline store**, **SQLite online store** |
| `iris_feature_store/feature_repo/iris_features.py` | Task 2: entity `iris` (`iris_id`), `FileSource` (`event_timestamp`, `created_timestamp`), feature view `iris_features` |
| `iris_feature_store/feature_repo/data/iris_data_adapted_for_feast.csv.dvc` | DVC import record of the dataset (source repo, `week_3`, exact commit) |
| `iris_feature_store/feature_repo_bq/feature_store.yaml` | Task 6 config: `provider: gcp`, **BigQuery offline store**, SQLite online store |
| `iris_feature_store/feature_repo_bq/iris_features_bq.py` | Task 6: the same entity and feature view, with a `BigQuerySource` |
| `iris_feature_store/feature_repo*/.gitignore` | Keeps generated `registry.db`, `online_store.db` and the Parquet copy out of Git |
| `artifacts/iris_feast_model.joblib.dvc` | DVC pointer to the model trained on Feast offline features |
| `week3_metrics.json` | Accuracy, offline/online consistency checks, latencies, BigQuery parity |
| `.dvc/`, `.dvcignore` | DVC setup and GCS remote (from Week 2, reused) |

Files from Week 2 (`train.py`, `metrics.json`, `data/*.dvc`, `artifacts/model.joblib.dvc`) are still
present because `week_3` was branched from `week_2`.

**Not committed** (generated or binary): `registry.db`, `online_store.db`, `iris_data.parquet`,
the CSV itself (DVC), and the model file (DVC).

### Feature definitions
| Object | Definition |
|---|---|
| Entity | `iris`, join key `iris_id` (INT64) |
| Data source | `FileSource(path="data/iris_data.parquet", timestamp_field="event_timestamp", created_timestamp_column="created_timestamp")` |
| Feature view | `iris_features`: 4 × Float64, `ttl = 5 years`, `online=True` |

**Why a timestamp?** Feast answers *"what was this entity's value as of time T?"*. Every row needs an
event time for point-in-time joins (no future data leaks into training) and for picking the latest
value during materialization. The standard Iris dataset has no time or entity ID; this course
dataset adds both. For a truly static dataset, one would add a row ID as the entity and a constant
collection timestamp.

**Why TTL = 5 years?** TTL is the maximum age of a value relative to the requested time. The data is
a fixed 2025 snapshot, so a short TTL would treat it as expired.

### How the tasks were done
| Task | What was done | Evidence in notebook |
|---|---|---|
| 1 Init | `feast init iris_feature_store`; removed the driver example; explicit `offline_store: file` + `online_store: sqlite` | config printed |
| 2 Define | `dvc import` dataset → Parquet (UTC) → `iris_features.py` | dtypes, `feast entities/feature-views list` |
| 3 Apply + materialize | `feast apply`; `feast materialize 2025-09-01T00:00:00 2025-10-02T00:00:00` | SQLite table holds 12 rows (3 plants × 4 features), `event_ts = 2025-10-01`; state `AVAILABLE_ONLINE` |
| 4 Offline → train | Entity df = (`iris_id`, `event_timestamp`, `species`) only; features come from `get_historical_features`; Decision Tree (depth 3, 60/40 stratified split) | 45 rows returned; identical to raw; point-in-time demo |
| 5 Online → infer | `IrisPredictor` loads the saved model, fetches features by `iris_id` with `get_online_features`, predicts | online features and predictions == raw latest rows |
| 6 BigQuery | Data loaded to `arcane-sentinel-252018.feast_iris.iris_source`; separate repo `feature_repo_bq` | parity checks + trade-off table |

### Results (`week3_metrics.json`)
| Check | Result |
|---|---|
| Training rows from offline store | **45** |
| Offline features identical to raw data | **True** |
| Test accuracy (18 test rows) | **1.000** |
| Online predictions for 1001 / 1002 / 1003 | **versicolor / setosa / setosa** (= true species) |
| Online features == latest raw rows | **True** |
| Online predictions == raw-data predictions | **True**, so no training–serving skew |
| Online lookup + prediction latency (3 plants) | **22.5 ms** |

*On the 100% accuracy:* only two species are present (setosa vs versicolor, which are trivially
separable), and the rows are repeated measurements of the same three plants, so test rows closely
resemble training rows. The goal of this assignment is consistent feature sourcing, not model quality.

**Point-in-time join demo (plant 1001):**
| Requested time | Returned | Why |
|---|---|---|
| 2025-09-20 18:00 | Sep 20 values (5.45) | latest measurement at or before 18:00 |
| 2025-09-21 09:00 | Sep 20 values (5.45), **not** Sep 21's 5.65 | Sep 21's reading was taken at 10:40, which is in the future at 09:00 |
| 2025-09-10 09:00 | file store: row **dropped**; BigQuery: row kept with **NULLs** | no measurement existed yet |
| plant 9999 | NaN | unknown entity |

### Task 6 (optional): BigQuery offline store
**Configuration changes:** a new repo `feature_repo_bq/` with `provider: gcp` and
`offline_store: {type: bigquery, project_id: arcane-sentinel-252018, dataset: feast_iris, location: us-central1}`.
The online store is still SQLite. The data source changed from `FileSource` to
`BigQuerySource(table="arcane-sentinel-252018.feast_iris.iris_source")`. The feature view, feature
list, training code and inference code are **unchanged**.

**Parity:** 45 rows; features identical to the file store; accuracy 1.000; online predictions identical.

| | Local file (Parquet) | BigQuery |
|---|---|---|
| Setup | None | Enable API, create dataset, load table, IAM |
| Offline retrieval (45 rows) | **0.14 s** | **8.68 s** (network + job overhead) |
| Scale | One VM's memory | Petabyte-scale, computed in the warehouse |
| Sharing | This VM only | Any user or service with IAM access |
| Cost | Free | Per query and storage (here within free tier) |
| Point-in-time join | pandas in memory | SQL in BigQuery (temp tables in `feast_iris`) |
| Entity row with no prior data | Dropped | Kept with NULLs |
| Best for | Learning, small data, local dev | Production-scale training data |

BigQuery suits the **offline** store only. Its ~seconds per query is fine for batch training but too
slow for online serving, which is why the online store stays SQLite (Redis, Firestore or Bigtable in
production).

### How to recreate
```bash
# 1. Code
git clone https://github.com/Draiken7/21f2000602_MLOPS_WEEKLY_ASSIGNMENT.git
cd 21f2000602_MLOPS_WEEKLY_ASSIGNMENT && git checkout week_3

# 2. Dependencies (Workbench: if scikit-learn fails to import after this, see "Issues" below)
pip install "feast[gcp]" "scikit-learn>=1.5" pandas joblib dvc dvc-gs db-dtypes

# 3. Dataset (from the course repo, via the DVC import record) + trained model (from GCS)
dvc pull
```
Then run `week3_feast_iris.ipynb` top to bottom (it regenerates Parquet, registry, online store and model),
or run the feature-store steps from the CLI:
```bash
cd iris_feature_store/feature_repo
feast apply
feast materialize 2025-09-01T00:00:00 2025-10-02T00:00:00
feast feature-views list          # STATE = AVAILABLE_ONLINE
```
Task 6 needs a GCP project with BigQuery enabled. Change `arcane-sentinel-252018` in
`feature_repo_bq/feature_store.yaml`, `iris_features_bq.py` and the notebook's `PROJECT_ID`.

### Issues encountered and fixes
| Issue | Cause | Fix |
|---|---|---|
| `week_3` not visible on GitHub | Branch existed only locally | `git push -u origin week_3` |
| `ImportError: ComplexWarning` from scikit-learn | Installing Feast upgraded NumPy to 2.x; the image's scikit-learn was built for NumPy 1.x | Upgrade scikit-learn and SciPy, restart kernel |
| `ImportError: _manual_routing` | pip upgrade over the micromamba-installed package left mixed old and new files | `pip uninstall scikit-learn`, delete `site-packages/sklearn`, reinstall with `--no-deps` |
| Point-in-time demo row "missing" | File offline store drops entity rows with no prior data (BigQuery keeps them as NULL) | Documented; always compare requested vs returned row counts |

---

## Week 2 — DVC (summary)
See branch `week_2`. Data (`data/iris.csv`) and model (`artifacts/model.joblib`) are versioned with DVC,
remote `gs://21f2000602-dvc-week2/dvc-store`. Tags `v1.0` (101 rows, acc 0.951) and `v2.0`
(150 rows, acc 0.917).
