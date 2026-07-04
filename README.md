# openIMIS Fraud Detection Module

AI-powered claims anomaly detector for openIMIS — Rules Engine + Isolation Forest.

---

## Problem Statement

Health insurance fraud is a systemic problem in Kenya and across low- and
middle-income countries. When a patient visits a hospital, the hospital submits
a claim to the insurer describing what services were provided and how much they
cost. The insurer then reviews the claim and decides how much to pay. This
process is almost entirely manual — a human reviewer reads each claim and makes
a judgement call.

This creates two serious problems:

**1. It is slow.** With hundreds or thousands of claims arriving daily,
reviewers cannot keep up. Claims sit in queues for days or weeks before being
processed.

**2. It misses fraud.** Patterns visible in real Kenyan insurer data include:
claims filed 6–8 months after the service date; invoices settled at a fraction
of the billed amount; and repeated use of vague ICD codes such as `Z51.9`
("medical care, unspecified") that provide no clinical justification for
high-value services. A reviewer looking at one claim in isolation cannot easily
detect that the same hospital has been systematically overbilling for months.

The openIMIS platform, used by 38.8 million beneficiaries across 14 countries,
currently has no automated fraud detection layer. Every claim must be manually
adjudicated.

---

## Solution Overview

A two-layer detection system integrated natively into openIMIS:

**Layer 1 — Rules Engine** (`rules.py`): Deterministic, auditable rules derived
directly from known fraud patterns in the dataset. Any claim that violates a
rule is immediately flagged with a plain-language explanation. This catches the
obvious cases at zero computational cost and is explainable to any reviewer.

**Layer 2 — Isolation Forest** (`engine.py`): An unsupervised ML model trained
on anonymised Kenyan insurer claim data. It scores every claim for statistical
anomaly, catching fraud patterns that no one thought to write a rule for — the
subtle, evolving schemes that slip past the rulebook.

Both layers run automatically every time a claim is saved in openIMIS via a
Django `post_save` signal. Results are stored in `tbl_FraudFlag` and exposed
via REST and GraphQL APIs. A claims reviewer sees a colour-coded risk badge
(LOW / MEDIUM / HIGH) on each claim. High-risk claims are prioritised for
human review; low-risk claims are processed faster.

When a reviewer overrides a decision, that override is logged and feeds back
into the next model retraining cycle, making the system progressively smarter.

---

## Architecture

```
Claim saved (Django post_save signal)
          │
          ▼
  ┌───────────────┐     ┌────────────────────┐
  │  Rules Engine │     │  Isolation Forest  │
  │  rules.py     │     │  engine.py         │
  └───────┬───────┘     └────────┬───────────┘
          │  fired_rules          │  anomaly_score
          └────────────┬──────────┘
                       ▼
             compute_risk_level()
             ┌─────────────────────────────┐
             │ 2+ rules fired       → HIGH │
             │ rules AND ML anomaly → HIGH │
             │ rules XOR ML anomaly → MED  │
             │ ML near-miss (<-0.1) → MED  │
             │ nothing flagged      → LOW  │
             └─────────────────────────────┘
                       │
                       ▼
              tbl_FraudFlag (DB)
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
      REST API      GraphQL    FHIR Extension
  /api/fraud_detect/ schema.py  fhir_extensions.py
```

---

## Features Used by the Model

| Feature | Description | Fraud Signal |
|---------|-------------|--------------|
| `invoice_inflation_ratio` | Invoice ÷ settled amount (capped at 10×) | Overbilling |
| `claim_lag_days` | Days between service date and claim submission | Backdating |
| `icd_is_vague` | 1 if ICD code is a known catch-all (e.g. Z51.9) | Diagnosis disguise |
| `provider_avg_inflation` | Average inflation ratio across all provider claims | Provider-level pattern |
| `provider_claim_count` | Total claims submitted by this provider | Volume anomaly |
| `member_claim_count` | Total claims for this member | Ghost beneficiary signal |
| `amount_vs_benchmark` | Invoice ÷ median invoice for this benefit code | Benchmark deviation |
| `had_pre_audit_adjustment` | 1 if the settled amount was reduced below the invoiced amount before audit | Original submission was not right (weak signal) |

> **On `had_pre_audit_adjustment`**: the fact that an adjustment was needed at all
> is itself a weak fraud signal — it means the original submission was off in some
> way, and that "offness" tends to correlate with the stronger signals fraud
> detectors care about (over-billing, vague codes, etc.). It is derived as
> `invoice_inflation_ratio > 1.0` and correlates ~0.58 with the proxy fraud label.
>
> Note this is measured against a **proxy** fraud rate, not a true one: we use
> "settled < 80% of invoice" as a stand-in for "the insurer detected something
> off", because no labelled fraud data exists.

---

## Rules Engine Checks

Layer 1 runs every claim through a set of deterministic, auditable rules
(`rules.py`). Each rule that fires attaches a plain-language reason to the claim.
Thresholds are tunable via constants at the top of `rules.py`.

| Rule | Fires when | Fraud Signal |
|------|-----------|--------------|
| Claim lag exceeds 90 days | Claim submitted more than 90 days after the service date | Backdating |
| Invoice inflation above 3x | Invoiced amount is more than 3× the approved amount | Overbilling |
| Vague ICD code used | Diagnosis is a known non-specific catch-all (e.g. `Z51.9`) | Diagnosis disguise |
| Claim filed after audit date | Submission date is later than the audit date (logically impossible) | Record tampering |
| High-value claim with vague diagnosis | Claimed amount exceeds 20,000 KES with a non-specific ICD code | Unjustified high-value claim |
| Invalid calendar date on claim | Any date field holds an impossible calendar date — e.g. 29 Feb in a non-leap year, 31 April, or 31 June | Data-entry error or record tampering |

> The **Invalid calendar date** rule inspects the raw string form of `date_from`,
> `date_claimed`, and `audit_date`. Values already parsed into `datetime.date`
> objects are always valid and are skipped; non-date-looking strings are ignored
> to avoid false positives. Both `YYYY-MM-DD` and `DD-MM-YYYY` formats are checked.

---

## GraphQL

The model choice flows directly from one hard constraint: **there are no real
fraud labels.** `proxy_fraud_label` ("settled < 80% of invoice") is used only for
*evaluation* — it is never fed to the model during training (`model.fit(X)` takes
no `y`). Given that, plus rare/unknown-shaped fraud, ~422k rows of tabular data,
8 numeric features, and the need to score inside a `post_save` signal, Isolation
Forest is the natural fit.

Its core idea: anomalies are **few and different**, so they are *easy to isolate* —
a random tree separates an outlier from the rest in very few splits, and the
average split-depth becomes the anomaly score.

| Property | Why it matters here |
|----------|---------------------|
| Unsupervised | Works with zero labels — the whole reason it was chosen |
| Fast training & O(log n) inference | Cheap enough to score inside a `post_save` signal |
| Handles multi-feature interactions | Catches "weird combinations" a single rule wouldn't |
| Few hyperparameters | Mainly `contamination` + `n_estimators`; easy to tune/retrain |

### Alternatives considered (unsupervised — the "no labels" world)

| Model | How it works | vs. Isolation Forest |
|-------|--------------|----------------------|
| **Local Outlier Factor (LOF)** | Flags points in low-density neighbourhoods vs. their neighbours | Good at *local* anomalies, but memory/compute-heavy (neighbour lookups) and awkward to score new points live. Poor fit for 422k rows scored on every save. |
| **One-Class SVM** | Learns a boundary around "normal" data | Powerful but scales ~O(n²); very slow to train at this size and sensitive to kernel/params. |
| **Elliptic Envelope** | Fits a Gaussian; flags points far from the centre | Assumes one roughly-Gaussian blob — claims data is skewed and multi-modal, so it underperforms. |
| **Autoencoder (neural net)** | Reconstructs input; high reconstruction error = anomaly | Can beat IsoForest on rich data, but needs more engineering, tuning, and compute, and is far less explainable. Overkill for 8 tabular features. |
| **DBSCAN / GMM** | Density/cluster membership; outliers fall outside | Not designed for scoring new points incrementally; harder to tune with little upside here. |

**Takeaway:** Isolation Forest is the standard first choice for tabular anomaly
detection because it best balances accuracy, speed, scalability, and simplicity.
The alternatives either scale poorly (One-Class SVM, LOF), assume a data shape we
don't have (Elliptic Envelope, GMM), or add heavy complexity (autoencoder) for
uncertain gain.

### If real labels become available (supervised — the upgrade path)

If verified fraud labels accumulate (e.g. from reviewer overrides in
`tbl_ReviewerOverride`, or an audited dataset), supervised models usually
**outperform any anomaly detector**, because they learn the actual decision
boundary rather than just "unusualness":

| Model | Strengths | Trade-offs |
|-------|-----------|------------|
| **Gradient-boosted trees (XGBoost / LightGBM)** | Usually best-in-class on tabular fraud; handles imbalance; feature importances | Needs labels; care with class imbalance |
| **Random Forest (supervised)** | Robust, explainable, strong baseline | Needs labels |
| **Logistic Regression** | Simple, transparent, regulator-friendly | Lower ceiling; needs labels |

> **Why not train supervised on the proxy label now?** A supervised model trained
> on `proxy_fraud_label` would largely just learn to reproduce the proxy rule
> (`settled < 80% of invoice`) — nearly circular. Using an *unsupervised* detector
> is more honest: it finds anomalies **independent of** the proxy definition, and
> we then measure how well those anomalies line up with it (ROC-AUC 0.847).

The natural evolution is **unsupervised now → semi-supervised → supervised (or a
hybrid that feeds the IsoForest score as a feature into a supervised model)** once
enough labels exist. The feedback loop (reviewer overrides logged and surfaced at
retraining) is already built to support that migration.

---

## Installation

### 1. Register the module

Add to `openimis-be_py/openimis.json` (at the end of the `"modules"` array):

```json
{ "name": "fraud_detect", "pip": "openimis-be-fraud-detect", "url": "fraud_detect.urls" }
```

### 2. Start the backend with the Docker Compose overlay

```bash
# From openimis-dist_dkr/
docker compose -f compose.yml -f compose.fraud-detect.yml up -d --no-deps backend worker
```

> **Startup time**: The first start on a fresh container installs `scikit-learn`,
> `joblib`, `pandas`, and `numpy` (~2–3 min). Subsequent restarts of the same
> container detect the packages are already present and skip the heavy install
> (~5–10 seconds).

### 3. Run migrations

```bash
docker compose -f compose.yml -f compose.fraud-detect.yml exec backend \
  python manage.py migrate fraud_detect
```

This creates three tables:
- `tbl_FraudFlag` — one fraud assessment row per scored claim
- `tbl_ReviewerOverride` — human reviewer decisions (feeds retraining)
- `tbl_FraudModelVersion` — tracks which `.joblib` artefact is active

### 4. Train the ML model

Place the feature CSV at `data/claims_features.csv` (must contain the 8 feature
columns listed above, plus an optional `proxy_fraud_label` column which is
ignored at inference time):

```bash
docker compose -f compose.yml -f compose.fraud-detect.yml exec backend \
  python manage.py retrain_fraud_model
```

The command saves `models/fraud_model.joblib` and `models/fraud_scaler.joblib`
to the host filesystem via the bind mount, so they survive container restarts.
It also evaluates the new model on a held-out 20% split and regenerates
[models/README.md](models/README.md) with the fresh performance report — so the
report is never stale. After training, restart the backend to load the new
artefacts:

```bash
docker compose -f compose.yml -f compose.fraud-detect.yml restart backend
until curl -sf http://localhost/api/fraud_detect/flags/ | grep -q count; do sleep 3; done
```

Retraining options:
```bash
python manage.py retrain_fraud_model --contamination 0.05 --n-estimators 300
python manage.py retrain_fraud_model --dry-run   # fit but do not save
```

### 5. Seed demo claims (optional)

```bash
docker compose -f compose.yml -f compose.fraud-detect.yml exec backend \
  python manage.py seed_demo_claims
```

Creates four synthetic claims (DEMO-A through DEMO-D) covering LOW, MEDIUM,
and HIGH risk scenarios. Each claim is auto-scored by the `post_save` signal
as soon as it is created.

> **ICD code note**: The seed command looks up Z51.9, J06.9, and H52.1 in the
> database. If those codes are absent (common in fresh installs), all claims
> fall back to the first available diagnosis. DEMO-B still reaches HIGH via
> 2 rules (lag + inflation); DEMO-C via ML anomaly detection.

---

## API Reference

All endpoints are mounted at `/api/fraud_detect/` by openIMIS's URL router
(`openimisurls.py` prefixes every module with `/api/<module_name>/`).

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/fraud_detect/flags/` | List all fraud flags. Supports filtering and pagination. |
| GET | `/api/fraud_detect/flags/{claim_id}/` | Get the flag for a specific claim |
| POST | `/api/fraud_detect/override/` | Submit a reviewer override decision |
| POST | `/api/fraud_detect/score/` | Score a raw claim dict on demand (no DB write) |
| POST | `/api/fraud_detect/rescore/{claim_id}/` | Score an existing claim from the DB and persist the result |
| POST | `/api/fraud_detect/claims/` | Create a real claim via the claim module, auto-score it, and return the flag |

> **Shell tip**: Write curl commands as a single line. A bare backslash-newline
> continuation in zsh can silently split the command, causing `command not found: -H`.

---

### GET `/api/fraud_detect/flags/`

List all fraud flags, newest first. Supports `risk_level`, `limit`, and `offset`
query parameters.

**Query parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `risk_level` | `HIGH` \| `MEDIUM` \| `LOW` | — | Filter by risk level |
| `limit` | integer 1–1000 | 100 | Number of results per page |
| `offset` | integer ≥ 0 | 0 | Number of results to skip |

#### Example: all flags (first page)

```bash
curl -s "http://localhost/api/fraud_detect/flags/" | python3 -m json.tool
```

```json
{
  "count": 1482,
  "limit": 100,
  "offset": 0,
  "results": [
    {
      "id": 42,
      "claim_id": 1017,
      "is_rule_flagged": true,
      "rule_flag_reasons": [
        {"name": "Invoice inflation above 3x", "description": "The invoiced amount is more than 3 times the amount that was approved for payment. This suggests deliberate overbilling."},
        {"name": "Vague ICD code used", "description": "ICD code Z51.9 is a known non-specific catch-all diagnosis."}
      ],
      "anomaly_score": -0.305,
      "is_ml_anomaly": true,
      "overall_risk_level": "HIGH",
      "created_at": "2026-07-01T11:22:04.123456Z",
      "updated_at": "2026-07-01T11:22:04.123456Z"
    }
  ]
}
```

#### Example: filter HIGH-risk flags, page 2

```bash
curl -s "http://localhost/api/fraud_detect/flags/?risk_level=HIGH&limit=50&offset=50" | python3 -m json.tool
```

```json
{
  "count": 203,
  "limit": 50,
  "offset": 50,
  "results": [ ... ]
}
```

#### Example: invalid risk_level → 400

```bash
curl -s "http://localhost/api/fraud_detect/flags/?risk_level=UNKNOWN" | python3 -m json.tool
```

```json
{
  "detail": "risk_level must be HIGH, MEDIUM, or LOW."
}
```

#### Example: non-integer limit → 400

```bash
curl -s "http://localhost/api/fraud_detect/flags/?limit=all" | python3 -m json.tool
```

```json
{
  "detail": "limit and offset must be integers."
}
```

---

### GET `/api/fraud_detect/flags/{claim_id}/`

Returns the current fraud assessment for a single claim by its integer ID.

#### Example: claim that has been scored

```bash
curl -s "http://localhost/api/fraud_detect/flags/1017/" | python3 -m json.tool
```

```json
{
  "id": 42,
  "claim_id": 1017,
  "is_rule_flagged": true,
  "rule_flag_reasons": [
    {"name": "Invoice inflation above 3x", "description": "The invoiced amount is more than 3 times the amount that was approved for payment. This suggests deliberate overbilling."},
    {"name": "Vague ICD code used", "description": "ICD code Z51.9 is a known non-specific catch-all diagnosis."}
  ],
  "anomaly_score": -0.305,
  "is_ml_anomaly": true,
  "overall_risk_level": "HIGH",
  "created_at": "2026-07-01T11:22:04.123456Z",
  "updated_at": "2026-07-01T11:22:04.123456Z"
}
```

#### Example: LOW-risk claim (no rules fired, ML near-neutral)

```bash
curl -s "http://localhost/api/fraud_detect/flags/1001/" | python3 -m json.tool
```

```json
{
  "id": 10,
  "claim_id": 1001,
  "is_rule_flagged": false,
  "rule_flag_reasons": [],
  "anomaly_score": 0.142,
  "is_ml_anomaly": false,
  "overall_risk_level": "LOW",
  "created_at": "2026-07-01T09:00:01.000000Z",
  "updated_at": "2026-07-01T09:00:01.000000Z"
}
```

#### Example: claim that has never been scored → 404

```bash
curl -s "http://localhost/api/fraud_detect/flags/99999/" | python3 -m json.tool
```

```json
{
  "detail": "No FraudFlag matches the given query."
}
```

---

### POST `/api/fraud_detect/override/`

Records a reviewer's manual decision on a previously scored claim. The claim
must already have a `FraudFlag` row (i.e. it must have been scored at least once).

**Request body**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `claim_id` | integer | ✅ | ID of the claim being reviewed |
| `reviewer_decision` | `APPROVE` \| `REJECT` \| `ESCALATE` | ✅ | Reviewer's verdict |
| `reviewer_id` | integer | ✅ | ID of the reviewer submitting the decision |
| `notes` | string | — | Free-text justification (optional) |

**Response (201)**: The created override record, including `original_risk_level`
captured from the flag at submission time.

#### Example: approve a LOW-risk claim

```bash
curl -s -X POST http://localhost/api/fraud_detect/override/ -H "Content-Type: application/json" -d '{"claim_id": 1001, "reviewer_decision": "APPROVE", "reviewer_id": 7, "notes": "Verified with hospital records. Legitimate."}' | python3 -m json.tool
```

```json
{
  "id": 88,
  "claim_id": 1001,
  "original_risk_level": "LOW",
  "reviewer_decision": "APPROVE",
  "reviewer_id": 7,
  "notes": "Verified with hospital records. Legitimate.",
  "created_at": "2026-07-01T14:05:11.000000Z"
}
```

#### Example: reject a HIGH-risk claim

```bash
curl -s -X POST http://localhost/api/fraud_detect/override/ -H "Content-Type: application/json" -d '{"claim_id": 1017, "reviewer_decision": "REJECT", "reviewer_id": 3, "notes": "Invoice inflated 33x. Provider flagged for audit."}' | python3 -m json.tool
```

```json
{
  "id": 89,
  "claim_id": 1017,
  "original_risk_level": "HIGH",
  "reviewer_decision": "REJECT",
  "reviewer_id": 3,
  "notes": "Invoice inflated 33x. Provider flagged for audit.",
  "created_at": "2026-07-01T14:07:44.000000Z"
}
```

#### Example: escalate for senior review

```bash
curl -s -X POST http://localhost/api/fraud_detect/override/ -H "Content-Type: application/json" -d '{"claim_id": 1022, "reviewer_decision": "ESCALATE", "reviewer_id": 5}' | python3 -m json.tool
```

```json
{
  "id": 90,
  "claim_id": 1022,
  "original_risk_level": "MEDIUM",
  "reviewer_decision": "ESCALATE",
  "reviewer_id": 5,
  "notes": "",
  "created_at": "2026-07-01T14:10:02.000000Z"
}
```

#### Example: invalid decision value → 400

```bash
curl -s -X POST http://localhost/api/fraud_detect/override/ -H "Content-Type: application/json" -d '{"claim_id": 1017, "reviewer_decision": "IGNORE", "reviewer_id": 3}' | python3 -m json.tool
```

```json
{
  "reviewer_decision": [
    "reviewer_decision must be one of: APPROVE, ESCALATE, REJECT"
  ]
}
```

#### Example: claim has no FraudFlag yet → 404

```bash
curl -s -X POST http://localhost/api/fraud_detect/override/ -H "Content-Type: application/json" -d '{"claim_id": 99999, "reviewer_decision": "APPROVE", "reviewer_id": 3}' | python3 -m json.tool
```

```json
{
  "detail": "No FraudFlag matches the given query."
}
```

---

### POST `/api/fraud_detect/score/`

Scores a claim dict on demand. No database read or write — useful for previewing
a score before submitting a claim or for integration testing.

**Request body** — all fields are optional; missing values fall back to safe
neutral defaults.

| Field | Type | Description |
|-------|------|-------------|
| `claimed_amount` | number | Total billed amount |
| `approved_amount` | number | Amount the insurer settled |
| `icd_code` | string | Diagnosis code (e.g. `Z51.9`) |
| `date_from` | ISO date string | Service/admission date (`YYYY-MM-DD`) |
| `date_claimed` | ISO date string | Date claim was submitted (`YYYY-MM-DD`) |
| `provider_avg_inflation` | number | Pre-computed provider inflation ratio |
| `provider_claim_count` | integer | Total claims from this provider |
| `member_claim_count` | integer | Total claims for this member |
| `amount_vs_benchmark` | number | Invoice ÷ median for this benefit code |

**Response fields**

| Field | Description |
|-------|-------------|
| `rules.is_flagged` | `true` if any rule fired |
| `rules.fired_rules` | List of `{name, description}` objects |
| `ml.anomaly_score` | Raw decision function value (more negative = more anomalous) |
| `ml.is_anomaly` | `true` if model predicts anomaly |
| `overall_risk_level` | `HIGH`, `MEDIUM`, or `LOW` |

#### Example: HIGH-risk claim (multiple rules + ML anomaly)

```bash
curl -s -X POST http://localhost/api/fraud_detect/score/ -H "Content-Type: application/json" -d '{"claimed_amount": 50000, "approved_amount": 1500, "icd_code": "Z51.9", "date_from": "2025-09-01", "date_claimed": "2026-07-01", "provider_avg_inflation": 3.5, "provider_claim_count": 80, "member_claim_count": 1, "amount_vs_benchmark": 8.0}' | python3 -m json.tool
```

```json
{
  "rules": {
    "is_flagged": true,
    "fired_rules": [
      {
        "name": "Invoice inflation above 3x",
        "description": "Claimed amount is 33.3x the approved amount (threshold: 3x)"
      },
      {
        "name": "Vague ICD code used",
        "description": "ICD code Z51.9 is a known non-specific catch-all diagnosis"
      },
      {
        "name": "High-value claim with vague diagnosis",
        "description": "Claimed amount 50000 exceeds 10000 threshold with a vague ICD code"
      },
      {
        "name": "Claim submitted long after service",
        "description": "Claim submitted 303 days after service date (threshold: 90 days)"
      }
    ]
  },
  "ml": {
    "anomaly_score": -0.305,
    "is_anomaly": true
  },
  "overall_risk_level": "HIGH"
}
```

#### Example: MEDIUM-risk claim (one rule, ML neutral)

```bash
curl -s -X POST http://localhost/api/fraud_detect/score/ -H "Content-Type: application/json" -d '{"claimed_amount": 8000, "approved_amount": 2000, "icd_code": "A09.0", "date_from": "2026-06-01", "date_claimed": "2026-07-01", "provider_avg_inflation": 1.2, "provider_claim_count": 15, "member_claim_count": 2, "amount_vs_benchmark": 1.1}' | python3 -m json.tool
```

```json
{
  "rules": {
    "is_flagged": true,
    "fired_rules": [
      {
        "name": "Invoice inflation above 3x",
        "description": "Claimed amount is 4.0x the approved amount (threshold: 3x)"
      }
    ]
  },
  "ml": {
    "anomaly_score": 0.042,
    "is_anomaly": false
  },
  "overall_risk_level": "MEDIUM"
}
```

#### Example: LOW-risk claim (no rules, ML normal)

```bash
curl -s -X POST http://localhost/api/fraud_detect/score/ -H "Content-Type: application/json" -d '{"claimed_amount": 3500, "approved_amount": 3200, "icd_code": "J18.9", "date_from": "2026-06-25", "date_claimed": "2026-06-28", "provider_avg_inflation": 1.1, "provider_claim_count": 12, "member_claim_count": 1, "amount_vs_benchmark": 0.9}' | python3 -m json.tool
```

```json
{
  "rules": {
    "is_flagged": false,
    "fired_rules": []
  },
  "ml": {
    "anomaly_score": 0.187,
    "is_anomaly": false
  },
  "overall_risk_level": "LOW"
}
```

#### Example: minimal request (no fields supplied — all defaults)

```bash
curl -s -X POST http://localhost/api/fraud_detect/score/ -H "Content-Type: application/json" -d '{}' | python3 -m json.tool
```

```json
{
  "rules": {
    "is_flagged": false,
    "fired_rules": []
  },
  "ml": {
    "anomaly_score": 0.0,
    "is_anomaly": false
  },
  "overall_risk_level": "LOW"
}
```

---

### POST `/api/fraud_detect/rescore/{claim_id}/`

Fetches the live `Claim` row from the database by ID, runs both the rules engine
and the ML model against it, and writes the result to `tbl_FraudFlag` using
`update_or_create`. Returns the saved `FraudFlag` record.

This is the correct endpoint when:
- A claim was created **before** the fraud detection module was installed (so
  the `post_save` signal never fired for it).
- You want to **re-score** a claim after retraining the model.
- You need to **backfill** fraud flags for a batch of existing claims.

The `post_save` signal handles scoring automatically for all future claim saves.
Use this endpoint only for claims that already exist in the DB without a flag.

**No request body required.**

**Response**
- `201 Created` — flag did not exist before; it was just created.
- `200 OK` — flag already existed; it has been updated with a fresh score.
- `404 Not Found` — no `Claim` with that ID exists in the database.
- `503 Service Unavailable` — the `claim` openIMIS module is not installed.

#### Example: score and persist a claim for the first time

```bash
curl -s -X POST http://localhost/api/fraud_detect/rescore/1042/ | python3 -m json.tool
```

```json
{
  "id": 95,
  "claim_id": 1042,
  "is_rule_flagged": true,
  "rule_flag_reasons": [
    {"name": "Invoice inflation above 3x", "description": "The invoiced amount is more than 3 times the amount that was approved for payment. This suggests deliberate overbilling."}
  ],
  "anomaly_score": -0.198,
  "is_ml_anomaly": false,
  "overall_risk_level": "MEDIUM",
  "created_at": "2026-07-01T15:30:00.000000Z",
  "updated_at": "2026-07-01T15:30:00.000000Z"
}
```

HTTP status: `201 Created`.

#### Example: re-score an already-flagged claim → 200 OK

The flag row is updated in-place. `created_at` stays the same; `updated_at` changes.

```bash
curl -s -w "\nHTTP %{http_code}\n" -X POST http://localhost/api/fraud_detect/rescore/1042/ | python3 -m json.tool
```

```json
{
  "id": 95,
  "claim_id": 1042,
  "is_rule_flagged": true,
  "rule_flag_reasons": [
    {"name": "Invoice inflation above 3x", "description": "The invoiced amount is more than 3 times the amount that was approved for payment. This suggests deliberate overbilling."}
  ],
  "anomaly_score": -0.198,
  "is_ml_anomaly": false,
  "overall_risk_level": "MEDIUM",
  "created_at": "2026-07-01T15:30:00.000000Z",
  "updated_at": "2026-07-01T16:05:33.000000Z"
}
```

HTTP status: `200 OK`.

#### Example: re-score after model retraining

```bash
# Retrain the model
docker compose -f compose.yml -f compose.fraud-detect.yml exec backend python manage.py retrain_fraud_model
docker compose -f compose.yml -f compose.fraud-detect.yml restart backend

# Now re-score any specific claim with the new model
curl -s -X POST http://localhost/api/fraud_detect/rescore/1017/ | python3 -m json.tool
```

The response shape is identical to the 200/201 examples above; the score values
will differ if the new model weights changed.

#### Example: backfill all unscored claims with a shell loop

```bash
# Fetch every claim_id that has no FraudFlag yet, then rescore each one
curl -s "http://localhost/api/fraud_detect/flags/?limit=1000" | python3 -c "
import json, sys
scored = {r['claim_id'] for r in json.load(sys.stdin)['results']}
print(' '.join(str(c) for c in scored))
" > /tmp/already_scored.txt

# (Then iterate over your full claim ID list and skip those already scored)
```

#### Example: soft-deleted or non-existent claim → 404

openIMIS soft-deletes records by setting `ValidityTo`. The default Claim manager
excludes them, so rescoring a superseded claim also returns 404.

```bash
curl -s -X POST http://localhost/api/fraud_detect/rescore/99999/ | python3 -m json.tool
```

```json
{
  "detail": "No Claim matches the given query."
}
```

HTTP status: `404 Not Found`.

#### Example: claim module not installed → 503

Occurs when the `fraud_detect` module is running in a test environment that does
not have the `claim` openIMIS module in `openimis.json`.

```bash
curl -s -X POST http://localhost/api/fraud_detect/rescore/1042/ | python3 -m json.tool
```

```json
{
  "detail": "The claim module is not installed in this environment."
}
```

HTTP status: `503 Service Unavailable`.

---

### POST `/api/fraud_detect/claims/`

Creates a **real openIMIS claim** through the claim module's ORM. Saving the claim
fires the `post_save` signal, which automatically scores it and writes a
`FraudFlag` row. The response returns **both** the new claim and its fraud
assessment — so you can create-and-see-the-score in a single request.

This is the endpoint to use for a live demo: submit a claim, get its risk level back instantly.

**Request body**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `claimed_amount` | number | ✅ | Total billed amount |
| `approved_amount` | number | — | Amount settled (defaults to `claimed_amount`) |
| `icd_code` | string | — | Diagnosis code (e.g. `Z51.9`); falls back to first available if absent |
| `date_from` | ISO date | — | Service date `YYYY-MM-DD` (defaults to today) |
| `date_claimed` | ISO date | — | Submission date `YYYY-MM-DD` (defaults to today) |
| `code` | string | — | Claim code (auto-generated `FD-YYYYMMDD-N` if omitted) |
| `insuree_id` | integer | — | Insuree FK (first active insuree if omitted) |
| `health_facility_id` | integer | — | Health facility FK (first active HF if omitted) |

**Response**
- `201 Created` — claim created and scored; body contains `claim` + `fraud_flag`.
- `400 Bad Request` — missing `claimed_amount`, bad dates, duplicate code, or
  no resolvable insuree / health facility / diagnosis.
- `503 Service Unavailable` — the `claim` openIMIS module is not installed.

> **Requires the full openIMIS stack** (claim, insuree, location, medical apps
> installed and migrated, with at least one active insuree, health facility, and
> diagnosis). On fresh installs where a requested ICD code is missing, the claim
> falls back to the first available diagnosis (same as `seed_demo_claims`).

#### Example: create a suspicious claim → HIGH

Inflated invoice (50,000 billed vs 1,500 approved) + 300-day lag.

```bash
curl -s -X POST http://localhost/api/fraud_detect/claims/ -H "Content-Type: application/json" -d '{"claimed_amount": 50000, "approved_amount": 1500, "icd_code": "Z51.9", "date_from": "2025-09-01", "date_claimed": "2026-07-01"}' | python3 -m json.tool
```

```json
{
  "claim": {
    "id": 34,
    "code": "FD-20260704-29",
    "claimed": 50000.0,
    "approved": 1500.0,
    "icd_code": "Z51.9",
    "date_from": "2025-09-01",
    "date_claimed": "2026-07-01"
  },
  "fraud_flag": {
    "id": 10,
    "claim_id": 34,
    "is_rule_flagged": true,
    "rule_flag_reasons": [
      {"name": "Claim lag exceeds 90 days", "description": "The claim was filed more than 90 days after the service was delivered. This is a strong indicator of backdated or fabricated claims."},
      {"name": "Invoice inflation above 3x", "description": "The invoiced amount is more than 3 times the amount that was approved for payment. This suggests deliberate overbilling."}
    ],
    "anomaly_score": -0.158,
    "is_ml_anomaly": true,
    "overall_risk_level": "HIGH",
    "created_at": "2026-07-04T10:50:18.562318Z",
    "updated_at": "2026-07-04T10:50:18.562336Z"
  }
}
```

HTTP status: `201 Created`.

#### Example: create a clean claim → LOW

Claimed equals approved, submitted the same week (all other fields default).

```bash
curl -s -X POST http://localhost/api/fraud_detect/claims/ -H "Content-Type: application/json" -d '{"claimed_amount": 3000, "approved_amount": 3000, "icd_code": "J06.9"}' | python3 -m json.tool
```

```json
{
  "claim": {
    "id": 35,
    "code": "FD-20260704-30",
    "claimed": 3000.0,
    "approved": 3000.0,
    "icd_code": "J06.9",
    "date_from": "2026-07-04",
    "date_claimed": "2026-07-04"
  },
  "fraud_flag": {
    "id": 11,
    "claim_id": 35,
    "is_rule_flagged": false,
    "rule_flag_reasons": [],
    "anomaly_score": 0.121,
    "is_ml_anomaly": false,
    "overall_risk_level": "LOW",
    "created_at": "2026-07-04T10:50:19.125686Z",
    "updated_at": "2026-07-04T10:50:19.125701Z"
  }
}
```

HTTP status: `201 Created`.

#### Example: missing claimed_amount → 400

```bash
curl -s -w "\nHTTP %{http_code}\n" -X POST http://localhost/api/fraud_detect/claims/ -H "Content-Type: application/json" -d '{"icd_code": "J06.9"}'
```

```json
{
  "detail": "claimed_amount is required."
}
```

HTTP status: `400 Bad Request`.

> **Tip**: The returned `claim.id` can be passed straight to
> `GET /api/fraud_detect/flags/{claim_id}/` or
> `POST /api/fraud_detect/rescore/{claim_id}/` for follow-up calls.

---

```graphql
query {
  fraudFlags(riskLevel: "HIGH", first: 20) {
    claimId
    overallRiskLevel
    anomalyScore
    ruleFlagReasons
  }
}
```

---

## Model Performance Report

The full performance report for the currently active model — classification
metrics, confusion matrix, ROC-AUC, and interpretation — lives alongside the
model artefacts in [models/README.md](models/README.md).

**At a glance** (8-feature model, 84,477-claim test set): accuracy **87.9%**,
**ROC-AUC 0.847**. See [models/README.md](models/README.md) for the full breakdown.

---

## Running Tests

```bash
docker compose -f compose.yml -f compose.fraud-detect.yml exec backend \
  python manage.py test fraud_detect --keepdb --verbosity=2
```

**36 tests** across three test classes:

| Test class | Count | What it verifies |
|---|---|---|
| `MLScoringTestCase` | 7 | `score_claim_ml()` handles missing artefacts and exceptions without crashing |
| `RiskLevelTestCase` | 9 | `compute_risk_level()` returns HIGH / MEDIUM / LOW for all combinations, including 2+ rules → HIGH |
| `RulesEngineTestCase` | 20 | All 5 rules fire at correct thresholds; boundary conditions; missing data handled gracefully |

Two expected warnings appear in the output:
- A traceback inside `test_returns_neutral_result_on_model_exception` — **intentional**, confirms error handling works.
- `Your models in app(s): 'claim', 'core', 'payroll' have changes not reflected in a migration` — an upstream openIMIS issue, unrelated to this module.

> **Test database setup**: The openIMIS test runner cannot build `test_imis`
> from scratch due to a known upstream deferred-SQL issue. Clone the live DB
> once before the first test run:
> ```bash
> docker compose exec db psql -U IMISuser -d postgres -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = 'IMIS' AND pid <> pg_backend_pid();"
> docker compose exec db psql -U IMISuser -d postgres -c "DROP DATABASE IF EXISTS test_imis;"
> docker compose exec db psql -U IMISuser -d postgres -c 'CREATE DATABASE test_imis TEMPLATE "IMIS";'
> ```
> Then always run tests with `--keepdb`.

---

## Responsible AI

- **Training data**: Anonymised claims from a Kenyan medical insurer (422,382
  rows). Member names and IDs were hashed (SHA-256 truncated to 12 chars) before
  model training. Raw PII was never committed to the repository.
- **Bias**: The model was trained on one insurer's data and may not generalise
  to other provider networks or country contexts without retraining on local data.
- **Intended use**: Flagging claims for **human review only**. The system must
  not be used to automatically reject claims without a reviewer seeing the flag.
- **Explainability**: The rules engine always produces a human-readable reason
  for each flag. The ML anomaly score is shown alongside the fired rules to help
  reviewers understand why a claim was flagged.
- **Known failure modes**: Novel fraud patterns not present in training data
  will not be detected until the model is retrained with new data.
- **Feedback loop**: Reviewer overrides are logged in `tbl_ReviewerOverride`
  and surfaced during the next `retrain_fraud_model` run, allowing the model to
  learn from human corrections over time.

---

## Known Limitations

- ICD vague-code detection uses a fixed list (`VAGUE_ICD_CODES` in `engine.py`
  and `rules.py`). Codes not on the list are treated as specific. The list
  should be expanded as local clinicians identify additional catch-all codes.
- The `provider_avg_inflation`, `provider_claim_count`, `member_claim_count`,
  and `amount_vs_benchmark` features default to neutral values (1.0 / 1) when
  not supplied at inference time. Real-time aggregate precomputation is a future
  enhancement.
- The Isolation Forest model is retrained manually. A scheduled Celery task
  for automatic periodic retraining is a planned improvement.
- The `proxy_fraud_label` used for evaluation (settled < 80% of invoice) is an
  imperfect proxy — some partial settlements are legitimate. True fraud labels
  would improve model calibration.

---

## Project Status

| Phase | Status |
|-------|--------|
| 0 — Environment Setup | Complete |
| 1 — Data Exploration | Complete |
| 2 — Feature Engineering | Complete (8 features) |
| 3 — Model Training | Complete (422k rows, ROC-AUC 0.847) |
| 4 — Django Module | Complete (models, signals, REST, GraphQL, migrations) |
| 5 — FHIR Extensions | Complete (`fhir_extensions.py`) |
| 6 — Feedback Loop | Complete (`retrain_fraud_model` management command) |
| 7 — Unit Tests | Complete (36 tests passing) |
| 8 — Frontend Badge | Complete |
| 9 — Performance Report | Complete (see [models/README.md](models/README.md)) |
| 10 — Documentation | Complete |
| 11 — Demo Preparation | Complete |

---

## Draft PR

> Link to be added once the draft PR is opened on `openimis/openimis-be_py`.
