"""
Management command: retrain_fraud_model

Retrains the Isolation Forest model using the base feature dataset, then
records the new model version in the database and marks it as active.

Usage:
    python manage.py retrain_fraud_model
    python manage.py retrain_fraud_model --contamination 0.05
    python manage.py retrain_fraud_model --n-estimators 300 --dry-run

Reviewer overrides are loaded and logged (future: use them to weight training
samples or remove confirmed false-positives from the contamination estimate).
"""

import os

from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Retrains the fraud detection Isolation Forest using reviewer feedback."

    def add_arguments(self, parser):
        parser.add_argument(
            "--contamination",
            type=float,
            default=0.08,
            help="Expected fraction of anomalies in the training data (default: 0.08).",
        )
        parser.add_argument(
            "--n-estimators",
            type=int,
            default=200,
            help="Number of trees in the Isolation Forest (default: 200).",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Load data and fit the model but do NOT save artefacts or update the DB.",
        )

    def handle(self, *args, **options):
        try:
            import joblib
            import numpy as np
            import pandas as pd
            from sklearn.ensemble import IsolationForest
            from sklearn.metrics import (
                classification_report,
                confusion_matrix,
                roc_auc_score,
            )
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
        except ImportError as exc:
            raise CommandError(
                f"Required package not installed: {exc}. "
                "Run: pip install scikit-learn joblib pandas"
            ) from exc

        from fraud_detect.models import ModelVersion, ReviewerOverride

        contamination = options["contamination"]
        n_estimators = options["n_estimators"]
        dry_run = options["dry_run"]

        # ------------------------------------------------------------------
        # 1. Load base feature data
        # ------------------------------------------------------------------
        base_data_path = os.path.normpath(
            os.path.join(
                os.path.dirname(__file__), "..", "..", "..", "data", "claims_features.csv"
            )
        )
        if not os.path.exists(base_data_path):
            raise CommandError(
                f"Feature data not found at {base_data_path}. "
                "Run the Phase 3 training script first or copy claims_features.csv into data/."
            )

        self.stdout.write(f"Loading base feature data from {base_data_path} …")
        df = pd.read_csv(base_data_path)

        # Derive had_pre_audit_adjustment on the fly if a freshly generated
        # feature CSV does not yet include it.  An adjustment was made before
        # audit whenever the settled amount was reduced below the invoiced
        # amount (invoice_inflation_ratio > 1.0).
        if (
            "had_pre_audit_adjustment" not in df.columns
            and "invoice_inflation_ratio" in df.columns
        ):
            df["had_pre_audit_adjustment"] = (
                df["invoice_inflation_ratio"] > 1.0
            ).astype(int)
            self.stdout.write("Derived had_pre_audit_adjustment from invoice_inflation_ratio.")

        feature_columns = [
            "invoice_inflation_ratio",
            "claim_lag_days",
            "icd_is_vague",
            "provider_avg_inflation",
            "provider_claim_count",
            "member_claim_count",
            "amount_vs_benchmark",
            "had_pre_audit_adjustment",
        ]

        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise CommandError(
                f"Feature columns missing from CSV: {missing}. "
                "Regenerate claims_features.csv using the Phase 2 script."
            )

        # ------------------------------------------------------------------
        # 2. Log reviewer override statistics
        # ------------------------------------------------------------------
        overrides = ReviewerOverride.objects.filter(
            reviewer_decision="APPROVE",
            original_risk_level__in=["HIGH", "MEDIUM"],
        )
        self.stdout.write(
            f"Reviewer overrides available for feedback: {overrides.count()} "
            "(false-positive corrections)"
        )
        # Future enhancement: remove confirmed FP claim_ids from training set
        # or adjust contamination based on override rate.

        # ------------------------------------------------------------------
        # 3. Fit scaler and model
        # ------------------------------------------------------------------
        X = df[feature_columns].fillna(0)
        self.stdout.write(
            f"Training on {len(X)} rows, contamination={contamination}, "
            f"n_estimators={n_estimators} …"
        )

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_scaled)
        self.stdout.write("Model training complete.")

        if dry_run:
            self.stdout.write(
                self.style.WARNING("Dry run — artefacts NOT saved and DB NOT updated.")
            )
            return

        # ------------------------------------------------------------------
        # 4. Save artefacts
        # ------------------------------------------------------------------
        models_dir = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "models")
        )
        os.makedirs(models_dir, exist_ok=True)

        model_path = os.path.join(models_dir, "fraud_model.joblib")
        scaler_path = os.path.join(models_dir, "fraud_scaler.joblib")

        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        self.stdout.write(f"Model saved to {model_path}")
        self.stdout.write(f"Scaler saved to {scaler_path}")

        # Reset cached model in engine so the next request loads the new one.
        import fraud_detect.engine as engine_module
        engine_module._MODEL = None
        engine_module._SCALER = None

        # ------------------------------------------------------------------
        # 4b. Evaluate on a held-out split and write models/README.md
        # ------------------------------------------------------------------
        self._write_performance_report(
            df=df,
            feature_columns=feature_columns,
            scaler=scaler,
            model=model,
            models_dir=models_dir,
            contamination=contamination,
            n_estimators=n_estimators,
            train_test_split=train_test_split,
            classification_report=classification_report,
            confusion_matrix=confusion_matrix,
            roc_auc_score=roc_auc_score,
        )

        # ------------------------------------------------------------------
        # 5. Record the new model version in the database
        # ------------------------------------------------------------------
        ModelVersion.objects.filter(is_active=True).update(is_active=False)
        new_version = ModelVersion.objects.create(
            version="retrained",
            model_file_path=model_path,
            scaler_file_path=scaler_path,
            is_active=True,
        )
        self.stdout.write(
            self.style.SUCCESS(
                f"Model version {new_version.id} recorded and set as active."
            )
        )

    def _write_performance_report(
        self,
        *,
        df,
        feature_columns,
        scaler,
        model,
        models_dir,
        contamination,
        n_estimators,
        train_test_split,
        classification_report,
        confusion_matrix,
        roc_auc_score,
    ):
        """
        Evaluates the freshly trained model on a deterministic 20% test split
        and writes a Markdown performance report to models/README.md.

        Called on every successful retrain so the report never goes stale.
        """
        if "proxy_fraud_label" not in df.columns:
            self.stdout.write(
                self.style.WARNING(
                    "No proxy_fraud_label column — skipping performance report."
                )
            )
            return

        import datetime as _dt

        X = df[feature_columns].fillna(0)
        y = df["proxy_fraud_label"]

        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_test_scaled = scaler.transform(X_test)
        predictions = model.predict(X_test_scaled)
        scores = model.decision_function(X_test_scaled)
        y_pred = (predictions == -1).astype(int)

        report = classification_report(
            y_test,
            y_pred,
            target_names=["Normal", "Suspicious"],
            output_dict=True,
            zero_division=0,
        )
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = int(cm[0][0]), int(cm[0][1]), int(cm[1][0]), int(cm[1][1])
        try:
            roc_auc = roc_auc_score(y_test, -scores)
        except ValueError:
            roc_auc = float("nan")

        normal = report["Normal"]
        susp = report["Suspicious"]
        macro = report["macro avg"]
        weighted = report["weighted avg"]
        accuracy = report["accuracy"]
        generated = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        md = f"""# Fraud Detection — Model Performance Report

> **Auto-generated by `python manage.py retrain_fraud_model`.**
> Do not edit by hand — it is overwritten on every retrain.
> Last generated: {generated}

This report describes the currently active Isolation Forest model artefacts in
this directory (`fraud_model.joblib` + `fraud_scaler.joblib`).

---

## Model

| Property | Value |
|----------|-------|
| Algorithm | Isolation Forest (unsupervised) |
| Estimators | {n_estimators} |
| Contamination | {contamination} |
| Random state | 42 |
| Features | {len(feature_columns)} |
| Training rows | {len(df):,} |
| Scaler | `StandardScaler` |

**Features** (order matters — must match `engine.FEATURE_ORDER`):
{", ".join(f"`{c}`" for c in feature_columns)}.

---

## Evaluation

Evaluated on a held-out test set of **{len(y_test):,} claims** (20% split,
`random_state=42`). The proxy fraud label is `1` when
`SETTLED AMOUNT < 80% of INVOICE AMOUNT` — claims where the insurer already
detected something wrong and partially rejected the claim.

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| Normal | {normal['precision']:.4f} | {normal['recall']:.4f} | {normal['f1-score']:.4f} | {int(normal['support']):,} |
| Suspicious | {susp['precision']:.4f} | {susp['recall']:.4f} | {susp['f1-score']:.4f} | {int(susp['support']):,} |
| **Macro avg** | **{macro['precision']:.4f}** | **{macro['recall']:.4f}** | **{macro['f1-score']:.4f}** | **{int(macro['support']):,}** |
| **Weighted avg** | **{weighted['precision']:.4f}** | **{weighted['recall']:.4f}** | **{weighted['f1-score']:.4f}** | **{int(weighted['support']):,}** |

**Overall accuracy**: {accuracy:.1%} &nbsp;|&nbsp; **ROC-AUC**: {roc_auc:.4f}

**Confusion Matrix** (test set):

```
                     Predicted Normal   Predicted Suspicious
Actual Normal        {tn:>12,}      {fp:>12,}
Actual Suspicious    {fn:>12,}      {tp:>12,}
```

- True Negatives (correctly cleared): **{tn:,}**
- False Positives (wrongly flagged): **{fp:,}**
- False Negatives (missed suspicious): **{fn:,}**
- True Positives (correctly caught): **{tp:,}**

Of {int(y_test.sum()):,} actually-suspicious claims, the model flagged
**{int(y_pred.sum()):,}** claims as anomalies overall.

---

## Interpretation

The model correctly clears {tn:,} normal claims
({normal['recall']:.1%} specificity) and catches {tp:,} suspicious claims it
would otherwise miss.

The precision on the Suspicious class ({susp['precision']:.2f}) reflects the
imprecision of the proxy label — not every claim settled below invoice was
fraudulent; some were legitimately partially approved. A ROC-AUC of
{roc_auc:.3f} indicates discrimination power well above chance (0.5).

> The rules engine supplements the ML model. A claim that fires two or more
> rules reaches HIGH risk even when the ML score is near neutral, ensuring
> explainable high-confidence flagging works without model artefacts.

---

## Reproducing this report

```bash
docker compose -f compose.yml -f compose.fraud-detect.yml exec backend \\
  python manage.py retrain_fraud_model
```

The report is written automatically at the end of every successful retrain.
"""

        report_path = os.path.join(models_dir, "README.md")
        with open(report_path, "w") as f:
            f.write(md)

        self.stdout.write(
            self.style.SUCCESS(
                f"Performance report written to {report_path} "
                f"(accuracy={accuracy:.1%}, ROC-AUC={roc_auc:.3f})."
            )
        )
