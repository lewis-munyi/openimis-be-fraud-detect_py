from django.urls import path

from .views import (
    ClaimFraudFlagView,
    FraudFlagListView,
    RescoreClaimView,
    ReviewerOverrideView,
    ScoreClaimView,
)

# openIMIS automatically mounts this module at /api/fraud_detect/ via openimisurls.py.
# Paths here are relative to that prefix, so the full URLs are:
#   /api/fraud_detect/flags/
#   /api/fraud_detect/flags/<claim_id>/
#   /api/fraud_detect/override/
#   /api/fraud_detect/score/
#   /api/fraud_detect/rescore/<claim_id>/

app_name = "fraud_detect"

urlpatterns = [
    # List all flags (with optional ?risk_level= filter and pagination)
    path("flags/", FraudFlagListView.as_view(), name="fraud-flag-list"),
    # Retrieve the flag for a specific claim
    path(
        "flags/<int:claim_id>/",
        ClaimFraudFlagView.as_view(),
        name="fraud-flag-detail",
    ),
    # Submit a reviewer override decision
    path("override/", ReviewerOverrideView.as_view(), name="reviewer-override"),
    # On-demand scoring endpoint (no DB persistence)
    path("score/", ScoreClaimView.as_view(), name="fraud-score"),
    # Re-score an existing claim and persist the result to tbl_FraudFlag
    path(
        "rescore/<int:claim_id>/",
        RescoreClaimView.as_view(),
        name="fraud-rescore",
    ),
]
