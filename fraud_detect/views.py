from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from django.shortcuts import get_object_or_404

from .engine import compute_risk_level, score_claim_ml
from .models import FraudFlag, ReviewerOverride
from .rules import evaluate_rules
from .serializers import FraudFlagSerializer, ReviewerOverrideSerializer
from .signals import _build_claim_dict


class ClaimFraudFlagView(APIView):
    """
    GET /api/fraud/flags/{claim_id}/

    Returns the current fraud flag assessment for a specific claim.
    Returns 404 if the claim has not been scored yet (no FraudFlag row exists).
    """

    def get(self, request, claim_id):
        flag = get_object_or_404(FraudFlag, claim_id=claim_id)
        serializer = FraudFlagSerializer(flag)
        return Response(serializer.data)


class FraudFlagListView(APIView):
    """
    GET /api/fraud/flags/?risk_level=HIGH&limit=100&offset=0

    Returns fraud flags, optionally filtered by risk level.
    Supports pagination via `limit` and `offset` query parameters.
    """

    def get(self, request):
        risk_level = request.query_params.get("risk_level")
        try:
            limit = int(request.query_params.get("limit", 100))
            offset = int(request.query_params.get("offset", 0))
        except (TypeError, ValueError):
            return Response(
                {"detail": "limit and offset must be integers."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Clamp limit to prevent unbounded queries
        limit = min(max(limit, 1), 1000)
        offset = max(offset, 0)

        queryset = FraudFlag.objects.all().order_by("-created_at")
        if risk_level:
            if risk_level not in {"HIGH", "MEDIUM", "LOW"}:
                return Response(
                    {"detail": "risk_level must be HIGH, MEDIUM, or LOW."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            queryset = queryset.filter(overall_risk_level=risk_level)

        total = queryset.count()
        page = queryset[offset: offset + limit]
        serializer = FraudFlagSerializer(page, many=True)
        return Response({
            "count": total,
            "limit": limit,
            "offset": offset,
            "results": serializer.data,
        })


class ReviewerOverrideView(APIView):
    """
    POST /api/fraud/override/

    Records a reviewer's decision to override the model's assessment.

    Expected request body (JSON):
      {
        "claim_id":          <int>,
        "reviewer_decision": "APPROVE" | "REJECT" | "ESCALATE",
        "reviewer_id":       <int>,
        "notes":             "<str, optional>"
      }
    """

    def post(self, request):
        serializer = ReviewerOverrideSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        claim_id = serializer.validated_data.get("claim_id")
        flag = get_object_or_404(FraudFlag, claim_id=claim_id)

        override = ReviewerOverride.objects.create(
            claim_id=claim_id,
            fraud_flag=flag,
            original_risk_level=flag.overall_risk_level,
            reviewer_decision=serializer.validated_data["reviewer_decision"],
            reviewer_id=serializer.validated_data["reviewer_id"],
            notes=serializer.validated_data.get("notes", ""),
        )

        return Response(
            ReviewerOverrideSerializer(override).data,
            status=status.HTTP_201_CREATED,
        )


class ScoreClaimView(APIView):
    """
    POST /api/fraud/score/

    Scores a raw claim dict on demand (without requiring a saved Claim row).
    Useful for previewing a score before submitting a claim, or for testing.

    Expected request body (JSON): claim fields dict (same keys as engine.py).
    """

    def post(self, request):
        claim_dict = request.data
        rules_result = evaluate_rules(claim_dict)
        ml_result = score_claim_ml(claim_dict)
        risk_level = compute_risk_level(rules_result, ml_result)

        return Response({
            "rules": rules_result,
            "ml": ml_result,
            "overall_risk_level": risk_level,
        })


class RescoreClaimView(APIView):
    """
    POST /api/fraud_detect/rescore/{claim_id}/

    Fetches the live Claim row from the database, scores it through both the
    rules engine and the ML model, and persists the result to tbl_FraudFlag
    (insert on first call, update on subsequent calls).

    Returns the saved FraudFlag record.  Returns 404 if the claim_id does not
    exist in the Claim table, or 503 if the claim module is not installed.
    """

    def post(self, request, claim_id):
        try:
            from claim.models import Claim
        except ImportError:
            return Response(
                {"detail": "The claim module is not installed in this environment."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        claim = get_object_or_404(Claim, id=claim_id)

        claim_dict = _build_claim_dict(claim)
        rules_result = evaluate_rules(claim_dict)
        ml_result = score_claim_ml(claim_dict)
        risk_level = compute_risk_level(rules_result, ml_result)

        flag, created = FraudFlag.objects.update_or_create(
            claim_id=claim_id,
            defaults={
                "is_rule_flagged": rules_result["is_flagged"],
                "rule_flag_reasons": rules_result["fired_rules"],
                "anomaly_score": ml_result["anomaly_score"],
                "is_ml_anomaly": ml_result["is_anomaly"],
                "overall_risk_level": risk_level,
            },
        )

        return Response(
            FraudFlagSerializer(flag).data,
            status=status.HTTP_201_CREATED if created else status.HTTP_200_OK,
        )
