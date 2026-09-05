"""
SatQuery AI — Phase 4 Infrastructure Detection Tests
"""

import pytest
from datetime import datetime, timezone
from app.schemas.infrastructure import (
    InfrastructureAnalysisRequest,
    InfraDetection,
    InfraSummary,
    InfraAnalysisResponse,
)

def test_infrastructure_request_schema():
    req = InfrastructureAnalysisRequest(
        scene_id="S2A_MSIL2A_20250118T053141_N0511_R105_T43QCA_20250118T092848",
        detection_types=["buildings"],
        confidence_threshold=0.8,
        force_reprocess=True,
    )
    assert req.scene_id == "S2A_MSIL2A_20250118T053141_N0511_R105_T43QCA_20250118T092848"
    assert "buildings" in req.detection_types
    assert req.confidence_threshold == 0.8
    assert req.force_reprocess is True

def test_infrastructure_detection_schema():
    det = InfraDetection(
        label="building_cluster",
        pixel_count=42,
        area_m2=4200.0,
        confidence=0.85,
        bbox=[100.0, 200.0, 150.0, 250.0],
        centroid=[125.0, 225.0],
        detection_method="ndbi_segmentation",
    )
    assert det.label == "building_cluster"
    assert det.area_m2 == 4200.0
    assert len(det.bbox) == 4

def test_infrastructure_response_schema():
    now_str = datetime.now(timezone.utc).isoformat()
    resp = InfraAnalysisResponse(
        result_id="inf_123",
        scene_id="S2A_test",
        source="Copernicus Data Space",
        model={"name": "test-model", "version": "1.0", "description": "test"},
        detections={
            "buildings": [],
            "roads": [],
            "construction_change_candidates": [],
        },
        summary=InfraSummary(),
        processing={
            "duration_seconds": 12.5,
            "real_data": True,
            "bands_used": ["B08", "B11"],
            "resolution_m": 10.0,
        },
        status="completed",
        created_at=now_str,
    )
    assert resp.result_id == "inf_123"
    assert resp.status == "completed"
    assert resp.processing.duration_seconds == 12.5
