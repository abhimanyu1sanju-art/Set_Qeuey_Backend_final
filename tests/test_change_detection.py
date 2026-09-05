import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_change_analyze_missing_auth(mocker):
    # Test that without Copernicus credentials (or if mocked to fail), the endpoint returns 503
    mocker.patch("app.services.cdse_auth.get_token", return_value=None)
    
    response = client.post(
        "/api/change/analyze",
        json={
            "before_scene_id": "S2A_MSIL2A_20250113T052211_N0511_R062_T43QFC_20250113T090333",
            "after_scene_id": "S2B_MSIL2A_20250118T052159_N0511_R062_T43QFC_20250118T080031",
            "method": "abs_diff",
            "threshold": 0.15,
            "force_reprocess": False
        }
    )
    
    assert response.status_code == 503
    assert "Copernicus credentials missing" in response.json()["detail"]

def test_get_change_results_not_found():
    response = client.get("/api/change/results/non_existent_id")
    assert response.status_code == 404

def test_get_change_preview_not_found():
    response = client.get("/api/change/preview/non_existent_id")
    assert response.status_code == 404
