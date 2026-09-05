"""
SatQuery AI — Phase 4 E2E Infrastructure Detection Test

Requirements:
- Valid Copernicus S3 credentials in Backend/.env
- MongoDB running locally (default satquery db)
- Sentinel-2 scene already present in the database:
  S2A_MSIL2A_20250118T053141_N0511_R105_T43QCA_20250118T092848

This script tests the entire pipeline:
1. Connecting to DB and finding the scene
2. Authenticating via S3
3. Reading multiple bands (B04, B08, B11, B02) directly from Copernicus
4. Resampling and calculating spectral indices (NDBI, BSI, NDVI)
5. OpenCV segmentation
6. Saving the PNG preview and MongoDB record
"""

import sys
import logging
from pathlib import Path
from pprint import pprint

# Setup paths to import the app
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings
from app.db.mongodb import verify_connection
from app.services.infrastructure_service import run_infrastructure_analysis

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger(__name__)

SCENE_ID = "S2A_MSIL2A_20250118T053141_N0511_R105_T43QCA_20250118T092848"

def main():
    logger.info("Starting Phase 4 E2E test")
    
    if not verify_connection():
        logger.error("MongoDB is not reachable")
        sys.exit(1)
        
    try:
        logger.info(f"Running infrastructure analysis for {SCENE_ID}...")
        result = run_infrastructure_analysis(
            scene_id=SCENE_ID,
            detection_types=["buildings", "roads", "construction"],
            confidence_threshold=0.5,
            force_reprocess=True,
        )
        
        logger.info("=== Analysis Complete ===")
        print("\nSUMMARY:")
        pprint(result["summary"])
        print(f"\nStatus: {result['status']}")
        print(f"Error: {result.get('error')}")
        print(f"Duration: {result['processing']['duration_seconds']}s")
        print(f"Bands used: {result['processing']['bands_used']}")
        print(f"Result ID: {result['result_id']}")
        
        if result["status"] == "completed":
            print("\nE2E Test PASS: Infrastructure detected natively without mock data.")
            sys.exit(0)
        else:
            print("\nE2E Test FAIL: Analysis did not complete successfully.")
            sys.exit(1)
            
    except Exception as exc:
        logger.error(f"Test failed with exception: {exc}")
        sys.exit(1)

if __name__ == "__main__":
    main()
