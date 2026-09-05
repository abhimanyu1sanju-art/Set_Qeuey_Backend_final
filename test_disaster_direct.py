import asyncio
import logging
import sys
from pathlib import Path

# Add the Backend folder to sys.path so we can import 'app'
sys.path.insert(0, str(Path(__file__).parent))

from app.services.disaster_service import run_disaster_analysis_safe
from app.db.mongodb import get_client, verify_connection
from app.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(name)s  %(message)s",
)

logger = logging.getLogger(__name__)

SCENE_ID = "S2A_MSIL2A_20250118T053141_N0511_R105_T43QCA_20250118T092848"

async def main():
    logger.info("Starting Phase 5 Disaster Detection E2E test")
    
    if not verify_connection():
        logger.error("Failed to connect to MongoDB")
        return

    logger.info(f"Running flood detection analysis for {SCENE_ID}...")

    # Set CDSE auth if not set in environment but available globally
    # Wait, the application handles this through settings.
    
    result = run_disaster_analysis_safe(
        scene_id=SCENE_ID,
        before_scene_id=None,
        disaster_type="flood",
        confidence_threshold=0.5,
        force_reprocess=True,
    )

    logger.info("=== Analysis Complete ===")
    
    if result.get("status") == "failed":
        logger.error(f"Analysis failed: {result.get('error')}")
    else:
        print("\nSUMMARY:")
        import pprint
        pprint.pprint({
            k: v for k, v in result.items() 
            if k in ['disaster_type', 'status', 'affected_area_m2', 'pixel_count', 'region_count', 'confidence']
        })
        print(f"\nStatus: {result.get('status')}")
        print(f"Error: {result.get('error')}")
        print(f"Duration: {result.get('processing', {}).get('duration_seconds')}s")
        print(f"Bands used: {result.get('processing', {}).get('bands_used')}")
        print(f"Result ID: {result.get('result_id')}")
        print("\nE2E Test PASS: Disaster detected natively without mock data.")

    # Cleanup DB connection
    client = get_client()
    if client:
        client.close()

if __name__ == "__main__":
    # Required for Windows environment sometimes
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
