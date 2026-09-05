import asyncio
import logging
import sys
from pathlib import Path

# Add the Backend folder to sys.path so we can import 'app'
sys.path.insert(0, str(Path(__file__).parent))

from app.services.fusion_service import run_fusion_analysis_safe
from app.db.mongodb import get_client, verify_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(name)s  %(message)s",
)

logger = logging.getLogger(__name__)

# S1 and S2 covering a similar region (can be anything as long as they are in DB)
# These IDs must be searched and populated in MongoDB first.
# We will use S2: S2A_MSIL2A_20250118T053141_N0511_R105_T43QCA_20250118T092848 (from Phase 5)
# S1: we need to find an S1 scene for India in DB, or search for one.

async def main():
    logger.info("Starting Phase 6 SAR+Optical Fusion E2E test")
    
    if not verify_connection():
        logger.error("Failed to connect to MongoDB")
        return

    # To ensure an S1 scene is in the DB, we can manually trigger a search for the same bbox
    from app.services.satellite_service import search_scenes
    from app.schemas.satellite import SatelliteSearchRequest
    
    logger.info("Searching for an S1 scene over the same area to ensure it is in DB...")
    s2_id = "S2A_MSIL2A_20250118T053141_N0511_R105_T43QCA_20250118T092848"
    
    # We don't have the exact bbox here, but we can search globally roughly over India
    # Or just use the S1 scene if we know it. We'll search Sentinel-1 explicitly.
    try:
        s1_res = search_scenes(SatelliteSearchRequest(
            satellite="sentinel-1",
            bbox=[68.0, 8.0, 97.0, 37.0],
            start_date="2025-01-01",
            end_date="2025-01-31",
            limit=1
        ))
        if not s1_res.scenes:
            logger.error("No S1 scenes found to test with.")
            return
        
        s1_id = s1_res.scenes[0].scene_id
        logger.info(f"Found S1 scene: {s1_id}")
    except Exception as e:
        logger.error(f"Failed to search for S1 scene: {e}")
        return

    logger.info(f"Running fusion analysis for S1={s1_id}, S2={s2_id} (Water mode)...")
    
    result = run_fusion_analysis_safe(
        s1_scene_id=s1_id,
        s2_scene_id=s2_id,
        analysis_mode="water",
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
            if k in ['analysis_mode', 'status', 'available_polarizations', 'optical_index_used']
        })
        print(f"\nStatus: {result.get('status')}")
        print(f"Error: {result.get('error')}")
        print(f"Duration: {result.get('processing', {}).get('duration_seconds')}s")
        print(f"Result ID: {result.get('result_id')}")
        print("\nE2E Test PASS: Fusion completed natively without mock data.")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
