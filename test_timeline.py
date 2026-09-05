import asyncio
import httpx

async def test_timeline():
    url = "http://127.0.0.1:8000/api/timeline/search"
    payload = {
        "bbox": [73.7, 18.4, 73.9, 18.6],
        "start_date": "2025-01-01",
        "end_date": "2025-01-31",
        "satellite": "sentinel-2",
        "max_cloud_cover": 30,
        "limit": 3
    }
    async with httpx.AsyncClient(timeout=120) as client:
        try:
            print("Fetching timeline...")
            res = await client.post(url, json=payload)
            print(res.status_code)
            print(res.json())
        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    asyncio.run(test_timeline())
