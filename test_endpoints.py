import httpx
import sys

base_url = "http://127.0.0.1:8000/api"

def main():
    # 1. Health
    print("Testing GET /api/health")
    r = httpx.get(f"{base_url}/health")
    print(r.status_code, r.json())
    assert r.status_code == 200

    # 2. Upload
    print("\nTesting POST /api/images/upload")
    from PIL import Image
    import io
    img = Image.new('RGB', (100, 100), color = 'red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)

    files = {'file': ('test.jpg', img_byte_arr.read(), 'image/jpeg')}
    r = httpx.post(f"{base_url}/images/upload", files=files)
    print(r.status_code, r.json())
    assert r.status_code == 200
    data = r.json()
    image_id = data['image_id']

    # 3. Get Image
    print(f"\nTesting GET /api/images/{image_id}")
    r = httpx.get(f"{base_url}/images/{image_id}")
    print(r.status_code, r.json())
    assert r.status_code == 200

    # 4. Delete Image
    print(f"\nTesting DELETE /api/images/{image_id}")
    r = httpx.delete(f"{base_url}/images/{image_id}")
    print(r.status_code, r.json())
    assert r.status_code == 200

    # Check deleted
    print(f"\nVerifying deletion GET /api/images/{image_id}")
    r = httpx.get(f"{base_url}/images/{image_id}")
    print(r.status_code, "Expected 404")
    assert r.status_code == 404

    print("\nAll endpoints verified successfully!")

if __name__ == "__main__":
    main()
