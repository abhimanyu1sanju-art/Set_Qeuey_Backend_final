import requests
import sys
import io
from PIL import Image

base_url = "http://127.0.0.1:8000/api"

print("Generating valid JPEG for upload test...")
img = Image.new("RGB", (100, 100), color="blue")
buf = io.BytesIO()
img.save(buf, format="JPEG")
img_bytes = buf.getvalue()

print("\n1. Testing POST /api/images/upload")
upload_res = requests.post(
    f"{base_url}/images/upload",
    files={"file": ("test_blue.jpg", img_bytes, "image/jpeg")}
)
if upload_res.status_code != 200:
    print(f"Upload failed: {upload_res.status_code} - {upload_res.text}")
    sys.exit(1)

data = upload_res.json()
image_id = data["image_id"]
print(f"Upload success! image_id: {image_id}")
print(f"Data: {data}")

print(f"\n2. Testing GET /api/images/{image_id}")
get_res = requests.get(f"{base_url}/images/{image_id}")
if get_res.status_code != 200:
    print(f"GET failed: {get_res.status_code} - {get_res.text}")
    sys.exit(1)

print(f"GET success! Data: {get_res.json()}")

print(f"\n3. Testing DELETE /api/images/{image_id}")
del_res = requests.delete(f"{base_url}/images/{image_id}")
if del_res.status_code != 200:
    print(f"DELETE failed: {del_res.status_code} - {del_res.text}")
    sys.exit(1)

print(f"DELETE success! Data: {del_res.json()}")

print(f"\n4. Verifying deletion GET /api/images/{image_id}")
verify_res = requests.get(f"{base_url}/images/{image_id}")
if verify_res.status_code == 404:
    print("Verification success! Image no longer exists.")
else:
    print(f"Verification failed: expected 404, got {verify_res.status_code}")
    sys.exit(1)

print("\nAll API endpoints tested successfully.")
