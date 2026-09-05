import requests
import json
import os
import time
import glob

BASE = 'http://127.0.0.1:8000/api'

# Find an image to test with
imgs = glob.glob('uploads/original/*')
if not imgs:
    print('ERROR: No images in uploads/original/')
    exit(1)

img_path = imgs[0]
print('Using image:', img_path)

# Upload image
with open(img_path, 'rb') as f:
    resp = requests.post(
        BASE + '/images/upload',
        files={'file': (os.path.basename(img_path), f, 'image/jpeg')}
    )
if resp.status_code not in (200, 201):
    print('Upload failed:', resp.text[:300])
    exit(1)
img = resp.json()
image_id = img['image_id']
print('Uploaded: image_id=' + image_id)

# Create session
resp = requests.post(BASE + '/sessions', json={'title': 'Phase5 Test', 'mode': 'single'})
session_id = resp.json()['session_id']
print('Session created: ' + session_id)

# Link image to session
resp = requests.post(BASE + '/sessions/' + session_id + '/images/' + image_id)
data = resp.json()
print('Image linked: ' + data.get('message', ''))

# === Q1 via POST /api/analysis (Phase 4 path) ===
print()
print('=== TEST 1: Q1 via POST /api/analysis ===')
resp = requests.post(BASE + '/analysis', json={
    'image_id': image_id,
    'query': 'What is the dominant land cover type visible in this image?',
    'session_id': session_id,
    'analysis_type': 'general'
})
if resp.status_code != 200:
    print('Q1 FAILED:', resp.text[:300])
    exit(1)
r1 = resp.json()
print('  Status : ' + r1['status'])
ans = str(r1.get('answer', ''))
print('  Answer : ' + ans[:120] + '...')
assert r1['status'] == 'completed', 'Q1 not completed'
print('  PASS Q1')
time.sleep(2)

# === Q2 via POST /api/sessions/{id}/ask (Phase 5 endpoint) ===
print()
print('=== TEST 2: Q2 via POST /api/sessions/{id}/ask ===')
resp = requests.post(
    BASE + '/sessions/' + session_id + '/ask',
    json={
        'question': 'Based on what you described, are there signs of urban development?',
        'analysis_type': 'general'
    }
)
if resp.status_code != 200:
    print('Q2 FAILED:', resp.text[:300])
    exit(1)
r2 = resp.json()
print('  Status         : ' + r2['status'])
print('  question_number: ' + str(r2['question_number']))
print('  image_id       : ' + r2['image_id'])
ans2 = str(r2.get('answer', ''))
print('  Answer         : ' + ans2[:120] + '...')
assert r2['status'] == 'completed', 'Q2 not completed'
assert r2['image_id'] == image_id, 'Q2 used wrong image! ' + r2['image_id']
assert r2['question_number'] >= 2, 'Q2 position wrong'
print('  PASS Q2 - same image reused, question_number=' + str(r2['question_number']))
time.sleep(2)

# === Q3 via /ask (context from Q1+Q2) ===
print()
print('=== TEST 3: Q3 via /ask with Q1+Q2 context ===')
resp = requests.post(
    BASE + '/sessions/' + session_id + '/ask',
    json={
        'question': 'Comparing with your previous answers, what is the single most notable feature?',
        'analysis_type': 'general'
    }
)
if resp.status_code != 200:
    print('Q3 FAILED:', resp.text[:300])
    exit(1)
r3 = resp.json()
print('  Status         : ' + r3['status'])
print('  question_number: ' + str(r3['question_number']))
ans3 = str(r3.get('answer', ''))
print('  Answer         : ' + ans3[:120] + '...')
assert r3['status'] == 'completed', 'Q3 not completed'
assert r3['question_number'] >= 3, 'Q3 position wrong'
print('  PASS Q3 - context-aware answer')

# === TEST 4: Verify MongoDB persistence ===
print()
print('=== TEST 4: MongoDB persistence check ===')
resp = requests.get(BASE + '/sessions/' + session_id)
s = resp.json()
questions = s['questions']
completed_qs = [q for q in questions if q['status'] == 'completed']
print('  Total questions   : ' + str(len(questions)))
print('  Completed Q&As    : ' + str(len(completed_qs)))
for i, q in enumerate(completed_qs, 1):
    text = q['question'][:60]
    print('  Q' + str(i) + ': ' + repr(text))
assert len(completed_qs) >= 3, 'Expected >=3 completed Qs, got ' + str(len(completed_qs))
print('  PASS - all Q&A persisted in MongoDB')

# === Summary ===
print()
print('=' * 50)
print('ALL PHASE 5 TESTS PASSED')
print('session_id : ' + session_id)
print('image_id   : ' + image_id)
print('=' * 50)
