"""
Phase 5 — Full conversation acceptance test.
Tests Q1 → Q2 → Q3 with same session and image reuse.
"""
import requests
import time
import glob
import os

BASE = 'http://127.0.0.1:8000/api'

imgs = glob.glob('uploads/original/*')
if not imgs:
    print('ERROR: No images in uploads/original/')
    exit(1)

img_path = imgs[0]
print('Using image:', img_path)

# Upload image
with open(img_path, 'rb') as f:
    r = requests.post(BASE + '/images/upload', files={'file': (os.path.basename(img_path), f, 'image/jpeg')})
image_id = r.json()['image_id']
print('image_id:', image_id)

# Create session
r = requests.post(BASE + '/sessions', json={'title': 'Conversation Test', 'mode': 'single'})
session_id = r.json()['session_id']
print('session_id:', session_id)

# Link image
r = requests.post(BASE + '/sessions/' + session_id + '/images/' + image_id)
print('Image linked:', r.json().get('message', ''))

# Q1 — via /api/analysis (simulating the first question which goes through analyzeImage)
print()
print('=== Q1: Is vegetation healthy? ===')
r = requests.post(BASE + '/analysis', json={
    'image_id': image_id,
    'query': 'Is vegetation healthy?',
    'session_id': session_id,
    'analysis_type': 'general'
})
q1 = r.json()
print('Status:', q1['status'])
print('Answer[:100]:', str(q1.get('answer', ''))[:100])
assert q1['status'] == 'completed', 'Q1 failed'
print('PASS Q1')
time.sleep(2)

# Q2 — via /ask (follow-up, image auto-resolved)
print()
print('=== Q2: Which area has the healthiest vegetation? ===')
r = requests.post(BASE + '/sessions/' + session_id + '/ask', json={
    'question': 'Which area has the healthiest vegetation?',
    'analysis_type': 'general'
})
q2 = r.json()
print('Status:', q2['status'])
print('question_number:', q2['question_number'])
print('image_id same?', q2['image_id'] == image_id)
print('Answer[:100]:', str(q2.get('answer', ''))[:100])
assert q2['status'] == 'completed', 'Q2 failed'
assert q2['image_id'] == image_id, 'Q2 used wrong image!'
assert q2['question_number'] >= 2, 'Q2 position wrong'
print('PASS Q2')
time.sleep(2)

# Q3 — via /ask (context from Q1+Q2)
print()
print('=== Q3: Are there signs of water stress? ===')
r = requests.post(BASE + '/sessions/' + session_id + '/ask', json={
    'question': 'Are there signs of water stress?',
    'analysis_type': 'general'
})
q3 = r.json()
print('Status:', q3['status'])
print('question_number:', q3['question_number'])
print('Answer[:100]:', str(q3.get('answer', ''))[:100])
assert q3['status'] == 'completed', 'Q3 failed'
assert q3['question_number'] >= 3, 'Q3 position wrong'
print('PASS Q3')

# Check MongoDB persistence
print()
print('=== MongoDB Persistence Check ===')
r = requests.get(BASE + '/sessions/' + session_id)
s = r.json()
qs = s['questions']
completed = [q for q in qs if q['status'] == 'completed']
print('Total questions:', len(qs))
print('Completed Q&As:', len(completed))
for i, q in enumerate(qs):
    text = q['question'][:55]
    print('  Q' + str(i+1) + ': ' + repr(text) + ' | status=' + q['status'])

# Verify no duplicates (Q2+ should not be saved twice)
q_texts = [q['question'].strip() for q in qs]
assert len(q_texts) == len(set(q_texts)), 'DUPLICATE QUESTIONS DETECTED!'
print('No duplicate questions: PASS')
assert len(completed) >= 3, 'Expected >=3 completed, got ' + str(len(completed))
print('All Q&A persisted: PASS')

# Verify image not re-uploaded for Q2/Q3
print()
print('=== Image reuse verification ===')
assert s['image_ids'] == [image_id], 'Session has wrong image_ids: ' + str(s['image_ids'])
print('Single image in session:', image_id)
print('Image NOT re-uploaded for Q2/Q3: PASS')

print()
print('=' * 50)
print('ALL PHASE 5 ACCEPTANCE TESTS PASSED')
print('session_id:', session_id)
print('image_id:', image_id)
print('=' * 50)
