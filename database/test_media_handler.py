"""
test_media_handler.py
Quick smoke-test for all three parts of media_handler.py
Run from the project root:  python database/test_media_handler.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from media_handler import detect_file_type, store_metadata, fetch_by_id

# ── Part 1: File Type Detection ──────────────────────────────────────────────
print("\n=== Part 1 — detect_file_type ===")

test_files = [
    ("photo.jpg",     "image"),
    ("photo.JPEG",    "image"),   # case-insensitive
    ("banner.png",    "image"),
    ("clip.mp4",      "video"),
    ("recording.avi", "video"),
    ("screencap.mov", "video"),
    ("voice.wav",     "audio"),
    ("song.mp3",      "audio"),
    ("data.csv",      "unknown"), # graceful fallback
    ("noextension",   "unknown"),
]

all_passed = True
for filename, expected in test_files:
    result = detect_file_type(filename)
    status = "✓" if result == expected else "✗"
    if result != expected:
        all_passed = False
    print(f"  {status}  {filename:<20} → {result}  (expected: {expected})")

print(f"\nPart 1 result: {'ALL PASSED' if all_passed else 'SOME FAILED'}")

# ── Part 2 & 3: Store + Retrieve ─────────────────────────────────────────────
print("\n=== Parts 2 & 3 — store_metadata + fetch_by_id ===")

samples = [
    ("interview.mp4",   "video",  "/uploads/video/interview.mp4"),
    ("portrait.jpg",    "image",  "/uploads/image/portrait.jpg"),
    ("statement.wav",   "audio",  "/uploads/audio/statement.wav"),
]

for filename, file_type, file_path in samples:
    record_id = store_metadata(filename, file_type, file_path)
    assert record_id is not None, f"Insert failed for {filename}"

    record = fetch_by_id(record_id)
    assert record is not None, f"Fetch failed for ID {record_id}"

    print(f"\n  Inserted record #{record_id}:")
    for key, value in record.items():
        print(f"    {key:<14}: {value}")

print("\n=== All tests completed ===\n")