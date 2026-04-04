"""
Liveliness-AI — Standalone Test for File Storage
================================================
Run this file to verify that file_service.py is working correctly.
"""

import asyncio
import io
import os
import sys

# Ensure Python can find the 'app' module from the root directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.file_service import save_file

class MockUploadFile:
    """A dummy class to mimic FastAPI's UploadFile object."""
    def __init__(self, filename: str, content: bytes):
        self.filename = filename
        self.file = io.BytesIO(content)

async def run_tests():
    print("\n" + "="*50)
    print("🚀 Starting File Storage Test")
    print("="*50)
    
    # 1. Create a fake video file in memory
    dummy_file = MockUploadFile(
        filename="test_deepfake_video.mp4",
        content=b"This is fake binary data representing a video."
    )
    
    try:
        # 2. Pass it to your service module
        result = await save_file(dummy_file)
        
        print("\n✅ SUCCESS: File saved correctly to the temp folder!")
        print(f"➔ Generated UUID Filename : {result.filename}")
        print(f"➔ Full Saved Path       : {result.file_path}")
        
        # 3. Read the file back from disk to verify contents
        with open(result.file_path, "rb") as f:
            saved_bytes = f.read()
            print(f"➔ Verified Content      : {saved_bytes}")
            
        # 4. Clean up the test file so it doesn't clutter your temp folder
        os.remove(result.file_path)
        print("\n🧹 Cleanup: Test file successfully removed from disk.")
        
    except Exception as e:
        print(f"\n❌ FAILED: An error occurred during testing.")
        print(f"Error details: {e}")

    print("="*50 + "\n")

if __name__ == "__main__":
    # Run the asynchronous test
    asyncio.run(run_tests())