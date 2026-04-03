from app.ai_engine.image_ela import process_image

file_path = "uploads/test.jpg"  # put your image here

score, explanation = process_image(file_path)

print("Score:", score)
print("Explanation:", explanation)