import os

dataset_path = r"C:\Users\acer\Downloads\Dataset from fundus images for the study of diabetic retinopathy_V03"

for folder in sorted(os.listdir(dataset_path)):
    full_path = os.path.join(dataset_path, folder)

    if os.path.isdir(full_path):
        count = sum(
            1 for file in os.listdir(full_path)
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        )
        print(folder, "=", count, "images")