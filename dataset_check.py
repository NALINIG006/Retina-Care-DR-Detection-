import os

dataset_path = r"C:\Users\acer\Downloads\Dataset from fundus images for the study of diabetic retinopathy_V03"

extensions = (".jpg", ".jpeg", ".png", ".bmp")

for folder in sorted(os.listdir(dataset_path)):
    folder_path = os.path.join(dataset_path, folder)

    if os.path.isdir(folder_path):
        count = 0

        for root, dirs, files in os.walk(folder_path):
            count += sum(
                1 for file in files
                if file.lower().endswith(extensions)
            )

        print(f"{folder}: {count} images")