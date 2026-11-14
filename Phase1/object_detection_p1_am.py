
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

# STEP 1: Locate the images and declare output folder

# Directory containing the input PNG images
phase1_folder = Path(__file__).resolve().parent
data_folder = phase1_folder / "test_data_p1"

# Directory where processed images will be saved
output_folder = phase1_folder / "detected_images"
output_folder.mkdir(parents=True, exist_ok=True)

image_paths = sorted(data_folder.glob("*.png"))

print(f"Found {len(image_paths)} PNG images.")

# STEP 2: Process each image
for image_path in image_paths:

    # Read the image
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read {image_path.name}")
        continue

    # STEP 3: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # STEP 4: Convert to black and white
    # Dark pixels become white (foreground)
    # Light pixels become black (background)
    _, binary = cv2.threshold(
        gray, 127, 255, cv2.THRESH_BINARY_INV
    )

    # STEP 5: Find contours
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Copy original image for visualization
    result = image.copy()

    # STEP 6: Draw bounding box
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(largest_contour)

        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        print(f"{image_path.name}: Bounding box = {(x, y, w, h)}")

    else:
        print(f"{image_path.name}: No object detected.")

    # STEP 7: Save the detected image

    output_path = output_folder / image_path.name

    cv2.imwrite(str(output_path), result)

    print(f"Saved: {output_path}")

    # # STEP 7: Visualize each processing stage
    # fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # axes[0].set_title("Original Image")

    # axes[1].imshow(binary, cmap="gray")
    # axes[1].set_title("Binary Image")

    # axes[2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    # axes[2].set_title("Detected Object")

    # for ax in axes:
    #     ax.axis("off")

    # plt.suptitle(image_path.name)
    # plt.tight_layout()
    # plt.show()
