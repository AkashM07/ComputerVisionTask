# Computer Vision — Phase 1: Object Detection

## Overview

This project introduces basic computer vision using Python and OpenCV.

**Goal:** Detect a digit on a piece of paper and draw a bounding box around it. We will start with a simple algorithm, understand each step, and gradually improve it.

**Script:** `object_detection_p1_am.py`

The script currently performs **object detection, not digit recognition** (identifying whether a digit is 0–9).

---

## 1. Python Setup

### Install Required Libraries

Run in the VS Code terminal:

`python -m pip install opencv-python matplotlib`

- **OpenCV (`cv2`):** Image processing and object detection.
- **Matplotlib:** Image visualization (if needed).
- **NumPy:** Numerical operations; installed automatically with OpenCV.

### Recommended: Virtual Environment

A virtual environment (`.venv`) keeps project packages separate from your global Python installation. This can prevent dependency conflicts.

**Windows PowerShell:**

1. Create environment: `python -m venv .venv`
2. Activate environment: `.\.venv\Scripts\Activate.ps1`
3. Install packages: `python -m pip install opencv-python matplotlib`
4. Verify installation: `python -c "import cv2; print(cv2.__version__)"`

`Activate.ps1` tells the terminal to use the Python environment inside `.venv`.

**VS Code:** Press `Ctrl + Shift + P` → **Python: Select Interpreter** → select `.venv\Scripts\python.exe`.

This ensures VS Code uses the Python environment containing OpenCV.

**Troubleshooting:** If installation fails with `WinError 2` involving NumPy or `f2py.exe`, create a fresh virtual environment and reinstall the packages inside it.

Add `.venv/` to `.gitignore` to prevent uploading the environment to GitHub.

For macOS/Linux, activate using `source .venv/bin/activate`.

---

## 2. Project Structure

    ComputerVisionTask/
    └── Phase1/
        ├── object_detection_p1_am.py
        ├── test_data_p1/
        │   ├── 00002.png
        │   ├── 00014.png
        │   └── ...
        └── detected_images/

- `test_data_p1/`: Contains input PNG images.
- `detected_images/`: Stores processed images with bounding boxes.

Run from the project directory:

`python Phase1/object_detection_p1_am.py`

---

## 3. How the Script Works

### Step 1 — Locate Images

`data_folder = Path(__file__).resolve().parent / "test_data_p1"`

`image_paths = sorted(data_folder.glob("*.png"))`

- `Path(__file__).resolve().parent` locates the script's directory.
- `.glob("*.png")` finds all PNG files in the folder.
- `sorted()` orders file paths alphabetically (lexicographically).
- `image_paths` is a Python list of file paths, not image data.

### Step 2 — Read Images

`image = cv2.imread(str(image_path))`

Loads each PNG as a NumPy array containing pixel values.

### Step 3 — Convert to Grayscale

`gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)`

Converts the image from three color channels into one brightness channel, simplifying processing.

### Step 4 — Binary Thresholding

`_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)`

Separates the digit from the paper:

- Bright pixels (greater than 127) → Black (0).
- Dark pixels (127 or below) → White (255).

The digit ideally becomes white against a black background.

### Step 5 — Find Contours

`contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)`

A contour represents the boundary of a detected object.

- `RETR_EXTERNAL`: Retrieves outer contours.
- `CHAIN_APPROX_SIMPLE`: Reduces unnecessary contour points.

### Step 6 — Select Largest Contour

`largest_contour = max(contours, key=cv2.contourArea)`

Selects the contour with the largest area, assuming it corresponds to the digit.

**Limitation:** Shadows or borders could be detected instead.

### Step 7 — Calculate and Draw Bounding Box

`x, y, w, h = cv2.boundingRect(largest_contour)`

Returns four values describing the rectangle:

- `x`: Horizontal position of the top-left corner.
- `y`: Vertical position of the top-left corner.
- `w`: Width of the rectangle.
- `h`: Height of the rectangle.

**The four vertices are:**

| Corner | Coordinates |
|---|---|
| Top-left | `(x, y)` |
| Top-right | `(x+w, y)` |
| Bottom-left | `(x, y+h)` |
| Bottom-right | `(x+w, y+h)` |

Draw the rectangle:

`cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)`

- Two opposite corners define the rectangle.
- `(0, 255, 0)` specifies green in BGR format.
- `2` specifies line thickness in pixels.

**Note:** Image coordinates begin at the top-left `(0,0)`. Positive `x` points right; positive `y` points downward.

### Step 8 — Save Results

`output_path = output_folder / image_path.name`

`cv2.imwrite(str(output_path), result)`

Saves the processed PNG with its bounding box inside `detected_images/`.

The output directory is created using:

`output_folder.mkdir(parents=True, exist_ok=True)`

This creates the folder if it does not already exist.

---

## 4. Testing and Debugging

Begin with 3–5 PNG images and examine:

- Does thresholding isolate the digit?
- Is the largest contour actually the digit?
- Does the bounding box correctly surround it?
- How do different digit sizes and positions affect detection?

**Current limitations:** The algorithm assumes dark digits on light paper, a fixed threshold of 127, and one primary object.

The goal is to identify where these assumptions fail before adding complexity.

---

## 5. Useful Git Commands

| Command | Purpose |
|---|---|
| `git status` | Check modified files. |
| `git branch --show-current` | See your current branch. |
| `git checkout <branch-name>` | Switch to an existing branch. |
| `git checkout -b <new-branch-name>` | Create and enter a new branch. |
| `git add .` | Stage changes. |
| `git commit -m "message"` | Save a local commit. |
| `git push -u origin <branch-name>` | Push a branch to GitHub and set upstream tracking. |

### Recommended Workflow

1. `git checkout main` — Switch to main.
2. `git pull origin main` — Get the latest code.
3. `git checkout -b my-branch` — Create your working branch.
4. Make and test changes.
5. `git add .` — Stage changes.
6. `git commit -m "Update object detection"` — Commit.
7. `git push -u origin my-branch` — Upload branch.

**Important:** Pushing a branch to GitHub does not merge it into `main`. Create a pull request for review and merging.

---

## 6. Future Improvements

Once the current pipeline is understood:

1. Improve thresholding and contour detection.
2. Extract and resize detected digits.
3. Implement Histogram of Oriented Gradients (HOG) features.
4. Train a Support Vector Machine (SVM) to recognize digits 0–9.

**Project philosophy: Start simple → Test → Debug → Understand → Improve.**