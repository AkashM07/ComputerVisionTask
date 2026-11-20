# Phase 1: Digit detector and recognizer (built from scratch)

Finds the digit on an A4 page, draws a box around it, and says which digit
(0-9) it is. Nothing is pretrained and there is no YOLO or neural-network
library. Every learned or computed part is written in NumPy so we can read
and change it:

| Stage | File | What it does |
|---|---|---|
| 1. Detect | `detect.py` | Threshold -> contours -> one box around all the ink |
| 2. Describe | `features.py` | Crop, scale onto a 32x32 canvas, then **HOG written by hand** |
| 3. Classify | `svm.py` | **Multiclass linear SVM written by hand**, trained with mini-batch SGD |
| Glue | `pipeline.py` | Runs the stages in order and draws the results |

OpenCV is used only for reading and writing images, resizing, and
`findContours`.

## Run it

From the repo root:

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -r Phase1/digit_recognizer/requirements.txt

python Phase1/digit_recognizer/train.py      # pick settings on validation, save the model
python Phase1/digit_recognizer/evaluate.py   # score it ONCE on test
python Phase1/digit_recognizer/predict.py Phase1/test_data_p1   # try it on any images
```

Everything it writes goes to `results/digit_recognizer/`, which is
gitignored: the model (`digit_svm.npz`), `search.json`, `test_report.txt`
and the annotated test pages. In the annotated pages, green is our box and
label and blue is the ground-truth box. Mistakes are saved as
`WRONG_true<d>_pred<d>_*.png`.

## How the data is used

| Split | Used for |
|---|---|
| train (3,500) | Fitting the SVM. Crops come from the ground-truth boxes. |
| validation (500) | Choosing the HOG cell size and the SVM regularisation |
| test (1,000) | Measured once at the end, with nothing tuned on it |

Validation and test are scored **end to end**: the classifier only ever sees
the crop from the detector's own box. `evaluate.py` also reports accuracy
on ground-truth crops. If those two numbers differ, the detector is the part
to fix; if both are low, the classifier is.

## The pieces, in plain terms

**Detection.** The page is white (255) and the digit is darker. Every pixel
below the threshold becomes "ink", `findContours` traces the ink blobs, and
the box is the smallest rectangle around all of them. Akash's
`object_detection_p1_am.py` keeps only the largest blob, which can cut off
a digit that breaks into pieces. The default threshold of 255 matches how
the dataset drew its boxes. For real photos use `threshold=None`, which
switches to Otsu's method and picks the threshold from the image's histogram.

**Normalising.** The crop is scaled so its longer side is 24 px, then centred
on a 32x32 black canvas with the ink set to 1. The aspect ratio is kept,
because squashing a `1` into a square would make it look like a `0`. After
this step a tiny 7 in the corner and a huge 7 in the middle look alike.

**HOG.** For every pixel we take the direction its edge points (0-180°) and
how strong that edge is. The canvas is split into cells (4x4 or 8x8 px), and
each cell builds a 9-bin histogram of those directions. Neighbouring cells
are grouped into blocks and normalised, so thick strokes and thin strokes
give similar numbers. The final feature is all of those histograms joined
into one vector: 1,764 numbers for 4 px cells, 324 for 8 px cells.

**SVM.** The model keeps one weight vector per digit. Each digit's score is a
dot product with the features, and the highest score wins. Training pushes
the correct digit's score to beat every other digit's score by at least 1
(the hinge loss). Regularisation keeps the weights small so the model doesn't
memorise the training pages.

## Limits

This dataset is clean: one upright printed digit, black on white, with no
noise, shadows or tilt. A high score here proves the pipeline works on
this generator. It doesn't show that it will work on a webcam photo of a
handwritten number. Expect the first real photos to break it, and treat that
as the next thing to learn from.

## Next steps (roughly in order)

1. **Make the data harder** in `DataCreation/generate_digit_dataset.py`:
   rotation (±15°), blur, noise, uneven lighting and shadows, coloured paper,
   handwriting-style fonts, several digits per page. Retrain and see which
   change hurts most.
2. **Several digits per page.** Detection then has to return one box per
   digit, for example one per contour group, instead of one box for all
   the ink.
3. **A small CNN written from scratch** (conv, ReLU, pool, softmax with
   backprop in NumPy) on the same 32x32 crops. Compare it with HOG + SVM.
   This is the step that teaches how YOLO-style models work inside.
4. **Phase 2 (Gazebo).** Put the numbered sheets on four walls and rotate the
   camera. Feed each camera frame to `DigitRecognizer.recognise()` and log
   the digits in the order they appear.
5. **Phase 3 (Logitech webcam).** Wrap `recognise()` in a
   `cv2.VideoCapture(0)` loop. Expect to switch to Otsu or adaptive
   thresholding and to filter blobs by size and shape. A real room has many
   dark objects besides the paper.
6. **Distance.** With one camera, use a known paper size: distance ≈ focal
   length × real height ÷ height in pixels, with the focal length found
   once by calibration (`cv2.calibrateCamera` on a checkerboard). Two views
   (stereo, or the drone moving) remove the need to know the size. This
   is the triangulation from the CV class project.
