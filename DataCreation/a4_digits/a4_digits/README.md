# Synthetic A4 digit dataset

5,000 single-channel grayscale PNG pages, 595 pixels wide by 842 high.
Each contains exactly one upright printed digit (0-9), black on white.
The page fills the image; there is no desk, paper border, rotation, noise,
shadow, handwriting, or perspective distortion. Edges are antialiased.

| Split | Images | Images per digit |
|---|---:|---:|
| train | 3,500 | 350 |
| validation | 500 | 50 |
| test | 1,000 | 100 |

Each split contains images/ and annotations.csv. image_path in the CSV is
relative to the dataset root, not the CSV's parent. Filenames do not encode
class labels. preview.png is an overview and is NOT a training example.
metadata.json records settings and generation versions.

## Annotation coordinates
Origin is the upper-left image corner. x increases right; y increases down.
Boxes tightly enclose all nonwhite pixels, including antialiased edge pixels.
x_min/y_min are inclusive; x_max/y_max are exclusive. Thus:
width = x_max - x_min, height = y_max - y_min.
Crop with page[y_min:y_max, x_min:x_max]. To draw an inclusive OpenCV
rectangle, use (x_min, y_min), (x_max-1, y_max-1).
The page has at least 16 white pixels of margin around every digit.
font_scale, font, stroke_thickness and target_height describe generation;
bbox_height records the actual ink height. pixel_sha256 hashes decoded
page pixels and is unique across all splits.

## Intended next steps
Locate digits using thresholding and contours; preserve aspect ratio when
resizing crops onto a padded fixed-size canvas; extract HOG features; train
an SVM using train only. Select preprocessing and model settings using
validation. Evaluate the final fixed pipeline once on test. Report digit
accuracy and confusion matrix, plus bounding-box intersection over union.
Never give the recognition model the true digit or ground-truth box at
end-to-end test time: it must use its predicted box. Ground-truth crops can
be used separately to isolate recognition performance.

Splits share the same font/size distributions and contain no identical pages.
That measures performance on this synthetic generator, not on handwriting,
photographs, unseen fonts, or real-world documents. Later add separate
challenge datasets for those conditions. More synthetic images alone do not
establish real-world performance.
