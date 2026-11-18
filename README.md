# Image Processing Lab (IP_Lab)

Image processing experiments in Python. Each experiment is one self-contained script with a docstring (goal, inputs, outputs) and comments explaining the steps.

## Naming convention

- Scripts: `exp<num>_<topic>.py` (experiment 1 has sub-parts: `exp1_<part>_<topic>.py`)
- Images: prefixed with the experiment that uses them, e.g. `exp4_input.jpg`. Output images are named after the script that creates them, e.g. `exp1_1_thumbnail.png`.

## Experiments

| # | Script | Topic | Input image | Output |
|---|---|---|---|---|
| 1.1 | `exp1_1_square_thumbnail.py` | Center-crop to a square, resize to 250x250 | `exp1_input.jpg` | `exp1_1_thumbnail.png` |
| 1.2 | `exp1_2_channel_split_grayscale.py` | Split B/G/R channels, grayscale conversion | `exp1_input.jpg` | plot |
| 1.3 | `exp1_3_smart_mirror.py` | Flip horizontally or vertically based on orientation | `exp1_input.jpg` | plot |
| 1.4 | `exp1_4_proportional_resize.py` | Resize to 500px width keeping aspect ratio | `exp1_input.jpg` | `exp1_4_resized_500px.png` |
| 1.5 | `exp1_5_contrast_vignette.py` | Contrast boost + Gaussian vignette | `exp1_input.jpg` | plot |
| 2 | `exp2_noise_filtering.py` | Salt & pepper / Gaussian noise; mean, median, Gaussian filters | `exp2_input.jpg` | plot |
| 3 | `exp3_histogram_equalization_matching.py` | Histogram equalization and matching | `exp3_original.jpg`, `exp3_reference.jpg` | plot |
| 4 | `exp4_edge_detection.py` | Roberts, Prewitt, Sobel, Laplacian, Canny | `exp4_input.jpg` | plot |
| 5 | `exp5_morphological_operations.py` | Erosion, dilation, boundary, small-object removal, skeleton | none (drawn in code) | plot |
| 6 | `exp6_huffman_compression.py` | Huffman coding, compression ratio | `exp6_input.png` | printed table + stats |
| 7 | `exp7_image_segmentation.py` | Edge-based (Canny) vs region-based (Watershed) | `exp7_input.jpg` | plot |
| 8 | `exp8_object_detection.py` | YOLOv8 object detection | `exp8_input.jpg`, `yolov8n.pt` | plot + printed detections |

> **Note on exp 2:** the original `spiderman.jpg` was not available, so `exp2_input.jpg` is a small crop of the "Original" panel from the lab report PDF. Replace it with the real photo for better results (keep the filename).

## Setup

```
pip install opencv-python numpy matplotlib pillow scikit-image ultralytics
```

Run scripts from inside this folder (they use relative image paths):

```
cd IP_Lab
python exp4_edge_detection.py
```

## Snippets

Every experiment is available as a snippet, so you can paste the full script anywhere. There are two versions: **Espanso** (works in any app) and **VS Code** (works only in VS Code `.py` files). Both use the same names, listed below.

| Experiment | Names |
|---|---|
| 1.1 Square thumbnail | `exp1_1`, `thumbnail` |
| 1.2 Channel split & grayscale | `exp1_2`, `channels`, `grayscale` |
| 1.3 Smart mirror | `exp1_3`, `mirror`, `flip` |
| 1.4 Proportional resize | `exp1_4`, `resize` |
| 1.5 Contrast + vignette | `exp1_5`, `vignette`, `contrast` |
| 2 Noise filtering | `exp2`, `noise`, `filtering` |
| 3 Histogram | `exp3`, `histogram` |
| 4 Edge detection | `exp4`, `edge` |
| 5 Morphology | `exp5`, `morphology` |
| 6 Huffman compression | `exp6`, `huffman` |
| 7 Segmentation | `exp7`, `segmentation`, `watershed` |
| 8 Object detection | `exp8`, `yolo`, `detection` |

### Espanso (global, any app)

Type a colon followed by the name, e.g. `:histogram` or `:exp4`. The code is pasted via the clipboard, so editor auto-indent and auto-closing brackets don't mangle it.

| Experiment | Triggers |
|---|---|
| 1.1 Square thumbnail | `:exp1_1`, `:thumbnail` |
| 1.2 Channel split & grayscale | `:exp1_2`, `:channels`, `:grayscale` |
| 1.3 Smart mirror | `:exp1_3`, `:mirror`, `:flip` |
| 1.4 Proportional resize | `:exp1_4`, `:resize` |
| 1.5 Contrast + vignette | `:exp1_5`, `:vignette`, `:contrast` |
| 2 Noise filtering | `:exp2`, `:noise`, `:filtering` |
| 3 Histogram | `:exp3`, `:histogram` |
| 4 Edge detection | `:exp4`, `:edge` |
| 5 Morphology | `:exp5`, `:morphology` |
| 6 Huffman compression | `:exp6`, `:huffman` |
| 7 Segmentation | `:exp7`, `:segmentation`, `:watershed` |
| 8 Object detection | `:exp8`, `:yolo`, `:detection` |

- Definitions live in `%APPDATA%\espanso\match\ip_lab.yml`. A copy is kept in this repo at `snippets/ip_lab.yml`; to set up on another machine, copy it into that folder.
- Espanso picks up changes automatically; if not, run `espanso restart`.
- Espanso needs to be running (check the tray icon, or `espanso status`).
- The trigger fires as soon as you finish typing it, so `:exp1` alone does nothing; type a full trigger such as `:exp1_1`.

### VS Code (`.py` files only)

Type the name (no colon) and press **Enter** or **Tab**, e.g. `histogram` or `exp4`. Typing just `exp1` lists all five experiment 1 snippets.

- Definitions live in `%APPDATA%\Code\User\snippets\python.json`. A backup from before they were added is `python.json.pre-ip-lab.bak` in the same folder.
- If a name doesn't appear, press **Ctrl+Space**, or set `"editor.snippetSuggestions": "top"` in VS Code settings.

### Notes for both

- Snippets are copies of the scripts and do not update automatically if a script changes.
- They use the image filenames above, so they only run in a folder containing those images; change the filename for your own image.
