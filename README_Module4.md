# Module 4 - Human Boundary Detection

## Objective

This module performs human segmentation and boundary detection using classical computer-vision methods for RGB and thermal images, and compares the obtained masks with SAM2 masks.

## RGB Method

The RGB workflow uses an interactive GrabCut-based pipeline. The user selects a person, refines foreground/background regions, and generates a final mask, segmented human image, and object boundary.

## Thermal Method

The thermal workflow uses blur/contrast processing, Otsu thresholding, morphology, warm-region selection, connected components, and optional GrabCut refinement to isolate the human region and estimate its boundary.

## SAM2 Comparison

The classical masks are compared with SAM2 masks using IoU, Dice score, precision, and recall.

### RGB result

- IoU: 0.9939
- Dice Score: 0.9970
- Precision: 0.9975
- Recall: 0.9964

### Thermal result

- IoU: 0.7648
- Dice Score: 0.8667
- Precision: 0.9944
- Recall: 0.7681

## Website Demo

Open `module4.html`. The page contains the RGB human workflow, thermal human workflow, SAM2 comparison, and Fourier-domain theory section.

## Saved Output Images

The permanent website result gallery is stored in `module4_results/`. It contains the original RGB and thermal inputs, masks, segmented results, boundaries, and the classical-versus-SAM2 comparison screenshots.

## Files

- `module4.html` - web interface
- `module4.js` - RGB, thermal, and comparison implementation
- `module4.css` - Module 4 styles and saved-result gallery styles
- `module4_results/` - saved experiment outputs
