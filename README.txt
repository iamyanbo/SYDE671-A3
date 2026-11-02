SYDE 671 - Assignment 3
Yanbo Cheng

Open index.html to read the illustrated writeup. It works offline.
Keep index.html with its accompanying folders. Alternatively, open
A3_Writeup.html for a single-file copy with all figures and fonts embedded.
A3_Writeup.pdf is the PDF copy for submission.

RECREATE THE RESULTS
Tested with Python 3.13 and the versions in requirements.txt.
From this folder, run:

    python -m pip install -r requirements.txt
    python prepare_data.py
    python run_all.py > reproduction_log.txt
    python build_writeup.py

prepare_data.py is optional: the ZIP already includes the working PNG inputs.
It rebuilds them from the original JPEGs and creates the two window crops.
run_all.py runs A3.py without waiting for plot windows to close.
To view the plots interactively instead, run python A3.py and close each figure
to continue. Runs take several minutes with the straightforward pixel loops.
build_writeup.py rebuilds index.html, the RANSAC figures and results_summary.json.
No absolute input paths, external images, hosted fonts or network access are
needed after installing the Python packages.

RECREATE THE PDF (OPTIONAL)
The PDF is already included. To rebuild it after rebuilding index.html:

    python -m pip install -r requirements-pdf.txt
    python -m playwright install chromium
    python build_pdf.py

The browser download is needed once. PDF export then uses the local HTML,
figures, equation fonts and styles without network access.

The report shows one original/Harris/ANMS comparison and one set of six
descriptors. Features are still computed for all three parking photos.

CONTENTS
  A3.py                 All assignment functions and the configured main section
  harris.py             The supplied Harris corner detector
  prepare_data.py       Original JPEGs -> working PNGs and window crops
  run_all.py            Non-interactive runner
  build_writeup.py      Results -> HTML writeup and summary
  build_pdf.py          Local HTML -> PDF writeup
  requirements.txt      Tested dependency versions
  requirements-pdf.txt  Optional dependency for rebuilding the PDF
  index.html            Writeup with equations, code snippets and figures
  A3_Writeup.html        Standalone copy of the writeup
  A3_Writeup.pdf         PDF copy of the writeup
  theme.css             Stylesheet from the A2 reference page
  math/                 Offline equation styles, fonts, and pre-rendered equations
  data/                 Working inputs and the original source JPEGs
  homography_output/    Reproducible results, homographies and matching data
  results_summary.json  Counts, timings and alignment errors for this run

DATA AND COORDINATES
Part A: campus/left.png comes from 20261027_092432.jpg; campus/right.png
comes from 20261027_092426.jpg. Original size: 3024 x 4032. Working size:
600 x 800. The eight point pairs in A3.py use working-image (x, y) coordinates.

Part B: 1000006627.jpg -> parking/left.png; 1000006625.jpg -> middle.png;
1000006626.jpg -> right.png. Originals: 960 x 1280. Working size: 480 x 640.
Each outer image is mapped into the middle image's coordinate system.

Rectification crops use original-JPEG pixel boxes (left, top, right, bottom):
  20261027_092426.jpg: (1154, 1895, 1250, 2011)
  20261027_092432.jpg: (2041, 1885, 2157, 2087)
The four window corners in A3.py are local to each crop.
Rectangle corner order is top-left, top-right, bottom-left, bottom-right.
The chosen output aspect ratios are illustrative, not physical measurements.

SCOPE
The mosaic counts follow the provided clarification: one Part A mosaic with
at least two images, and one Part B mosaic with at least three images.
The two rectification examples are additional interpolation demonstrations.
Part A and Part B use different scenes, so their overview is not a controlled
comparison of manual versus automatic alignment on the same photographs.

REPRODUCIBILITY AND LIMITATIONS
RANSAC uses seed 671 for each pair, 1000 trials and a 3-pixel inlier threshold.
The descriptor squared-error ratio threshold is 0.5. Figure rendering and
measured times may vary across machines; compare the output pixels/matrices.
The methods use own homography fitting, inverse warping, interpolation,
ANMS, descriptors, ratio matching and RANSAC. OpenCV is used only to obtain
the Gaussian kernel in the A2 Gaussian-filter helper; high-level homography
and perspective-warp functions are not used.
Visible sky seams, close-car ghosting and black canvas boundaries are retained
in the report. No inpainting or generated scene content is included.

REFERENCES
Brown, Szeliski and Winder (2005), Multi-Image Matching using Multi-Scale
Oriented Patches:
https://www.microsoft.com/en-us/research/wp-content/uploads/2005/06/cvpr05.pdf
Visual writeup reference:
https://github.com/iamyanbo/SYDE671-A2
