from pathlib import Path
import html
import ast
import textwrap
import json
import os

os.environ.setdefault("MPLBACKEND", "Agg")
import imageio.v3 as iio
import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from harris import get_harris_corners

root = Path(__file__).resolve().parent
out = root / "homography_output"


def figure(path, caption, show_caption=False):
    label = f'<figcaption>{caption}</figcaption>' if show_caption else ''
    return (f'<figure><a href="{path}"><img src="{path}" alt="{html.escape(caption)}"></a>'
            + label + '</figure>')


def matrix(value):
    return '<pre>' + html.escape(np.array2string(value, precision=6, suppress_small=True)) + '</pre>'


manual = np.load(out / "manual_data.npz")
p1, p2, H = manual["points1"], manual["points2"], manual["H"]
A, b = [], []
for (x, y), (u, v) in zip(p1, p2):
    A.extend([[x, y, 1, 0, 0, 0, -u*x, -u*y],
              [0, 0, 0, x, y, 1, -v*x, -v*y]])
    b.extend([u, v])
q = np.column_stack((p1, np.ones(len(p1)))) @ H.T
manual_errors = np.linalg.norm(q[:, :2] / q[:, 2:] - p2, axis=1)

summary = {"manual_pairs": len(p1), "manual_mean_error_px": float(manual_errors.mean()),
           "manual_median_error_px": float(np.median(manual_errors)), "features": [],
           "matching": [], "interpolation": []}
for name in ["left", "middle", "right"]:
    image = iio.imread(root / "data" / "parking" / f"{name}.png")
    _, coords = get_harris_corners(rgb2gray(image))
    summary["features"].append({"image": name, "harris": coords.shape[1], "anms": 500})

for index, name in enumerate(["left", "right"], start=1):
    data = np.load(out / f"matching_data_{index}.npz")
    points1, points2, inliers, transform = [data[key] for key in ["points1", "points2", "inliers", "H"]]
    projected = np.column_stack((points1[inliers], np.ones(len(inliers)))) @ transform.T
    errors = np.linalg.norm(projected[:, :2] / projected[:, 2:] - points2[inliers], axis=1)
    summary["matching"].append({"pair": f"{name} to middle", "ratio_matches": len(points1),
                                "ransac_inliers": len(inliers),
                                "median_refitted_error_px": float(np.median(errors))})

for filename, inliers_only in [("matching_overview.png", False), ("ransac_inliers.png", True)]:
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    for index, (name, ax) in enumerate(zip(["left", "right"], axes), start=1):
        data = np.load(out / f"matching_data_{index}.npz")
        points1, points2 = data["points1"], data["points2"]
        selected = data["inliers"] if inliers_only else np.arange(len(points1))
        first = iio.imread(root / "data" / "parking" / f"{name}.png")
        second = iio.imread(root / "data" / "parking" / "middle.png")
        ax.imshow(np.concatenate((first, second), axis=1))
        for a, b in zip(points1[selected], points2[selected]):
            ax.plot([a[0], b[0] + first.shape[1]], [a[1], b[1]], "-o",
                    markersize=2, linewidth=.5, alpha=.7)
        label = "inliers" if inliers_only else "matches"
        ax.set_title(f"{name.capitalize()} to middle: {len(selected)} {label}")
        ax.axis("off")
    fig.tight_layout(pad=.5)
    fig.savefig(out / filename, dpi=120, bbox_inches="tight", pad_inches=.03)
    plt.close(fig)

for name in ["campus_warp", "campus_right_window", "campus_left_window"]:
    times = np.loadtxt(out / f"{name}_times.txt")
    summary["interpolation"].append({"example": name, "nearest_seconds": float(times[0]),
                                      "bilinear_seconds": float(times[1])})
(root / "results_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


equations = json.loads((root / "math" / "equations.json").read_text(encoding="utf-8"))


def eq(name):
    return '<div class="equation">' + equations[name] + '</div>'


source = (root / "A3.py").read_text(encoding="utf-8")
functions = {node.name: ast.get_source_segment(source, node)
             for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}


def code(text):
    return '<p class="section-label">Code snippet</p><pre><code>' + html.escape(textwrap.dedent(text).strip()) + '</code></pre>'


solve_code = functions['computeH']
solve_code = solve_code.replace(', show_equations=False', '')
start = solve_code.index('    if show_equations:')
end = solve_code.index('    h = np.linalg.lstsq', start)
solve_code = solve_code[:start] + solve_code[end:]
bilinear_code = functions['warpImageBilinear']
bilinear_sample = bilinear_code[bilinear_code.index('                # Four neighbors'):]
bilinear_sample = bilinear_sample[:bilinear_sample.rindex('    return warped')]
mosaic_code = functions['createMosaic']
blend_sample = mosaic_code[mosaic_code.index('    # Add both images'):]
warp_calls = "imwarped_nn = warpImageNearestNeighbor(im, H)\nimwarped_bil = warpImageBilinear(im, H)"


def excerpt(name, start, end=None):
    text = functions[name]
    begin = text.index(start)
    finish = text.index(end, begin) if end else len(text)
    return text[begin:finish].rstrip()


match_errors = excerpt('matchDescriptors', '        errors = []', '        order =')
match_order = excerpt('matchDescriptors', '        order =', '        if second_error == 0:')
match_ratio = excerpt('matchDescriptors', '        ratio =', '    return np.array')
ransac_sample = excerpt('ransacHomography', '        sample =', '        if H is None:')
ransac_test = excerpt('ransacHomography', '        inliers = []', '        if len(inliers) >')
ransac_best = excerpt('ransacHomography', '        if len(inliers) >', '    # Refit')
ransac_refit = excerpt('ransacHomography', '    H = computeH(points1[best_inliers]')


main = next(node for node in ast.parse(source).body if isinstance(node, ast.If))
cases = ast.literal_eval(next(node.value for node in main.body if isinstance(node, ast.Assign)
                            and any(isinstance(t, ast.Name) and t.id == 'rectifications' for t in node.targets)))
_, corners, width, height = cases[0]
rectification_code = (
    "# Top-left, top-right, bottom-left, bottom-right\n"
    f"corners = np.array([{corners[0]!r}, {corners[1]!r},\n"
    f"                    {corners[2]!r}, {corners[3]!r}])\n"
    f"rectangle = np.array([[0, 0], [{width - 1}, 0],\n"
    f"                      [0, {height - 1}], [{width - 1}, {height - 1}]])\n"
    "H = computeH(corners, rectangle)\n"
    "imwarped_nn = warpImageNearestNeighbor(im, H)\n"
    "imwarped_bil = warpImageBilinear(im, H)"
)
matching_code = '\n\n'.join(textwrap.dedent(part).strip() for part in [match_errors, match_order, match_ratio])


point_rows = ''.join(f'<tr><td>{i}</td><td>({x:g}, {y:g})</td><td>({u:g}, {v:g})</td></tr>'
                     for i, ((x, y), (u, v)) in enumerate(zip(p1, p2), start=1))
feature_rows = ''.join(f'<tr><td>{r["image"].capitalize()}</td><td>{r["harris"]:,}</td><td>{r["anms"]}</td></tr>' for r in summary['features'])
match_rows = ''.join(f'<tr><td>{r["pair"]}</td><td>{r["ratio_matches"]}</td><td>{r["ransac_inliers"]}</td></tr>' for r in summary['matching'])
time_rows = ''.join(f'<tr><td>{r["example"].replace("_", " ")}</td><td>{r["nearest_seconds"]:.3f} s</td><td>{r["bilinear_seconds"]:.3f} s</td></tr>' for r in summary['interpolation'])

page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Yanbo Cheng - SYDE 671 Assignment 3.">
<title>Assignment 3 | SYDE 671</title>
<link rel="stylesheet" href="math/katex.min.css"><link rel="stylesheet" href="theme.css">
<style>.gallery.sources img{{height:300px;aspect-ratio:auto;object-fit:contain}}</style>
</head><body><main class="wrap">
<p class="eyebrow">SYDE 671 &middot; Assignment 3</p><h1>A3</h1>
<p class="meta">Yanbo Cheng &middot; 21287485</p>

<h2>PART ONE</h2>
<h3 id="photos">1.1 &middot; Source photographs</h3>
<p>Two overlapping campus photos are used for the manual mosaic. The three parking-lot photos are used for the automatic mosaic.</p>
<div class="inputs">{figure('data/campus/left.png','Campus: left',True)}{figure('data/campus/right.png','Campus: right',True)}</div>
<div class="gallery sources">{figure('data/parking/left.png','Parking lot: left',True)}{figure('data/parking/middle.png','Parking lot: middle',True)}{figure('data/parking/right.png','Parking lot: right',True)}</div>

<h3 id="homography">1.2 &middot; Recovering the homography</h3>
<p>H maps a point (x, y) in the left photo to (u, v) in the right photo. Setting its last entry to 1 leaves eight unknowns:</p>
{eq('divide')}
<p>Multiplying by the denominator and collecting the unknowns gives two linear equations per point pair:</p>
{eq('rearrange')}
<p>For eight unknowns, four pairs of (x, y) points are enough to solve for H. Eight pairs are used for better accuracy, giving 16 equations to solve with least squares.</p>
{code(solve_code)}
{figure('homography_output/part_A_manual_points.png','Matching points have the same number.')}
<details><summary>Selected point coordinates</summary><div class="table-wrap"><table><thead><tr><th>Pair</th><th>Left (x, y)</th><th>Right (u, v)</th></tr></thead><tbody>{point_rows}</tbody></table></div></details>
<p>The resulting H is:</p>
{matrix(H)}

<h3 id="warping">1.3 &middot; Image warping and rectification</h3>
<p>The four image corners are mapped through H to find the output bounds. For each output pixel, H inverse gives the source position. Nearest neighbour rounds that position to a pixel:</p>
{code(functions['warpImageNearestNeighbor'])}
<p>Bilinear interpolation averages the four surrounding pixels: first across the top and bottom rows, then between those two colors:</p>
{code(bilinear_sample)}
<details><summary>Full bilinear function</summary>{code(functions['warpImageBilinear'])}</details>
<p>The H from 1.2 is used for both warpers:</p>
{code(warp_calls)}
{figure('homography_output/campus_warp_comparison.png','Warping the left photo using H from 1.2.')}
<p>For rectification, the four window corners are matched to four corners of a rectangle using <code>computeH</code>. Warping with this H straightens the window.</p>
{code(rectification_code)}
{figure('homography_output/campus_right_window_rectification.png','Right-photo window: selected corners, nearest neighbour, and bilinear.')}
{figure('homography_output/campus_left_window_rectification.png','Left-photo window: selected corners, nearest neighbour, and bilinear.')}
<div class="table-wrap"><table><caption>Runtime in seconds (one run)</caption><thead><tr><th>Image</th><th>Nearest neighbour</th><th>Bilinear</th></tr></thead><tbody>{time_rows}</tbody></table></div>
<p>Nearest neighbour looks coarser because it rounds the position, causing jagged edges. Bilinear smooths the neighboring colors, but takes longer because it uses four pixels for each output color.</p>

<h3 id="manual-mosaic">1.4 &middot; Blending the images</h3>
<p>The left photo is warped with <code>warpImageBilinear</code> and placed on the same canvas as the right photo. A warped ones image counts where its pixels are present. Dividing the added pixel values by the count averages the overlap.</p>
{code(mosaic_code)}
{figure('homography_output/part_A_sources_and_mosaic.png','The two photos and their mosaic.')}


<h2>PART TWO</h2>
<h3 id="corners">2.1 &middot; Harris corners and ANMS</h3>
<p>The supplied Harris function is used to find the corners. For each corner, ANMS finds stronger corners using <code>scores[i] &lt; 0.9 * scores[j]</code>. The Euclidean distance to each of these corners is calculated, and the smallest distance is taken as the radius. The 500 corners with the largest radii are kept.</p>
{code(functions['ANMS'])}
<div class="table-wrap"><table><thead><tr><th>Image</th><th>Harris corners</th><th>After ANMS</th></tr></thead><tbody>{feature_rows}</tbody></table></div>
{figure('homography_output/corners_1.png','Original photo, Harris corners, and 500 ANMS corners.')}

<h3 id="descriptors">2.2 &middot; Feature descriptors</h3>
<p>Each corner uses a 40 &times; 40 patch, blurred with the Gaussian filter from A2. Taking every fifth row and column gives an 8 &times; 8 descriptor.</p>
<p>We subtract the mean and divide by the standard deviation:</p>
{code(functions["extractDescriptors"])}
{figure('homography_output/descriptors_1.png','Six normalized 8 by 8 descriptors.')}

<h3 id="matching">2.3 &middot; Feature matching</h3>
{code(functions['matchDescriptors'])}
<p>Descriptors are compared using the sum of squared differences (SSD). A smaller SSD means a closer match. The best match is kept only if its error is less than half the second-best error. If the two errors are similar, the match is rejected.</p>
{figure('homography_output/matching_overview.png','Candidate matches for both pairs.')}

<h3 id="ransac">2.4 &middot; RANSAC and automatic mosaicing</h3>
{code(functions['ransacHomography'])}
<p>Each trial randomly chooses four matched point pairs and uses <code>computeH</code> to calculate H. That H maps every source point to the second image. The Euclidean distance from the mapped point to its matching point is the error; matches below 3 pixels are counted as inliers. This is repeated 1000 times, and the group with the most inliers is kept. H is then calculated again using all the inlier pairs, instead of only the four sampled pairs.</p>
<div class="table-wrap"><table><thead><tr><th>Pair</th><th>Matches from 2.3</th><th>RANSAC inliers</th></tr></thead><tbody>{match_rows}</tbody></table></div>
{figure('homography_output/ransac_inliers.png','Point matches kept by RANSAC for both pairs.')}
<details><summary>Recovered homographies</summary><p>Left to middle:</p>{matrix(np.loadtxt(out/'automatic_H_1.txt'))}<p>Right to middle:</p>{matrix(np.loadtxt(out/'automatic_H_2.txt'))}</details>
<p>The middle photo stays fixed. The left and right photos are warped using their H matrices, then blended as in 1.4.</p>
{figure('homography_output/part_B_sources_and_mosaic.png','The three source photos and automatic mosaic.')}
{figure('homography_output/part_A_and_B.png','Manual and automatic mosaics.')}

</main></body></html>'''
(root / "index.html").write_text(page, encoding="utf-8")
(root / ".nojekyll").touch()

# Make a portable copy with the styles, fonts and figures embedded.
import base64
import mimetypes
import re


def data_url(path):
    kind = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    return "data:" + kind + ";base64," + base64.b64encode(path.read_bytes()).decode("ascii")


standalone = page
for stylesheet in ["theme.css", "math/katex.min.css"]:
    path = root / stylesheet
    css = path.read_text(encoding="utf-8")
    css = re.sub(r'url\(([^)]+)\)', lambda m: 'url(' + data_url(path.parent / m.group(1).strip('"\'')) + ')', css)
    standalone = standalone.replace(f'<link rel="stylesheet" href="{stylesheet}">', '<style>' + css + '</style>')
standalone = re.sub(r'<a href="[^\"]+\.(?:png|jpg|jpeg)">(<img[^>]+>)</a>', r'\1', standalone)


def embed(match):
    attr, value = match.group(1), match.group(2)
    if value.startswith(("http:", "https:", "data:", "#")):
        return match.group(0)
    path = root / value
    result = f'{attr}="{data_url(path)}"'
    if attr == "href":
        result += f' download="{path.name}"'
    return result


standalone = re.sub(r'(src|href)="([^\"]+)"', embed, standalone)
(root / "A3_Writeup.html").write_text(standalone, encoding="utf-8")
print("Built the A2-style writeup and its standalone copy.")
