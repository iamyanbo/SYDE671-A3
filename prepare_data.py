from pathlib import Path
from PIL import Image, ImageOps

data = Path(__file__).resolve().parent / "data"

photos = [
    ("campus", "20261027_092432.jpg", "left", (600, 800)),
    ("campus", "20261027_092426.jpg", "right", (600, 800)),
    ("parking", "1000006627.jpg", "left", (480, 640)),
    ("parking", "1000006625.jpg", "middle", (480, 640)),
    ("parking", "1000006626.jpg", "right", (480, 640)),
]
for scene, filename, name, size in photos:
    image = ImageOps.exif_transpose(Image.open(data / scene / "originals" / filename))
    image = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    image.save(data / scene / f"{name}.png")

rectification = data / "rectification"
rectification.mkdir(exist_ok=True)
Image.open(data / "campus" / "originals" / "20261027_092426.jpg").crop(
    (1154, 1895, 1250, 2011)).save(rectification / "campus_right_window.png")
Image.open(data / "campus" / "originals" / "20261027_092432.jpg").crop(
    (2041, 1885, 2157, 2087)).save(rectification / "campus_left_window.png")
print("Prepared five working photos and two window crops.")
