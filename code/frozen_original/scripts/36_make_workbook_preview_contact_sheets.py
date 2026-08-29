from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(r"F:\CC\outputs\paper_results_v1\workbook_previews")
OUT = ROOT.parent
files = sorted(ROOT.glob("*.png"))

for page, start in enumerate(range(0, len(files), 10), start=1):
    selected = files[start:start + 10]
    canvas = Image.new("RGB", (1800, 2300), "white")
    draw = ImageDraw.Draw(canvas)
    for index, file in enumerate(selected):
        image = Image.open(file).convert("RGB")
        image.thumbnail((840, 390))
        col = index % 2
        row = index // 2
        x = 40 + col * 890
        y = 45 + row * 450
        draw.text((x, y), file.stem, fill="#111827")
        canvas.paste(image, (x, y + 25))
    canvas.save(OUT / f"workbook_preview_contact_{page}.png")

print(f"Created {page} contact sheets for {len(files)} worksheet previews.")
