from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(r"F:\CC\outputs\submission_freeze_2026-07-25\workbook_previews")
OUT = ROOT.parent
files = sorted(ROOT.glob("*.png"))

for page, start in enumerate(range(0, len(files), 6), start=1):
    selected = files[start : start + 6]
    canvas = Image.new("RGB", (2100, 1800), "white")
    draw = ImageDraw.Draw(canvas)
    for index, file in enumerate(selected):
        image = Image.open(file).convert("RGB")
        image.thumbnail((990, 780))
        col = index % 2
        row = index // 2
        x = 40 + col * 1030
        y = 35 + row * 585
        draw.text((x, y), file.stem, fill="#111827")
        canvas.paste(image, (x, y + 25))
    canvas.save(OUT / f"submission_workbook_contact_{page}.png", dpi=(150, 150))

print(f"Created {page} contact sheets for {len(files)} previews.")

