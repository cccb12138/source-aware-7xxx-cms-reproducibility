from pathlib import Path

from PIL import Image, ImageDraw


root = Path(r"F:\CC\outputs\supplementary_workbook_previews")
output = Path(r"F:\CC\outputs\supplementary_workbook_contact_sheets")
output.mkdir(parents=True, exist_ok=True)

files = sorted(root.glob("*.png"))
per_sheet = 5
thumb_width = 460
margin = 18
label_height = 30

for page, start in enumerate(range(0, len(files), per_sheet), start=1):
    group = files[start : start + per_sheet]
    thumbs = []
    for file in group:
        image = Image.open(file).convert("RGB")
        ratio = min(thumb_width / image.width, 300 / image.height)
        resized = image.resize((max(1, int(image.width * ratio)), max(1, int(image.height * ratio))))
        thumbs.append((file.stem, resized))

    height = margin + sum(label_height + image.height + margin for _, image in thumbs)
    canvas = Image.new("RGB", (thumb_width + 2 * margin, height), "white")
    draw = ImageDraw.Draw(canvas)
    y = margin
    for label, image in thumbs:
        draw.text((margin, y), label, fill="black")
        y += label_height
        canvas.paste(image, (margin, y))
        y += image.height + margin
    canvas.save(output / f"contact_{page}.png")

print(f"Created {len(list(output.glob('*.png')))} contact sheets for {len(files)} worksheet previews.")
