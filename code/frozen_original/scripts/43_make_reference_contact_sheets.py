from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SRC = Path(r"F:\CC\tmp\pdfs\reference_low_text")
OUT = SRC / "contact_sheets"
OUT.mkdir(parents=True, exist_ok=True)

paths = sorted(SRC.glob("*.png"))
for page_no, start in enumerate(range(0, len(paths), 6), start=1):
    batch = paths[start : start + 6]
    thumb_w, thumb_h = 640, 830
    canvas = Image.new("RGB", (thumb_w * 3, thumb_h * 2), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, path in enumerate(batch):
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumb_w - 24, thumb_h - 50))
        x = (idx % 3) * thumb_w + (thumb_w - image.width) // 2
        y = (idx // 3) * thumb_h + 36
        canvas.paste(image, (x, y))
        draw.text(((idx % 3) * thumb_w + 12, (idx // 3) * thumb_h + 10), path.stem, fill="black")
    canvas.save(OUT / f"reference_low_text_contact_{page_no}.png", dpi=(150, 150))

