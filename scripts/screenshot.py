"""Screenshots of a JudgeJitter report, for the README and launch posts.

    pip install playwright && playwright install chromium
    python scripts/screenshot.py sample/report.html docs

Writes report.png (top of the page), summary.png, flips.png and repeats.png.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from playwright.async_api import async_playwright


async def main(report: Path, out: Path, executable: str | None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path=executable) if executable else await p.chromium.launch()
        page = await browser.new_page(
            viewport={"width": 1000, "height": 1200}, color_scheme="light", device_scale_factor=2
        )
        await page.goto(report.resolve().as_uri())
        await page.screenshot(path=str(out / "report.png"), clip={"x": 0, "y": 0, "width": 1000, "height": 1180})
        for section in ("summary", "flips", "repeats"):
            await page.locator(f"#{section}").screenshot(path=str(out / f"{section}.png"))
        await browser.close()
    print(f"wrote 4 screenshots to {out}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: screenshot.py REPORT.html OUT_DIR [CHROME_PATH]")
    asyncio.run(main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else None))
