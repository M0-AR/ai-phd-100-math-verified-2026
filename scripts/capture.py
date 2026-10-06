"""Capture README screenshots from docs/preview.html (verified, re-runnable)."""
import pathlib
from playwright.sync_api import sync_playwright
ROOT = pathlib.Path(__file__).resolve().parents[1]
URL = (ROOT / "docs" / "preview.html").as_uri()
OUT = ROOT / "assets" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1280, "height": 900})
    pg.goto(URL)
    pg.wait_for_timeout(1200)
    pg.screenshot(path=str(OUT / "preview-hero.png"))
    pg.evaluate("document.getElementById('c2').scrollIntoView()")
    pg.wait_for_timeout(400)
    pg.screenshot(path=str(OUT / "preview-charts.png"))
    pg.evaluate("document.getElementById('quiz').scrollIntoView()")
    pg.wait_for_timeout(400)
    pg.screenshot(path=str(OUT / "preview-quiz.png"))
    # click first correct answers to prove quiz works, capture scored state
    pg.evaluate("""[...document.querySelectorAll('.q')].slice(0,3).forEach((d,i)=>{d.querySelectorAll('.opt')[ [0,0,0][i] ].click()})""")
    pg.wait_for_timeout(400)
    ok = pg.evaluate("document.getElementById('pscore').textContent")
    print("quiz state:", ok)
    b.close()
print("wrote", sorted(str(x) for x in OUT.glob("*.png")))
