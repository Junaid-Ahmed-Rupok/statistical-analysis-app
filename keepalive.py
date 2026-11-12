"""
Keeps the Streamlit Community Cloud app awake.

Opens the app in a real headless browser (this counts as a genuine visit),
clicks the wake-up button if the app had gone to sleep, then confirms the
app actually rendered. Exits with code 1 if it did not, so GitHub marks the
run as failed and emails the repo owner.
"""
import os
import sys
import time

from playwright.sync_api import sync_playwright

URL = os.environ["APP_URL"]
# Text that only exists once the app itself has rendered (the hero heading).
MARKER = os.environ.get("APP_MARKER", "Turn Raw Data Into Powerful Insights")
WAKE_BUTTON = "Yes, get this app back up!"
MAX_WAIT_SECONDS = 180


def app_is_up(page) -> bool:
    # Streamlit Cloud renders the app inside an iframe, so check every frame.
    for frame in page.frames:
        try:
            if MARKER in frame.content():
                return True
        except Exception:
            continue
    return False


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(URL, wait_until="domcontentloaded", timeout=90_000)

        try:
            page.get_by_role("button", name=WAKE_BUTTON).click(timeout=20_000)
            print("App was asleep - clicked the wake-up button.")
        except Exception:
            print("No wake-up button found (app is probably already awake).")

        deadline = time.time() + MAX_WAIT_SECONDS
        while time.time() < deadline:
            if app_is_up(page):
                print("App is up and rendered.")
                browser.close()
                return 0
            page.wait_for_timeout(5_000)

        print(f"App did not render within {MAX_WAIT_SECONDS}s.")
        browser.close()
        return 1


if __name__ == "__main__":
    sys.exit(main())
