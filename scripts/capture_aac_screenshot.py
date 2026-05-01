from playwright.sync_api import sync_playwright


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1500, "height": 1000})
        page.on("dialog", lambda dialog: dialog.dismiss())
        page.goto("http://127.0.0.1:8000/aac/", wait_until="networkidle")
        page.select_option("#userSelect", "demo_user")
        page.click("#startBtn")
        page.wait_for_timeout(1200)
        page.fill("#partnerName", "Omer")
        page.fill("#partnerInput", "hii")
        page.screenshot(path="outputs/aac_debug_before_send.png", full_page=True)
        page.click("#sendPartnerBtn")
        page.wait_for_timeout(12000)
        card_count = page.locator("#normalCards .card").count()
        if card_count == 0:
            dialog_button = page.locator("text=OK").first
            if dialog_button.count() > 0:
                dialog_button.click()
            page.wait_for_timeout(2000)
            card_count = page.locator("#normalCards .card").count()
        if card_count == 0:
            body_preview = page.inner_text("body")[:1000]
            raise RuntimeError(f"No cards rendered. Body preview: {body_preview}")
        page.screenshot(path="outputs/aac_hi_clean_ui.png", full_page=True)
        browser.close()


if __name__ == "__main__":
    main()
