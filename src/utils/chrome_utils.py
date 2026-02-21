import time
import urllib

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager

from src.logging import logger


def chrome_browser_options():
    logger.debug("Setting Chrome browser options")
    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-gpu")
    options.add_argument("window-size=1200x800")
    options.add_argument("--disable-background-timer-throttling")
    options.add_argument("--disable-backgrounding-occluded-windows")
    options.add_argument("--disable-translate")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--no-first-run")
    options.add_argument("--no-default-browser-check")
    options.add_argument("--disable-logging")
    options.add_argument("--disable-autofill")
    options.add_argument("--disable-plugins")
    options.add_argument("--disable-animations")
    options.add_argument("--disable-cache")
    options.add_argument("--incognito")
    options.add_argument("--allow-file-access-from-files")
    options.add_argument("--disable-web-security")
    logger.debug("Using Chrome in incognito mode")
    return options


def init_browser() -> webdriver.Chrome:
    try:
        options = chrome_browser_options()
        driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
        logger.debug("Chrome browser initialized successfully.")
        return driver
    except Exception as e:
        logger.error(f"Failed to initialize browser: {str(e)}")
        raise RuntimeError(f"Failed to initialize browser: {str(e)}")


def _to_data_url(html_content: str) -> str:
    encoded_html = urllib.parse.quote(html_content)
    return f"data:text/html;charset=utf-8,{encoded_html}"


def _enable_manual_edit_mode(driver: webdriver.Chrome) -> None:
    driver.execute_script(
        """
        document.designMode = 'on';
        document.body.setAttribute('contenteditable', 'true');
        """
    )


def HTML_to_PDF(html_content: str, driver: webdriver.Chrome, enable_manual_review: bool = True) -> str:
    """
    Convert HTML to PDF (base64). If enable_manual_review is True, user can edit in browser
    and confirm in terminal before final export.
    """
    if not isinstance(html_content, str) or not html_content.strip():
        raise ValueError("HTML content must be a non-empty string.")

    try:
        driver.get(_to_data_url(html_content))
        time.sleep(1.5)

        if enable_manual_review:
            _enable_manual_edit_mode(driver)
            logger.info("Preview mode enabled. You can edit directly in the browser window.")
            print("\n[Preview] Edit the resume in Chrome.\nWhen done, return here and press Enter to export PDF.")
            try:
                input()
            except EOFError:
                logger.warning("No interactive stdin detected; exporting current preview as-is.")

            try:
                edited_html = driver.execute_script("return document.documentElement.outerHTML;")
                if isinstance(edited_html, str) and edited_html.strip():
                    # Reload clean HTML snapshot so editing handles/styles don't leak into final PDF
                    driver.get(_to_data_url(edited_html))
                    time.sleep(0.6)
            except Exception as e:
                raise RuntimeError(
                    "Could not capture edited preview. Keep the browser window open until export finishes."
                ) from e

        pdf_base64 = driver.execute_cdp_cmd(
            "Page.printToPDF",
            {
                "printBackground": True,
                "landscape": False,
                "paperWidth": 8.27,
                "paperHeight": 11.69,
                "marginTop": 0.8,
                "marginBottom": 0.8,
                "marginLeft": 0.5,
                "marginRight": 0.5,
                "displayHeaderFooter": False,
                "preferCSSPageSize": True,
                "generateDocumentOutline": False,
                "generateTaggedPDF": False,
                "transferMode": "ReturnAsBase64",
            },
        )
        return pdf_base64["data"]
    except Exception as e:
        logger.error(f"WebDriver exception: {e}")
        raise RuntimeError(f"WebDriver exception: {e}")
