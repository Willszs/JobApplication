import time
import urllib
import os
import sys
from pathlib import Path

from selenium.common.exceptions import SessionNotCreatedException, WebDriverException
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager

from src.logging import logger


def _resolve_chrome_profile_paths() -> tuple[str, str]:
    """
    Resolve persistent Chrome profile paths.
    Priority:
      1) JOBAI_CHROME_USER_DATA_DIR / JOBAI_CHROME_PROFILE_DIR env vars
      2) ./user_data/chrome_profile and "Default"
    """
    user_data_dir = os.getenv("JOBAI_CHROME_USER_DATA_DIR")
    if not user_data_dir:
        user_data_dir = str((Path.cwd() / "user_data" / "chrome_profile").resolve())

    profile_dir = os.getenv("JOBAI_CHROME_PROFILE_DIR", "Default")
    return user_data_dir, profile_dir


def is_headless_mode_enabled() -> bool:
    """Return True when JOBAI_HEADLESS asks Chrome to run without a visible window."""
    return os.getenv("JOBAI_HEADLESS", "").strip().lower() in {"1", "true", "yes", "on"}


def _chrome_profile_summary() -> str:
    user_data_dir, profile_dir = _resolve_chrome_profile_paths()
    return f"user-data-dir={user_data_dir}, profile-directory={profile_dir}"


def _chrome_startup_hint(error: Exception) -> str:
    message = str(error).lower()
    profile_summary = _chrome_profile_summary()

    if "user data directory is already in use" in message or "profile" in message and "use" in message:
        return (
            "Chrome 配置目录可能正在被其他 Chrome 进程占用。"
            f"请关闭使用该配置的 Chrome 窗口后重试，或设置 JOBAI_CHROME_USER_DATA_DIR 指向一个新目录。当前配置：{profile_summary}"
        )
    if "cannot find chrome binary" in message or "chrome not reachable" in message:
        return "找不到或无法启动 Google Chrome。请确认 Google Chrome 已安装，并能从 /Applications 打开。"
    if "chromedriver" in message or "driver" in message or "session not created" in message:
        return "ChromeDriver 与当前 Chrome 可能不匹配，或下载失败。请检查网络后重试；必要时更新 Chrome。"
    if "network" in message or "connection" in message or "temporary failure" in message:
        return "初始化 ChromeDriver 时可能需要联网下载驱动。请检查网络后重新运行。"

    return f"Chrome/Selenium 初始化失败。当前配置：{profile_summary}"


def chrome_browser_options(headless: bool = False):
    logger.debug("Setting Chrome browser options")
    options = Options()
    if not headless:
        options.add_argument("--start-maximized")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--ignore-certificate-errors")
    # Keep browser features closer to a normal user session to reduce anti-bot triggers.
    options.add_argument("--disable-gpu")
    options.add_argument("window-size=1200x800")
    options.add_argument("--disable-background-timer-throttling")
    options.add_argument("--disable-backgrounding-occluded-windows")
    options.add_argument("--disable-translate")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--no-first-run")
    options.add_argument("--no-default-browser-check")
    options.add_argument("--disable-logging")
    options.add_argument("--allow-file-access-from-files")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    if headless:
        options.add_argument("--headless=new")
        options.add_argument("window-size=1200x800")

    # Reuse a persistent browser profile so login/cookies survive across runs.
    user_data_dir, profile_dir = _resolve_chrome_profile_paths()
    Path(user_data_dir).mkdir(parents=True, exist_ok=True)
    options.add_argument(f"--user-data-dir={user_data_dir}")
    options.add_argument(f"--profile-directory={profile_dir}")

    logger.debug(f"Using Chrome user-data-dir: {user_data_dir}")
    logger.debug(f"Using Chrome profile-directory: {profile_dir}")
    logger.debug(f"Using Chrome in {'headless' if headless else 'standard'} mode (incognito disabled)")
    return options


def init_browser(headless: bool = False) -> webdriver.Chrome:
    try:
        options = chrome_browser_options(headless=headless)
        driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
        # Mask common Selenium fingerprints on every new document.
        driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {
                "source": """
                    Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
                    Object.defineProperty(navigator, 'languages', { get: () => ['zh-CN', 'zh', 'en-US', 'en'] });
                    Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4] });
                    window.chrome = window.chrome || { runtime: {} };
                """
            },
        )
        logger.debug(f"Chrome browser initialized successfully. headless={headless}")
        return driver
    except (SessionNotCreatedException, WebDriverException) as e:
        hint = _chrome_startup_hint(e)
        logger.error(f"Chrome/Selenium 启动失败：{e}")
        raise RuntimeError(hint) from e
    except Exception as e:
        hint = _chrome_startup_hint(e)
        logger.error(f"Chrome 初始化失败：{e}")
        raise RuntimeError(hint) from e


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


def _flush_stdin_buffer() -> None:
    if not sys.stdin or not sys.stdin.isatty():
        return
    try:
        import termios

        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except Exception:
        # Best-effort only.
        return


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
            _flush_stdin_buffer()
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
                    "无法读取你在 Chrome 里编辑后的预览内容。导出完成前请不要关闭预览窗口。"
                ) from e
        else:
            logger.info("Headless/export-only mode enabled. Skipping manual Chrome preview.")

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
    except WebDriverException as e:
        logger.error(f"Chrome PDF 导出失败：{e}")
        raise RuntimeError(
            "Chrome PDF 导出失败。请确认 Chrome 窗口没有被手动关闭；"
            "如果问题反复出现，可以关闭所有 Chrome 后重试。"
        ) from e
    except Exception as e:
        logger.error(f"PDF 导出失败：{e}")
        raise RuntimeError(f"PDF 导出失败：{e}") from e
