"""
This module contains the FacadeManager class, which is responsible for managing the interaction between the user and other components of the application.
"""
from datetime import date
import hashlib
import re
import time
import sys
import inquirer
from pathlib import Path
from urllib.parse import urlparse

import requests
from loguru import logger

from src.job import Job
from src.libs.resume_and_cover_builder.llm.llm_job_parser import LLMParser
from src.utils.chrome_utils import HTML_to_PDF
from .config import global_config


class ResumeFacade:
    def __init__(self, api_key, style_manager, resume_generator, resume_object, output_path):
        lib_directory = Path(__file__).resolve().parent
        global_config.STRINGS_MODULE_RESUME_PATH = lib_directory / "resume_prompt/strings_feder-cr.py"
        global_config.STRINGS_MODULE_RESUME_JOB_DESCRIPTION_PATH = lib_directory / "resume_job_description_prompt/strings_feder-cr.py"
        global_config.STRINGS_MODULE_COVER_LETTER_JOB_DESCRIPTION_PATH = lib_directory / "cover_letter_prompt/strings_feder-cr.py"
        global_config.STRINGS_MODULE_NAME = "strings_feder_cr"
        global_config.STYLES_DIRECTORY = lib_directory / "resume_style"
        global_config.LOG_OUTPUT_FILE_PATH = output_path
        global_config.API_KEY = api_key
        self.style_manager = style_manager
        self.resume_generator = resume_generator
        self.resume_generator.set_resume_object(resume_object)
        self.selected_style = None

    def set_driver(self, driver):
        self.driver = driver

    def _safe_quit_driver(self):
        try:
            if getattr(self, "driver", None):
                self.driver.quit()
        except Exception:
            # User may have closed the window manually during preview.
            pass

    def prompt_user(self, choices: list[str], message: str) -> str:
        questions = [
            inquirer.List("selection", message=message, choices=choices),
        ]
        return inquirer.prompt(questions)["selection"]

    def prompt_for_text(self, message: str) -> str:
        questions = [
            inquirer.Text("text", message=message),
        ]
        return inquirer.prompt(questions)["text"]

    @staticmethod
    def _flush_stdin_buffer() -> None:
        """
        Clear pending stdin bytes so the next input() truly waits for user action.
        This avoids inquirer/newline leftovers auto-skipping confirmation prompts.
        """
        if not sys.stdin or not sys.stdin.isatty():
            return
        try:
            import termios

            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
        except Exception:
            # Not all platforms/terminals support tcflush; best-effort only.
            return

    @classmethod
    def _wait_for_enter(cls, message: str, *, flush_buffer: bool = True) -> bool:
        if message:
            print(message)
        if flush_buffer:
            cls._flush_stdin_buffer()
        try:
            input()
            return True
        except EOFError:
            return False

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "")).strip()

    @staticmethod
    def _path_component(text: str, fallback: str) -> str:
        normalized = ResumeFacade._normalize_text(text)
        if not normalized:
            return fallback

        normalized = re.sub(r"[^\w\s.-]+", "", normalized, flags=re.UNICODE)
        normalized = re.sub(r"[\s_]+", "-", normalized)
        normalized = normalized.strip(".-")
        if not normalized:
            return fallback
        return normalized[:60].strip(".-") or fallback

    def _suggest_output_folder_name(self, fallback_source: str) -> str:
        today = date.today().isoformat()
        company = self._path_component(getattr(self.job, "company", ""), "")
        role = self._path_component(getattr(self.job, "role", ""), "")

        if company or role:
            return "_".join(part for part in (today, company, role) if part)

        source = fallback_source or getattr(self.job, "link", "") or today
        source_hash = hashlib.md5(source.encode()).hexdigest()[:10]
        if source.startswith("manual_chinese_jd_"):
            return f"{today}_manual-chinese-jd_{source_hash}"
        return f"{today}_job_{source_hash}"

    @staticmethod
    def _normalize_job_url(job_url: str) -> str:
        url = (job_url or "").strip().strip("\"").strip("'")
        if not url:
            raise ValueError("Job URL is required.")
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", url):
            url = f"https://{url.lstrip('/')}"
        return url

    @staticmethod
    def _is_boss_url(job_url: str) -> bool:
        if not job_url:
            return False
        try:
            host = (urlparse(job_url).netloc or "").lower()
        except Exception:
            host = str(job_url).lower()
        return "zhipin.com" in host

    @staticmethod
    def _is_linkedin_url(job_url: str) -> bool:
        if not job_url:
            return False
        try:
            host = (urlparse(job_url).netloc or "").lower()
        except Exception:
            host = str(job_url).lower()
        return "linkedin.com" in host

    @staticmethod
    def _fetch_public_job_html(job_url: str) -> str:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        try:
            response = requests.get(job_url, headers=headers, timeout=15)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Public job HTML fetch failed for {}: {}", job_url, exc)
            return ""

        html = response.text or ""
        if len(html) < 1000:
            logger.warning("Public job HTML fetch returned a short response for {}.", job_url)
            return ""
        return html

    @classmethod
    def _contains_blocked_page_markers(cls, text: str) -> bool:
        normalized = cls._normalize_text(text)
        if not normalized:
            return True

        lower_text = normalized.lower()
        blocked_markers = [
            "正在加载中",
            "请完成验证",
            "请先完成验证",
            "验证码",
            "访问受限",
            "登录后查看",
            "登录后可见",
            "请先登录",
            "security check",
            "verification required",
            "zusätzliche verifizierung erforderlich",
            "zusatzliche verifizierung erforderlich",
            "verification successful. waiting for",
            "waiting for",
            "captcha",
            "access denied",
            "sign in to continue",
            "sign in to view",
            "please enable javascript",
            "loading...",
            "loading…",
        ]
        return any(marker.lower() in lower_text for marker in blocked_markers)

    @classmethod
    def _looks_like_llm_placeholder(cls, text: str) -> bool:
        normalized = cls._normalize_text(text).lower()
        if not normalized:
            return True

        placeholder_markers = [
            "no job description information has been provided",
            "please provide the job description",
            "please provide the full job description",
            "so i can conduct a thorough analysis",
            "outline the key skills and requirements",
            "the provided content does not include a complete job description",
            "未提供职位描述",
            "请提供职位描述",
            "无法根据当前信息提取职位描述",
        ]
        return any(marker in normalized for marker in placeholder_markers)

    @classmethod
    def _is_job_description_usable(cls, job_description: str) -> bool:
        text = cls._normalize_text(job_description)
        if not text:
            return False

        if cls._contains_blocked_page_markers(text):
            return False
        if cls._looks_like_llm_placeholder(text):
            return False

        # Too short usually means page wasn't loaded or anti-bot wall was scraped.
        if len(text) < 50:
            return False

        # Structured bullet outputs are valid JD extracts too.
        lines = [ln.strip() for ln in (job_description or "").splitlines() if ln.strip()]
        bullet_like_pattern = re.compile(r"^(?:[•\-\–\*]|[0-9]+[\.\)]|[①②③④⑤⑥⑦⑧⑨⑩]|[一二三四五六七八九十]+[、\.])")
        bullet_count = sum(1 for ln in lines if bullet_like_pattern.match(ln))
        if bullet_count >= 3 and len(text) >= 80:
            return True

        lower_text = text.lower()
        zh_signals = ["岗位", "职责", "任职", "要求", "经验", "技能", "职位", "工作内容", "你将负责", "负责"]
        en_signals = ["responsibil", "requirement", "qualification", "experience", "skills", "about the role"]
        de_signals = [
            "aufgaben", "verantwort", "anforderung", "qualifikation",
            "kenntnisse", "erfahrung", "fähigkeiten", "voraussetzung",
            "stellenbeschreibung", "was wir bieten",
        ]
        if (
            not any(signal in text for signal in zh_signals)
            and not any(signal in lower_text for signal in en_signals)
            and not any(signal in lower_text for signal in de_signals)
        ):
            # Generic structural fallback for valid extracted JD content
            # (for cases where bullets/paragraphs do not include our language keyword sets).
            if len(text) < 160 or len(lines) < 4:
                return False
            if not any(token in text for token in ("•", ":", "：", ";", "；")):
                return False

        return True

    def _wait_for_readable_page(self, timeout_seconds: int = 20) -> None:
        deadline = time.time() + max(1, timeout_seconds)
        current_url = ""
        try:
            current_url = self.driver.current_url
        except Exception:
            current_url = ""

        boss_markers = ("职位描述", "岗位职责", "任职要求", "工作内容", "职位要求", "福利待遇")
        while time.time() < deadline:
            try:
                page_text = self.driver.find_element("tag name", "body").text
            except Exception:
                time.sleep(0.8)
                continue

            normalized_page_text = self._normalize_text(page_text)
            if len(normalized_page_text) >= 120 and not self._contains_blocked_page_markers(normalized_page_text):
                if self._is_boss_url(current_url):
                    if any(marker in normalized_page_text for marker in boss_markers):
                        return
                else:
                    return
            time.sleep(1.0)

    def _prompt_boss_manual_login(self, job_url: str) -> None:
        if not self._is_boss_url(job_url):
            return

        print(
            "\n[BOSS Login Step]\n"
            "请先在浏览器中完成 BOSS 登录/验证。\n"
            "完成后回到终端按 Enter；如果提示仍在登录页，请继续登录后再按 Enter。"
        )

        while True:
            confirmed = self._wait_for_enter("按 Enter 继续...")
            if not confirmed:
                logger.warning("No interactive stdin detected; continuing without manual confirmation for BOSS login step.")
                return

            try:
                current_url = (self.driver.current_url or "").lower()
            except Exception:
                current_url = ""

            if "job_detail" in current_url:
                return
            if "login" in current_url or "verify" in current_url:
                print("检测到你还在登录/验证页面，请先完成后再按 Enter。")
                continue

            # Some BOSS flows bounce to homepage after login; reopen the target URL once.
            try:
                logger.info("Reopening target BOSS job URL after manual login step.")
                self.driver.get(job_url)
                time.sleep(1.2)
                reopened_url = (self.driver.current_url or "").lower()
                if "job_detail" in reopened_url:
                    return
                if "login" in reopened_url or "verify" in reopened_url:
                    print("跳回登录/验证页了，请完成验证后再按 Enter。")
                    continue
            except Exception:
                logger.warning("Could not verify/reopen current BOSS URL after manual login step.")
            return

    @staticmethod
    def _collect_manual_job_description() -> str:
        print(
            "\n[Fallback] 自动抓取仍失败。你可以直接粘贴 JD 文本。\n"
            "粘贴完成后，输入单独一行 END 结束。\n"
            "如果不想手动粘贴，直接按 Enter 跳过。"
        )

        lines: list[str] = []
        while True:
            try:
                line = input()
            except EOFError:
                break

            if not lines and not line.strip():
                return ""
            if line.strip().upper() == "END":
                break
            lines.append(line)

        return "\n".join(lines).strip()

    def _extract_job_fields_from_html(self, body_html: str, job_url: str) -> None:
        self.llm_job_parser = LLMParser(openai_api_key=global_config.API_KEY)
        self.llm_job_parser.set_body_html(body_html)

        self.job = Job()
        self.job.role = self.llm_job_parser.extract_role()
        self.job.company = self.llm_job_parser.extract_company_name()
        self.job.description = self.llm_job_parser.extract_job_description()
        self.job.location = self.llm_job_parser.extract_location()
        self.job.link = job_url

    def _extract_job_fields_from_current_page(self, job_url: str) -> None:
        body_element = self.driver.find_element("tag name", "body")
        body_html = body_element.get_attribute("outerHTML")
        self._extract_job_fields_from_html(body_html, job_url)

    def _try_public_linkedin_extraction(self, job_url: str) -> bool:
        if not self._is_linkedin_url(job_url):
            return False

        logger.info("Trying LinkedIn public HTML extraction before asking for login.")
        body_html = self._fetch_public_job_html(job_url)
        if not body_html:
            return False

        self._extract_job_fields_from_html(body_html, job_url)
        if self._is_job_description_usable(self.job.description):
            logger.info("LinkedIn public HTML extraction succeeded.")
            return True

        logger.warning(
            "LinkedIn public HTML extraction did not produce a usable JD. Preview: {}",
            self._normalize_text(getattr(self.job, "description", ""))[:240],
        )
        return False

    def link_to_job(self, job_url):
        normalized_job_url = self._normalize_job_url(job_url)
        try:
            self.driver.get(normalized_job_url)
        except Exception as e:
            logger.exception(f"Failed to open job URL: {normalized_job_url} ({e})")
            raise RuntimeError(
                "无法打开职位链接。请确认链接完整有效，且以 https:// 开头后重试。"
            ) from e

        self.driver.implicitly_wait(10)
        if self._try_public_linkedin_extraction(normalized_job_url):
            return

        self._prompt_boss_manual_login(normalized_job_url)
        initial_wait = 40 if self._is_boss_url(normalized_job_url) else 20
        self._wait_for_readable_page(timeout_seconds=initial_wait)
        logger.info(f"Extracting job details from URL: {normalized_job_url}")
        self._extract_job_fields_from_current_page(normalized_job_url)

        if self._is_job_description_usable(self.job.description):
            return

        logger.warning(
            "Extracted JD still looks unusable after first pass. Preview: {}",
            self._normalize_text(self.job.description)[:240],
        )
        logger.warning(
            "Job description looks incomplete (likely anti-bot page, loading screen, or login wall). "
            "Please complete verification/login in browser, then press Enter to retry extraction."
        )
        confirmed = self._wait_for_enter(
            "\n[Notice] Unable to read full JD content from current page.\n"
            "Please complete login/verification in the opened browser window,\n"
            "then return here and press Enter to retry extraction.\n"
            "按 Enter 重试..."
        )
        if not confirmed:
            logger.warning("No interactive stdin detected; retrying extraction once without manual confirmation.")

        retry_wait = 45 if self._is_boss_url(normalized_job_url) else 25
        self._wait_for_readable_page(timeout_seconds=retry_wait)
        self._extract_job_fields_from_current_page(normalized_job_url)

        if not self._is_job_description_usable(self.job.description):
            manual_jd = self._collect_manual_job_description()
            if manual_jd:
                self.job.description = manual_jd
                logger.info("Using manually provided job description fallback.")
                return

            raise RuntimeError(
                "Could not extract complete job description from this URL. "
                "The page may require login/captcha or block automated access. "
                "You can rerun and paste JD text when prompted."
            )

    def create_resume_pdf_job_tailored(self, enable_manual_review: bool = True) -> tuple[bytes, str]:
        style_path = self.style_manager.get_style_path()
        if style_path is None:
            raise ValueError("You must choose a style before generating the PDF.")
        if not self._is_job_description_usable(getattr(self.job, "description", "")):
            raise RuntimeError(
                "Job description is still incomplete/invalid. "
                "Please finish login/verification on the job page and retry."
            )

        html_resume = self.resume_generator.create_resume_job_description_text(style_path, self.job.description)
        suggested_name = self._suggest_output_folder_name(self.job.link)

        result = HTML_to_PDF(html_resume, self.driver, enable_manual_review=enable_manual_review)
        self._safe_quit_driver()
        return result, suggested_name

    def create_resume_pdf(self, enable_manual_review: bool = True) -> tuple[bytes, str]:
        style_path = self.style_manager.get_style_path()
        if style_path is None:
            raise ValueError("You must choose a style before generating the PDF.")

        html_resume = self.resume_generator.create_resume(style_path)
        result = HTML_to_PDF(html_resume, self.driver, enable_manual_review=enable_manual_review)
        self._safe_quit_driver()
        return result

    def create_cover_letter(self, enable_manual_review: bool = True) -> tuple[bytes, str]:
        style_path = self.style_manager.get_style_path()
        if style_path is None:
            raise ValueError("You must choose a style before generating the PDF.")
        if not self._is_job_description_usable(getattr(self.job, "description", "")):
            raise RuntimeError(
                "Job description is still incomplete/invalid. "
                "Please finish login/verification on the job page and retry."
            )

        cover_letter_html = self.resume_generator.create_cover_letter_job_description(
            style_path,
            self.job.description,
            self.job.company,
        )
        suggested_name = self._suggest_output_folder_name(self.job.link)

        result = HTML_to_PDF(cover_letter_html, self.driver, enable_manual_review=enable_manual_review)
        self._safe_quit_driver()
        return result, suggested_name
