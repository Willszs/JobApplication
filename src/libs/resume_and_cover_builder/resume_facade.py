"""
This module contains the FacadeManager class, which is responsible for managing the interaction between the user and other components of the application.
"""
import hashlib
import inquirer
from pathlib import Path

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
    def _is_job_description_usable(job_description: str) -> bool:
        text = (job_description or "").strip()
        if not text:
            return False

        blocked_markers = [
            "正在加载中",
            "loading",
            "请完成验证",
            "验证码",
            "登录后",
            "访问受限",
            "missing or incomplete",
            "please provide the full job description",
        ]
        lower_text = text.lower()
        if any(marker in text for marker in blocked_markers if any("\u4e00" <= c <= "\u9fff" for c in marker)):
            return False
        if any(marker in lower_text for marker in blocked_markers if not any("\u4e00" <= c <= "\u9fff" for c in marker)):
            return False

        # Too short usually means page wasn't loaded or anti-bot wall was scraped.
        if len(text) < 50:
            return False

        return True

    def _extract_job_fields_from_current_page(self, job_url: str) -> None:
        body_element = self.driver.find_element("tag name", "body")
        body_html = body_element.get_attribute("outerHTML")
        self.llm_job_parser = LLMParser(openai_api_key=global_config.API_KEY)
        self.llm_job_parser.set_body_html(body_html)

        self.job = Job()
        self.job.role = self.llm_job_parser.extract_role()
        self.job.company = self.llm_job_parser.extract_company_name()
        self.job.description = self.llm_job_parser.extract_job_description()
        self.job.location = self.llm_job_parser.extract_location()
        self.job.link = job_url

    def link_to_job(self, job_url):
        self.driver.get(job_url)
        self.driver.implicitly_wait(10)
        logger.info(f"Extracting job details from URL: {job_url}")
        self._extract_job_fields_from_current_page(job_url)

        if self._is_job_description_usable(self.job.description):
            return

        logger.warning(
            "Job description looks incomplete (likely anti-bot page, loading screen, or login wall). "
            "Please complete verification/login in browser, then press Enter to retry extraction."
        )
        print(
            "\n[Notice] Unable to read full JD content from current page.\n"
            "Please complete login/verification in the opened browser window,\n"
            "then return here and press Enter to retry extraction."
        )
        try:
            input()
        except EOFError:
            logger.warning("No interactive stdin detected; retrying extraction once without manual confirmation.")

        self._extract_job_fields_from_current_page(job_url)

        if not self._is_job_description_usable(self.job.description):
            raise RuntimeError(
                "Could not extract complete job description from this URL. "
                "The page may require login/captcha or block automated access."
            )

    def create_resume_pdf_job_tailored(self) -> tuple[bytes, str]:
        style_path = self.style_manager.get_style_path()
        if style_path is None:
            raise ValueError("You must choose a style before generating the PDF.")

        html_resume = self.resume_generator.create_resume_job_description_text(style_path, self.job.description)
        suggested_name = hashlib.md5(self.job.link.encode()).hexdigest()[:10]

        result = HTML_to_PDF(html_resume, self.driver, enable_manual_review=True)
        self._safe_quit_driver()
        return result, suggested_name

    def create_resume_pdf(self) -> tuple[bytes, str]:
        style_path = self.style_manager.get_style_path()
        if style_path is None:
            raise ValueError("You must choose a style before generating the PDF.")

        html_resume = self.resume_generator.create_resume(style_path)
        result = HTML_to_PDF(html_resume, self.driver, enable_manual_review=True)
        self._safe_quit_driver()
        return result

    def create_cover_letter(self) -> tuple[bytes, str]:
        style_path = self.style_manager.get_style_path()
        if style_path is None:
            raise ValueError("You must choose a style before generating the PDF.")

        cover_letter_html = self.resume_generator.create_cover_letter_job_description(
            style_path,
            self.job.description,
            self.job.company,
        )
        suggested_name = hashlib.md5(self.job.link.encode()).hexdigest()[:10]

        result = HTML_to_PDF(cover_letter_html, self.driver, enable_manual_review=True)
        self._safe_quit_driver()
        return result, suggested_name
