"""
This module is responsible for generating resumes and cover letters using the LLM model.
"""
# app/libs/resume_and_cover_builder/resume_generator.py
from string import Template
from typing import Any

from src.libs.resume_and_cover_builder.llm.llm_generate_resume import LLMResumer
from src.libs.resume_and_cover_builder.llm.llm_generate_resume_from_job import LLMResumeJobDescription
from src.libs.resume_and_cover_builder.llm.llm_generate_cover_letter_from_job import LLMCoverLetterJobDescription
from .module_loader import load_module
from .config import global_config


class ResumeGenerator:
    def __init__(self):
        self.resume_object = None

    def set_resume_object(self, resume_object):
        self.resume_object = resume_object

    def _create_resume(self, gpt_answerer: Any, style_path: str) -> str:
        # 让 LLM 知道简历对象
        gpt_answerer.set_resume(self.resume_object)

        # HTML 模板
        template = Template(global_config.html_template)

        # 读取 CSS
        try:
            with open(style_path, "r", encoding="utf-8") as f:
                style_css = f.read()
        except FileNotFoundError:
            raise ValueError(f"The style file was not found at path: {style_path}")
        except Exception as e:
            raise RuntimeError(f"Error while reading the CSS file: {e}")

        # 生成主体 HTML（内部会调用各 section，包括“≥80% 覆盖”的 Work Experience）
        body_html = gpt_answerer.generate_html_resume()
        if not isinstance(body_html, str) or not body_html.strip():
            raise ValueError("Generated resume HTML is empty.")

        return template.substitute(body=body_html, style_css=style_css)


    # -------------------------
    # Plain resume (no JD)
    # -------------------------
    def create_resume(self, style_path: str) -> str:
        strings = load_module(global_config.STRINGS_MODULE_RESUME_PATH, global_config.STRINGS_MODULE_NAME)
        gpt_answerer = LLMResumer(global_config.API_KEY, strings)
        return self._create_resume(gpt_answerer, style_path)

    # -------------------------
    # Resume tailored to JD (JD text provided)
    # -------------------------
    def create_resume_job_description_text(self, style_path: str, job_description_text: str) -> str:
        strings = load_module(
            global_config.STRINGS_MODULE_RESUME_JOB_DESCRIPTION_PATH,
            global_config.STRINGS_MODULE_NAME
        )

        # LLM 封装，并注入 JD
        gpt_answerer = LLMResumeJobDescription(global_config.API_KEY, strings)
        gpt_answerer.set_job_description_from_text(job_description_text)

        # 生成并返回 HTML
        return self._create_resume(gpt_answerer, style_path)

    # -------------------------
    # Cover letter tailored to JD
    # -------------------------
    def create_cover_letter_job_description(self, style_path: str, job_description_text: str, company_name: str = "") -> str:
        strings = load_module(
            global_config.STRINGS_MODULE_COVER_LETTER_JOB_DESCRIPTION_PATH,
            global_config.STRINGS_MODULE_NAME
        )
        gpt_answerer = LLMCoverLetterJobDescription(global_config.API_KEY, strings)
        gpt_answerer.set_resume(self.resume_object)
        gpt_answerer.set_job_description_from_text(job_description_text)
        gpt_answerer.set_company_name(company_name)

        cover_letter_html = gpt_answerer.generate_cover_letter()
        if not isinstance(cover_letter_html, str) or not cover_letter_html.strip():
            raise ValueError("Generated cover letter HTML is empty.")

        template = Template(global_config.html_template)
        with open(style_path, "r", encoding="utf-8") as f:
            style_css = f.read()

        return template.substitute(body=cover_letter_html, style_css=style_css)
