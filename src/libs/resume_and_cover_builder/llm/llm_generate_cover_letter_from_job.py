"""
This creates the cover letter (in html, utils will then convert in PDF) matching with job description and plain-text resume
"""
# app/libs/resume_and_cover_builder/llm_generate_cover_letter_from_job.py
import textwrap
from ..utils import LoggerChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from requests.exceptions import HTTPError as HTTPStatusError
from loguru import logger
from src.libs.resume_and_cover_builder.utils import current_month_year

today = current_month_year(tz="Europe/Berlin", lang="en")

# Load environment variables from .env file
load_dotenv()


class LLMCoverLetterJobDescription:
    def __init__(self, openai_api_key, strings):
        self.llm_cheap = LoggerChatModel(ChatOpenAI(model_name="gpt-4.1", openai_api_key=openai_api_key, temperature=0.2))
        self.llm_embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.strings = strings
        self.company_name = ""

    @staticmethod
    def _preprocess_template_string(template: str) -> str:
        """
        Preprocess the template string by removing leading whitespace and indentation.
        Args:
            template (str): The template string to preprocess.
        Returns:
            str: The preprocessed template string.
        """
        return textwrap.dedent(template)

    def set_resume(self, resume) -> None:
        """
        Set the resume text to be used for generating the cover letter.
        Args:
            resume (str): The plain text resume to be used.
        """
        self.resume = resume

    def set_job_description_from_text(self, job_description_text) -> None:
        """
        Set the job description text to be used for generating the cover letter.
        Args:
            job_description_text (str): The plain text job description to be used.
        """
        logger.debug("Starting job description summarization...")
        prompt = ChatPromptTemplate.from_template(self.strings.summarize_prompt_template)
        chain = prompt | self.llm_cheap | StrOutputParser()
        output = chain.invoke({"text": job_description_text})
        self.job_description = output
        logger.debug(f"Job description summarization complete: {self.job_description}")

    def set_company_name(self, company_name: str) -> None:
        """Set company name extracted from the job posting."""
        self.company_name = (company_name or "").strip()

    @staticmethod
    def _replace_company_placeholders(html: str, company_name: str) -> str:
        if not html:
            return html
        safe_company = (company_name or "").strip() or "Hiring Team"
        replacements = ["[Company Name]", "{company_name}", "Company Name"]
        for marker in replacements:
            html = html.replace(marker, safe_company)
        return html

    def generate_cover_letter(self) -> str:
        """
        Generate the cover letter based on the job description and resume.
        Returns:
            str: The generated cover letter
        """
        logger.debug("Starting cover letter generation...")
        prompt_template = self._preprocess_template_string(self.strings.cover_letter_template)
        logger.debug(f"Cover letter template after preprocessing: {prompt_template}")

        prompt = ChatPromptTemplate.from_template(prompt_template)
        logger.debug(f"Prompt created: {prompt}")

        chain = prompt | self.llm_cheap | StrOutputParser()
        logger.debug(f"Chain created: {chain}")

        input_data = {
            "job_description": self.job_description,
            "resume": self.resume,
            "company_name": self.company_name,
            "today": today,
        }
        logger.debug(f"Input data: {input_data}")

        output = chain.invoke(input_data)
        output = self._replace_company_placeholders(output, self.company_name)
        logger.debug(f"Cover letter generation result: {output}")

        logger.debug("Cover letter generation completed")
        return output
