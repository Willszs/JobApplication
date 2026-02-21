import base64
import os
from pathlib import Path
import traceback
from typing import Tuple, Dict
import inquirer
import yaml

from src.libs.resume_and_cover_builder import ResumeFacade, ResumeGenerator, StyleManager
from src.resume_schemas.resume import Resume
from src.logging import logger
from src.utils.chrome_utils import init_browser
from src.utils.constants import (
    PLAIN_TEXT_RESUME_YAML,
    SECRETS_YAML,
)
# from ai_hawk.bot_facade import AIHawkBotFacade
# from ai_hawk.job_manager import AIHawkJobManager
# from ai_hawk.llm.llm_manager import GPTAnswerer


class ConfigError(Exception):
    """
    Custom exception for configuration-related errors.
    
    Use this when something is wrong with configuration files or values,
    e.g., missing keys, invalid types, unreadable paths, or parse failures.
    
    """
    pass

# temporarily commented because no job seeking functions apply now.
class ConfigValidator:
    """Validates YAML and secrets files used by resume generation."""

    @staticmethod
    def load_yaml(yaml_path: Path) -> dict:
        """Load and parse a YAML file."""
        try:
            with open(yaml_path, "r") as stream:
                return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ConfigError(f"Error reading YAML file {yaml_path}: {exc}")
        except FileNotFoundError:
            raise ConfigError(f"YAML file not found: {yaml_path}")

    @staticmethod
    def validate_secrets(secrets_yaml_path: Path) -> str:
        """Validate the secrets YAML file and retrieve the LLM API key."""
        secrets = ConfigValidator.load_yaml(secrets_yaml_path)
        mandatory_secrets = ["llm_api_key"]

        for secret in mandatory_secrets:
            if secret not in secrets:
                raise ConfigError(f"Missing secret '{secret}' in {secrets_yaml_path}")

            if not secrets[secret]:
                raise ConfigError(f"Secret '{secret}' cannot be empty in {secrets_yaml_path}")

        return secrets["llm_api_key"]


class FileManager:
    """
    make sure the folder and files are existing and generate the output folder
    Handles file system operations and validations.
    
    """
    REQUIRED_FILES = [SECRETS_YAML, PLAIN_TEXT_RESUME_YAML]

    @staticmethod
    def validate_data_folder(app_data_folder: Path) -> Tuple[Path, Path, Path]:
        """Validate the existence of the data folder and required files."""
        if not app_data_folder.is_dir():
            raise FileNotFoundError(f"Data folder not found: {app_data_folder}")

        missing_files = [file for file in FileManager.REQUIRED_FILES if not (app_data_folder / file).exists()]
        if missing_files:
            raise FileNotFoundError(f"Missing files in data folder: {', '.join(missing_files)}")

        output_folder = app_data_folder / "output"
        output_folder.mkdir(exist_ok=True)

        return (
            app_data_folder / SECRETS_YAML,
            app_data_folder / PLAIN_TEXT_RESUME_YAML,
            output_folder,
        )

    @staticmethod
    def get_uploads(plain_text_resume_file: Path) -> Dict[str, Path]:
        """Convert resume file paths to a dictionary."""
        if not plain_text_resume_file.exists():
            raise FileNotFoundError(f"Plain text resume file not found: {plain_text_resume_file}")

        uploads = {"plainTextResume": plain_text_resume_file}

        return uploads


def resolve_data_folder() -> Path:
    """
    Resolve runtime data folder.

    Priority:
    1) JOBAI_USER_DATA_DIR env var
    2) ./user_data
    3) ./data_folder (backward compatibility)
    """
    env_path = os.getenv("JOBAI_USER_DATA_DIR")
    if env_path:
        return Path(env_path)

    user_data = Path("user_data")
    if user_data.exists():
        return user_data

    return Path("data_folder")


def create_cover_letter(parameters: dict, llm_api_key: str):
    """Generate a cover letter PDF tailored to a job description URL."""
    try:
        logger.info("Generating a cover letter based on provided parameters.")
        resume_facade = _build_resume_facade(parameters, llm_api_key)
        job_url = _prompt_job_url()
        resume_facade.link_to_job(job_url)
        result_base64, suggested_name = resume_facade.create_cover_letter()
        output_path = Path(parameters["outputFileDirectory"]) / suggested_name / "cover_letter_tailored.pdf"
        _write_pdf(output_path, result_base64)
    except Exception as e:
        logger.exception(f"An error occurred while creating the CV: {e}")
        raise


def create_resume_pdf_job_tailored(parameters: dict, llm_api_key: str):
    """Generate a resume PDF tailored to a job description URL."""
    try:
        logger.info("Generating a CV based on provided parameters.")
        resume_facade = _build_resume_facade(parameters, llm_api_key)
        job_url = _prompt_job_url()
        resume_facade.link_to_job(job_url)
        result_base64, suggested_name = resume_facade.create_resume_pdf_job_tailored()
        output_path = Path(parameters["outputFileDirectory"]) / suggested_name / "resume_tailored.pdf"
        _write_pdf(output_path, result_base64)
    except Exception as e:
        logger.exception(f"An error occurred while creating the CV: {e}")
        raise


def create_resume_pdf(parameters: dict, llm_api_key: str):
    """Generate a base resume PDF without job tailoring."""
    try:
        logger.info("Generating a CV based on provided parameters.")
        resume_facade = _build_resume_facade(parameters, llm_api_key)
        result_base64 = resume_facade.create_resume_pdf()
        output_path = Path(parameters["outputFileDirectory"]) / "resume_base.pdf"
        _write_pdf(output_path, result_base64)
    except Exception as e:
        logger.exception(f"An error occurred while creating the CV: {e}")
        raise


def _load_plain_text_resume(parameters: dict) -> str:
    with open(parameters["uploads"]["plainTextResume"], "r", encoding="utf-8") as file:
        return file.read()


def _select_style(style_manager: StyleManager) -> None:
    available_styles = style_manager.get_styles()
    if not available_styles:
        logger.warning("No styles available. Proceeding without style selection.")
        return

    choices = style_manager.format_choices(available_styles)
    questions = [
        inquirer.List(
            "style",
            message="Select a style for the resume:",
            choices=choices,
        )
    ]
    style_answer = inquirer.prompt(questions)
    if not style_answer or "style" not in style_answer:
        logger.warning("No style selected. Proceeding with default style.")
        return

    selected_choice = style_answer["style"]
    for style_name in available_styles.keys():
        if selected_choice.startswith(style_name):
            style_manager.set_selected_style(style_name)
            logger.info(f"Selected style: {style_name}")
            return

    logger.warning("Style selection did not match known styles.")


def _build_resume_facade(parameters: dict, llm_api_key: str) -> ResumeFacade:
    plain_text_resume = _load_plain_text_resume(parameters)
    style_manager = StyleManager()
    _select_style(style_manager)
    resume_generator = ResumeGenerator()
    resume_object = Resume(plain_text_resume)
    resume_generator.set_resume_object(resume_object)

    resume_facade = ResumeFacade(
        api_key=llm_api_key,
        style_manager=style_manager,
        resume_generator=resume_generator,
        resume_object=resume_object,
        output_path=Path(parameters["outputFileDirectory"]),
    )
    resume_facade.set_driver(init_browser())
    return resume_facade


def _prompt_job_url() -> str:
    questions = [inquirer.Text("job_url", message="Please enter the URL of the job description:")]
    answers = inquirer.prompt(questions)
    if not answers or not answers.get("job_url"):
        raise ValueError("Job URL is required.")
    return answers["job_url"]


def _write_pdf(output_path: Path, result_base64: str) -> None:
    try:
        pdf_data = base64.b64decode(result_base64)
    except base64.binascii.Error as e:
        logger.error("Error decoding Base64: %s", e)
        raise

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as file:
        file.write(pdf_data)
    logger.info(f"PDF saved at: {output_path}")

        
def handle_inquiries(selected_action: str, parameters: dict, llm_api_key: str):
    """
    Decide which function to call based on the selected user actions.

    :param selected_action: Action selected by the user.
    :param parameters: Configuration parameters dictionary.
    :param llm_api_key: API key for the language model.
    """
    try:
        if selected_action:
            if "Generate Resume" == selected_action:
                logger.info("Crafting a standout professional resume...")
                create_resume_pdf(parameters, llm_api_key)
                
            if "Generate Resume Tailored for Job Description" == selected_action:
                logger.info("Customizing your resume to enhance your job application...")
                create_resume_pdf_job_tailored(parameters, llm_api_key)
                
            if "Generate Tailored Cover Letter for Job Description" == selected_action:
                logger.info("Designing a personalized cover letter to enhance your job application...")
                create_cover_letter(parameters, llm_api_key)

        else:
            logger.warning("No actions selected. Nothing to execute.")
    except Exception as e:
        logger.exception(f"An error occurred while handling inquiries: {e}")
        raise

def prompt_user_action() -> str:
    """
    Use inquirer to ask the user which action they want to perform.

    :return: Selected action.
    """
    try:
        questions = [
            inquirer.List(
                'action',
                message="Select the action you want to perform:",
                choices=[
                    "Generate Resume",
                    "Generate Resume Tailored for Job Description",
                    "Generate Tailored Cover Letter for Job Description",
                ],
            ),
        ]
        answer = inquirer.prompt(questions)
        if answer is None:
            logger.warning("No answer provided. The user may have interrupted.")
            return ""
        return answer.get('action', "")
    except Exception as e:
        logger.error(f"An error occurred while prompting action: {e}")
        return ""


def main():
    """Main entry point for the AIHawk Job Application Bot."""
    try:
        # Define and validate the data folder
        data_folder = resolve_data_folder()
        secrets_file, plain_text_resume_file, output_folder = FileManager.validate_data_folder(data_folder)
        llm_api_key = ConfigValidator.validate_secrets(secrets_file)

        # Prepare parameters
        parameters = {
            "uploads": FileManager.get_uploads(plain_text_resume_file),
            "outputFileDirectory": output_folder,
        }

        # Interactive prompt for user to select actions
        selected_actions = prompt_user_action()

        # Handle selected actions and execute them
        handle_inquiries(selected_actions, parameters, llm_api_key)

    except ConfigError as ce:
        logger.error(f"Configuration error: {ce}")
        logger.error(
            "Refer to the configuration guide for troubleshooting: "
            "https://github.com/feder-cr/Auto_Jobs_Applier_AIHawk?tab=readme-ov-file#configuration"
        )
    except FileNotFoundError as fnf:
        logger.error(f"File not found: {fnf}")
        logger.error("Ensure all required files are present in the data folder.")
    except RuntimeError as re:
        logger.error(f"Runtime error: {re}")
        logger.debug(traceback.format_exc())
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
