from __future__ import annotations

import base64
from datetime import date
import os
import hashlib
from pathlib import Path
import traceback
from typing import Tuple, Dict
import inquirer
import yaml

from src.logging import logger
from src.utils.constants import (
    PLAIN_TEXT_RESUME_YAML,
    SECRETS_YAML,
)



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
            raise ConfigError(f"YAML 格式错误：{yaml_path}。请检查缩进、冒号和引号是否正确。原始错误：{exc}")
        except FileNotFoundError:
            raise ConfigError(f"找不到配置文件：{yaml_path}")

    @staticmethod
    def validate_secrets(secrets_yaml_path: Path) -> str:
        """Validate the secrets YAML file and retrieve the LLM API key."""
        secrets = ConfigValidator.load_yaml(secrets_yaml_path)
        if not isinstance(secrets, dict):
            raise ConfigError(f"{secrets_yaml_path} 内容为空或格式不正确，请参考 data_folder_example/secrets.example.yaml。")

        mandatory_secrets = ["llm_api_key"]

        for secret in mandatory_secrets:
            if secret not in secrets:
                raise ConfigError(f"{secrets_yaml_path} 缺少必填项：{secret}")

            if not secrets[secret]:
                raise ConfigError(f"{secrets_yaml_path} 里的 {secret} 不能为空，请填入你的 LLM API key。")

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
            raise FileNotFoundError(
                f"找不到数据目录：{app_data_folder}。请创建该目录，或设置 JOBAI_USER_DATA_DIR 指向你的数据目录。"
            )

        missing_files = [file for file in FileManager.REQUIRED_FILES if not (app_data_folder / file).exists()]
        if missing_files:
            examples = ", ".join(f"data_folder_example/{file}" for file in missing_files)
            raise FileNotFoundError(
                f"{app_data_folder} 缺少必要文件：{', '.join(missing_files)}。"
                f"可以参考或复制示例文件：{examples}"
            )

        output_folder = app_data_folder / "output"
        output_folder.mkdir(exist_ok=True)
        photo_folder = app_data_folder / "photo"
        photo_folder.mkdir(exist_ok=True)

        return (
            app_data_folder / SECRETS_YAML,
            app_data_folder / PLAIN_TEXT_RESUME_YAML,
            output_folder,
        )

    @staticmethod
    def get_uploads(plain_text_resume_file: Path) -> Dict[str, Path]:
        """Convert resume file paths to a dictionary."""
        if not plain_text_resume_file.exists():
            raise FileNotFoundError(
                f"找不到简历数据文件：{plain_text_resume_file}。请参考 data_folder_example/plain_text_resume.example.yaml。"
            )

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
        job_url = _prompt_job_url()
        resume_facade = _build_resume_facade(parameters, llm_api_key)
        resume_facade.link_to_job(job_url)
        result_base64, suggested_name = resume_facade.create_cover_letter()
        output_path = Path(parameters["outputFileDirectory"]) / suggested_name / "cover_letter.pdf"
        _write_pdf(output_path, result_base64)
    except Exception as e:
        logger.exception(f"An error occurred while creating the CV: {e}")
        raise


def create_resume_pdf_job_tailored(parameters: dict, llm_api_key: str):
    """Generate a resume PDF tailored to a job description URL."""
    try:
        logger.info("Generating a CV based on provided parameters.")
        job_url = _prompt_job_url()
        resume_facade = _build_resume_facade(parameters, llm_api_key)
        resume_facade.link_to_job(job_url)
        result_base64, suggested_name = resume_facade.create_resume_pdf_job_tailored()
        output_path = Path(parameters["outputFileDirectory"]) / suggested_name / "resume.pdf"
        _write_pdf(output_path, result_base64)
    except Exception as e:
        logger.exception(f"An error occurred while creating the CV: {e}")
        raise


def _prompt_pasted_jd_text() -> str:
    print(
        "\n请粘贴中文JD全文（可多行）。\n"
        "粘贴完成后，输入单独一行 END 结束："
    )
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip().upper() == "END":
            break
        lines.append(line)

    jd_text = "\n".join(lines).strip()
    if not jd_text:
        raise ValueError("未收到JD内容，请重新运行后粘贴完整中文JD。")
    return jd_text


def create_resume_pdf_from_pasted_chinese_jd(parameters: dict, llm_api_key: str):
    """Generate a resume PDF tailored to pasted Chinese JD text."""
    try:
        from src.job import Job
        from src.utils.language import detect_jd_language

        logger.info("Generating a CV from pasted Chinese JD.")
        jd_text = _prompt_pasted_jd_text()
        if detect_jd_language(jd_text) != "zh":
            raise ValueError(
                "该选项仅支持中文JD。你粘贴的内容看起来不是中文JD，请改用“Generate Resume Tailored for Job Description”。"
            )

        resume_facade = _build_resume_facade(parameters, llm_api_key)
        jd_hash = hashlib.md5(jd_text.encode("utf-8")).hexdigest()[:12]
        resume_facade.job = Job(
            description=jd_text,
            link=f"manual_chinese_jd_{jd_hash}",
        )
        result_base64, suggested_name = resume_facade.create_resume_pdf_job_tailored()
        output_path = Path(parameters["outputFileDirectory"]) / suggested_name / "resume.pdf"
        _write_pdf(output_path, result_base64)
    except Exception as e:
        logger.exception(f"An error occurred while creating the CV from pasted Chinese JD: {e}")
        raise


def create_resume_pdf(parameters: dict, llm_api_key: str):
    """Generate a base resume PDF without job tailoring."""
    try:
        logger.info("Generating a CV based on provided parameters.")
        resume_facade = _build_resume_facade(parameters, llm_api_key)
        result_base64 = resume_facade.create_resume_pdf()
        output_path = Path(parameters["outputFileDirectory"]) / f"{date.today().isoformat()}_base-resume" / "resume.pdf"
        _write_pdf(output_path, result_base64)
    except Exception as e:
        logger.exception(f"An error occurred while creating the CV: {e}")
        raise


def _load_plain_text_resume(parameters: dict) -> str:
    with open(parameters["uploads"]["plainTextResume"], "r", encoding="utf-8") as file:
        return file.read()


def _set_default_style(style_manager: StyleManager) -> None:
    available_styles = style_manager.get_styles()
    if not available_styles:
        raise ValueError("No resume styles available. Please add at least one style file.")

    preferred_style = "Modern Grey"
    if preferred_style in available_styles:
        style_manager.set_selected_style(preferred_style)
        logger.info(f"Using default style: {preferred_style}")
        return

    selected_style = next(iter(available_styles.keys()))
    style_manager.set_selected_style(selected_style)
    logger.warning(
        f"Preferred style '{preferred_style}' not found. Falling back to available style: {selected_style}"
    )

def _build_resume_facade(parameters: dict, llm_api_key: str) -> ResumeFacade:
    from src.libs.resume_and_cover_builder import ResumeFacade, ResumeGenerator, StyleManager
    from src.resume_schemas.resume import Resume
    from src.utils.chrome_utils import init_browser

    plain_text_resume = _load_plain_text_resume(parameters)
    style_manager = StyleManager()
    _set_default_style(style_manager)
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
        logger.error(f"PDF 数据解析失败：{e}")
        raise

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as file:
        file.write(pdf_data)
    logger.info(f"PDF saved at: {output_path}")


ACTION_HANDLERS = {
    "Generate Resume": (
        "Crafting a standout professional resume...",
        create_resume_pdf,
    ),
    "Generate Resume Tailored for Job Description": (
        "Customizing your resume to enhance your job application...",
        create_resume_pdf_job_tailored,
    ),
    "Generate Tailored Cover Letter for Job Description": (
        "Designing a personalized cover letter to enhance your job application...",
        create_cover_letter,
    ),
    "Paste Chinese JD and Generate Tailored Resume": (
        "Generating tailored resume from pasted Chinese JD...",
        create_resume_pdf_from_pasted_chinese_jd,
    ),
}


def handle_inquiries(selected_action: str, parameters: dict, llm_api_key: str):
    """
    Decide which function to call based on the selected user actions.

    :param selected_action: Action selected by the user.
    :param parameters: Configuration parameters dictionary.
    :param llm_api_key: API key for the language model.
    """
    try:
        if not selected_action:
            logger.warning("No actions selected. Nothing to execute.")
            return

        handler_entry = ACTION_HANDLERS.get(selected_action)
        if not handler_entry:
            logger.warning(f"Unknown action selected: {selected_action}")
            return

        log_message, handler = handler_entry
        logger.info(log_message)
        handler(parameters, llm_api_key)
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
                choices=list(ACTION_HANDLERS.keys()),
            ),
        ]
        answer = inquirer.prompt(questions)
        if answer is None:
            logger.warning("No answer provided. The user may have interrupted.")
            return ""
        return answer.get('action', "")
    except Exception as e:
        logger.error(f"菜单显示失败：{e}")
        return ""


def log_user_friendly_error(error_type: str, message: str, *, hint: str | None = None) -> None:
    """Log a concise Chinese error with an optional next step."""
    logger.error(f"{error_type}：{message}")
    if hint:
        logger.error(f"处理建议：{hint}")


def runtime_error_hint(error: RuntimeError) -> str:
    """Return a helpful next step for common runtime failures."""
    message = str(error).lower()
    if "chrome" in message or "webdriver" in message or "selenium" in message:
        return (
            "请确认 Google Chrome 已安装；如果提示配置目录被占用，请关闭旧 Chrome 窗口，"
            "或设置 JOBAI_CHROME_USER_DATA_DIR 指向一个新目录。"
        )
    if "api" in message or "openai" in message or "connection" in message or "timeout" in message:
        return "请检查 user_data/secrets.yaml 里的 llm_api_key 是否有效，并确认网络可以访问对应 LLM 服务。"
    return "请查看上方错误信息；如果需要定位代码问题，可以把完整日志发给我。"


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
        log_user_friendly_error("配置错误", str(ce), hint="检查 user_data/secrets.yaml 和 user_data/plain_text_resume.yaml。")
    except FileNotFoundError as fnf:
        log_user_friendly_error("文件缺失", str(fnf), hint="默认数据目录是 user_data，也可以用 JOBAI_USER_DATA_DIR 指定其他目录。")
    except ValueError as ve:
        log_user_friendly_error("输入或数据错误", str(ve), hint="请按提示补全输入，或检查简历/JD 内容是否为空。")
    except RuntimeError as re:
        log_user_friendly_error("运行错误", str(re), hint=runtime_error_hint(re))
        logger.debug(traceback.format_exc())
    except Exception as e:
        logger.exception(f"未预期错误：{e}")
        logger.error("处理建议：这是未覆盖的异常。请保留完整日志，方便继续定位。")


if __name__ == "__main__":
    main()
