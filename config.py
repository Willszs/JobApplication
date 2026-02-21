# In this file, you can set the configurations of the app.

from src.utils.constants import DEBUG, ERROR, LLM_MODEL, OPENAI

# ==== Logging ====
LOG_LEVEL = 'DEBUG'          # 调高日志级别，便于看到过程
# ==== Logging ====
LOG_LEVEL = 'DEBUG'          # 调高日志级别，便于看到过程
LOG_SELENIUM_LEVEL = ERROR
LOG_TO_FILE = False
LOG_TO_CONSOLE = True        # 打印到控制台，排错更直观
LOG_TO_CONSOLE = True        # 打印到控制台，排错更直观

# ==== Runtime ====
MINIMUM_WAIT_TIME_IN_SECONDS = 5   # 先调小，避免等待太久
# ==== Runtime ====
MINIMUM_WAIT_TIME_IN_SECONDS = 5   # 先调小，避免等待太久
JOB_APPLICATIONS_DIR = "job_applications"
JOB_SUITABILITY_SCORE = 7
JOB_MAX_APPLICATIONS = 5
JOB_MIN_APPLICATIONS = 1

# ==== LLM 选择 ====
LLM_MODEL_TYPE = 'openai'          # openai / anthropic / gemini / ollama
LLM_MODEL = 'gpt-4.1'              # 你账户可用的具体模型名
LLM_API_URL = ''                   # 只在 ollama/self-host 时需要，OpenAI 留空
# ==== LLM 选择 ====
LLM_MODEL_TYPE = 'openai'          # openai / anthropic / gemini / ollama
LLM_MODEL = 'gpt-4.1'              # 你账户可用的具体模型名
LLM_API_URL = ''                   # 只在 ollama/self-host 时需要，OpenAI 留空
