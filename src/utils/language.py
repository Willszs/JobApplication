import re


def detect_jd_language(jd_text: str) -> str:
    """Return 'zh' for Chinese-dominant job descriptions, otherwise 'en'."""
    text = (jd_text or "").strip()
    if not text:
        return "en"

    zh_markers = (
        "岗位职责", "工作职责", "职位描述", "工作内容", "任职要求",
        "岗位要求", "我们希望", "你将负责", "你将做什么", "加分项",
        "负责", "要求", "职位", "岗位", "职责",
    )
    if any(marker in text for marker in zh_markers):
        return "zh"

    zh_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin_chars = len(re.findall(r"[A-Za-z]", text))

    if zh_chars >= 10 and zh_chars >= int(latin_chars * 0.1):
        return "zh"
    if zh_chars >= 8 and zh_chars > latin_chars:
        return "zh"
    return "en"
