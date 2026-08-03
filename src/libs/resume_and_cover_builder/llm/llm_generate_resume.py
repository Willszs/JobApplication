"""
Create a class that generates a resume based on a resume and a resume template.
"""
# app/libs/resume_and_cover_builder/gpt_resume.py
import textwrap
import os
import base64
import mimetypes
from pathlib import Path
from src.libs.resume_and_cover_builder.utils import LoggerChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from collections import Counter
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
import re  # 顶部若没有，补上

import html
from bs4 import BeautifulSoup
from config import LLM_MODEL
from src.libs.resume_and_cover_builder.config import global_config

def _is_chinese_char(ch: str) -> bool:
    """Rudimentary CJK check for length counting."""
    return any([
        '\u4e00' <= ch <= '\u9fff',   # CJK Unified Ideographs
        '\u3400' <= ch <= '\u4dbf',   # CJK Extension A
        '\u3000' <= ch <= '\u303f',   # CJK Symbols & Punctuation
    ])

def _count_len(s: str) -> tuple[int, bool]:
    """
    Return (length, is_chinese_like).
    For Chinese-like text count characters; for English-like count words.
    """
    s = s.strip()
    if not s:
        return 0, False
    cn_chars = sum(1 for c in s if _is_chinese_char(c))
    is_cn = cn_chars >= max(1, len(s) // 2)
    if is_cn:
        return len(s), True
    words = re.findall(r"[A-Za-z0-9\-\+\./%]+", s)
    return len(words), False

def _normalize_work_section(html_text: str,
                            min_bullets: int = 3,
                            max_bullets: int = 6) -> str:
    """
    Enforce output hygiene after generation:
    - Each role has 3–6 bullets (if too many, keep the most informative; if too few, leave as-is).
    - Freelance roles have at most 2 bullets to avoid overweighting short-term work.
    """
    if 'BeautifulSoup' not in globals() or BeautifulSoup is None:
        # Gracefully skip if bs4 is not installed
        try:
            from loguru import logger  # local import to avoid hard dependency up top
            logger.warning("[POST] BeautifulSoup not installed; skip normalization.")
        except Exception:
            pass
        return html_text

    try:
        soup = BeautifulSoup(html_text, "html.parser")
        section = soup.find("section", id=lambda x: x and x.lower() in ("work-experience", "work experience"))
        if not section:
            return html_text  # unexpected structure; return as is

        for entry in section.select("div.entry"):
            ul = entry.find("ul", class_="compact-list") or entry.find("ul")
            if not ul:
                continue

            li_nodes = ul.find_all("li")
            title_node = entry.select_one(".entry-title")
            title_text = title_node.get_text(" ", strip=True).lower() if title_node else ""
            entry_max_bullets = 2 if "freelance" in title_text else max_bullets

            # If too many bullets, keep top-K by text length (proxy for informativeness)
            if len(li_nodes) > entry_max_bullets:
                li_nodes_sorted = sorted(li_nodes, key=lambda li: len(li.get_text(" ", strip=True)), reverse=True)
                keep = set(li_nodes_sorted[:entry_max_bullets])
                for li in li_nodes:
                    if li not in keep:
                        li.decompose()
                li_nodes = [li for li in ul.find_all("li")]

            # If too few bullets (< min_bullets), do NOT fabricate content; leave as-is.

            # Do not truncate overlong bullets here. Truncating after generation creates
            # unfinished sentences in the final resume; prompt rules should keep bullets concise.
            for li in li_nodes:
                text = li.get_text(" ", strip=True)
                if text.endswith(("...", "…")):
                    logger.warning("[POST] Work experience bullet ends with ellipsis; leaving text unchanged for manual review: {}", text)

        return str(section)
    except Exception as e:
        try:
            from loguru import logger
            logger.debug("[POST] normalize skip due to error: {}", e)
        except Exception:
            pass
        return html_text


def _resolve_user_data_folder() -> Path | None:
    """Resolve runtime user_data folder from current config/env."""
    output_path = getattr(global_config, "LOG_OUTPUT_FILE_PATH", None)
    if output_path:
        try:
            return Path(output_path).resolve().parent
        except Exception:
            pass

    env_path = os.getenv("JOBAI_USER_DATA_DIR")
    if env_path:
        candidate = Path(env_path).expanduser()
        if candidate.exists():
            return candidate.resolve()

    for folder_name in ("user_data", "data_folder"):
        candidate = (Path.cwd() / folder_name).resolve()
        if candidate.exists():
            return candidate

    return None


def _find_profile_photo_src() -> str | None:
    """Auto-discover a profile photo from user_data/photo (or user_data/photos)."""
    user_data_folder = _resolve_user_data_folder()
    if not user_data_folder:
        return None

    supported_suffixes = {".jpg", ".jpeg", ".png", ".webp"}
    preferred_stems = {"profile", "photo", "avatar", "headshot"}
    candidate_folders = [user_data_folder / "photo", user_data_folder / "photos"]

    for folder in candidate_folders:
        if not folder.is_dir():
            continue

        images = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in supported_suffixes]
        if not images:
            continue

        def _sort_key(path: Path) -> tuple[int, float, str]:
            stem_priority = 0 if path.stem.lower() in preferred_stems else 1
            try:
                mtime_sort = -path.stat().st_mtime
            except OSError:
                mtime_sort = 0.0
            return (stem_priority, mtime_sort, path.name.lower())

        selected = sorted(images, key=_sort_key)[0].resolve()
        logger.info("Using profile photo from: {}", selected)
        data_uri = _image_path_to_data_uri(selected)
        if data_uri:
            return data_uri
        logger.warning("Falling back to file URI for profile photo: {}", selected)
        return selected.as_uri()

    return None


def _image_path_to_data_uri(image_path: Path) -> str | None:
    """Encode local image as data URI to avoid file:// loading restrictions in browser preview."""
    if not image_path.exists() or not image_path.is_file():
        return None

    mime_type, _ = mimetypes.guess_type(image_path.name)
    if mime_type is None:
        suffix_to_mime = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }
        mime_type = suffix_to_mime.get(image_path.suffix.lower())
    if not mime_type:
        return None

    try:
        image_bytes = image_path.read_bytes()
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"
    except OSError as exc:
        logger.warning("Failed to read profile photo {}: {}", image_path, exc)
        return None


def _inject_profile_photo_in_header(header_html: str) -> str:
    """Inject profile photo into header if one is found in the photo folder."""
    if not isinstance(header_html, str) or not header_html.strip():
        return header_html

    photo_src = _find_profile_photo_src()
    if not photo_src:
        return header_html

    try:
        soup = BeautifulSoup(header_html, "html.parser")
        header = soup.find("header")
        if not header:
            return header_html

        if header.find("img", class_="profile-photo"):
            return str(soup)

        header_classes = header.get("class", [])
        if "with-photo" not in header_classes:
            header_classes.append("with-photo")
            header["class"] = header_classes

        header_main = soup.new_tag("div")
        header_main["class"] = ["header-main"]

        for child in list(header.contents):
            header_main.append(child.extract())

        header.append(header_main)

        photo = soup.new_tag("img")
        photo["class"] = ["profile-photo"]
        photo["src"] = photo_src
        photo["alt"] = "Profile photo"
        header.append(photo)
        return str(soup)
    except Exception as exc:
        logger.debug("Profile photo injection skipped due to parse error: {}", exc)
        return header_html


# Load environment variables from .env file
load_dotenv()


class LLMResumer:

    def __init__(self, openai_api_key, strings):
        self.llm_cheap = LoggerChatModel(
            ChatOpenAI(
                model_name=LLM_MODEL, openai_api_key=openai_api_key, temperature=0.4
            )
        )
        self.strings = strings
        self.target_resume_language = "en"

    @staticmethod
    def _normalize_language_code(language_code: str | None) -> str:
        if not language_code:
            return "en"

        normalized = str(language_code).strip().lower()
        if normalized.startswith(("zh", "cn", "chinese", "中文")):
            return "zh"
        return "en"

    def _get_target_resume_language(self) -> str:
        return self._normalize_language_code(getattr(self, "target_resume_language", "en"))

    @staticmethod
    def _strip_inline_image_data_uris(resume_html: str) -> tuple[str, dict[str, str]]:
        """
        Replace huge inline image data URIs with placeholders before sending HTML to the LLM.
        This keeps localization requests small and prevents rate-limit errors from oversized payloads.
        """
        if not isinstance(resume_html, str) or not resume_html.strip():
            return resume_html, {}

        try:
            soup = BeautifulSoup(resume_html, "html.parser")
            placeholder_to_src: dict[str, str] = {}
            idx = 0
            for img in soup.find_all("img"):
                src = img.get("src")
                if isinstance(src, str) and src.startswith("data:image/") and ";base64," in src:
                    placeholder = f"__JOBAI_IMG_DATA_URI_{idx}__"
                    placeholder_to_src[placeholder] = src
                    img["src"] = placeholder
                    idx += 1
            return str(soup), placeholder_to_src
        except Exception as exc:
            logger.debug("[LOC] Failed to strip inline image data URIs: {}", exc)
            return resume_html, {}

    @staticmethod
    def _restore_inline_image_data_uris(localized_html: str, placeholder_to_src: dict[str, str]) -> str:
        """Restore inline image data URIs after localization."""
        if not isinstance(localized_html, str) or not localized_html.strip():
            return localized_html
        if not placeholder_to_src:
            return localized_html

        restored = localized_html
        for placeholder, src in placeholder_to_src.items():
            restored = restored.replace(placeholder, src)
        return restored

    def _localize_resume_html_if_needed(self, resume_html: str) -> str:
        """
        Localize full resume HTML according to target language.
        For Chinese target, force Simplified Chinese output while preserving proper nouns/brands/links.
        """
        if not isinstance(resume_html, str) or not resume_html.strip():
            return resume_html

        if self._get_target_resume_language() != "zh":
            return resume_html

        localization_prompt = self._preprocess_template_string(
            """
            You are a professional resume localization editor.
            Rewrite the user-visible text in the HTML resume into Simplified Chinese.

            Mandatory constraints:
            - Keep ALL HTML tags/attributes/id/class/style unchanged.
            - Keep URLs, emails, phone numbers, and date formats unchanged.
            - Keep proper nouns in original form when they are brands, products, platforms, tools, libraries, certificates, or company names.
              Examples that must stay as-is when present: GitLab, LinkedIn, GitHub, CMake, Jenkins, GoogleTest, CI/CD.
            - Keep foreign company names in their original script/casing.
            - Keep acronyms and technical tokens unchanged when translation would reduce clarity.
            - Do NOT add or remove sections, bullets, or links.
            - Return HTML only, no markdown fences, no explanations.

            HTML:
            {resume_html}
            """
        )
        prompt = ChatPromptTemplate.from_template(localization_prompt)
        chain = prompt | self.llm_cheap | StrOutputParser()

        try:
            llm_input_html, image_placeholders = self._strip_inline_image_data_uris(resume_html)
            if image_placeholders:
                logger.info("[LOC] Stripped {} inline image data URI(s) before localization.", len(image_placeholders))

            localized_html = (chain.invoke({"resume_html": llm_input_html}) or "").strip()
            localized_html = localized_html.replace("```html", "").replace("```", "").strip()

            if not localized_html:
                logger.warning("[LOC] Empty localization output. Falling back to original HTML.")
                return resume_html
            if "<body" not in localized_html.lower():
                logger.warning("[LOC] Localization output is not a full body block. Falling back to original HTML.")
                return resume_html

            localized_html = self._restore_inline_image_data_uris(localized_html, image_placeholders)
            unresolved_placeholders = [p for p in image_placeholders if p in localized_html]
            if unresolved_placeholders:
                logger.warning("[LOC] Some image placeholders were not restored. Falling back to original HTML.")
                return resume_html

            logger.info("[LOC] Resume localized to Simplified Chinese based on JD language.")
            return localized_html
        except Exception as exc:
            logger.warning("[LOC] Localization step failed, fallback to original HTML: {}", exc)
            return resume_html

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

    @staticmethod
    def _normalize_additional_skills_html(output: str) -> str:
        """Normalize Skills HTML into stable grouped rows."""
        if not isinstance(output, str) or not output.strip():
            return output
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(output, "html.parser")
            section = soup.find("section", id=lambda value: value in ("skills", "skills-languages"))
            if section:
                section["id"] = "skills"
            else:
                section = soup.new_tag("section")
                section["id"] = "skills"
                heading = soup.new_tag("h2")
                heading.string = "Skills"
                section.append(heading)
                soup = BeautifulSoup(str(section) + str(soup), "html.parser")
                section = soup.find("section", id="skills")

            heading = soup.find("h2")
            if heading and heading.get_text(" ", strip=True).lower() == "additional skills":
                heading.string = "Skills"
            elif section and not section.find("h2"):
                heading = soup.new_tag("h2")
                heading.string = "Skills"
                section.insert(0, heading)

            def clean_skill_text(text: str) -> str:
                text = re.sub(r"\s+", " ", (text or "")).strip()
                text = text.strip("|•·-–— ")
                text = re.sub(r"\s+[·•]\s*$", "", text).strip()
                text = re.sub(r"\s+([,;:])", r"\1", text)
                return text

            skill_texts: list[str] = []
            source_nodes = []
            if section:
                source_nodes = section.select(".skill-item")
                if not source_nodes:
                    source_nodes = section.find_all("li")
                if not source_nodes:
                    source_nodes = [
                        node for node in section.find_all(["p", "div"], recursive=True)
                        if node.get_text(" ", strip=True)
                    ]

            for node in source_nodes:
                text = clean_skill_text(node.get_text(" ", strip=True))
                if not text or "[" in text or "]" in text:
                    continue
                if text.lower().startswith("languages:"):
                    text = "Languages: " + text.split(":", 1)[1].strip()
                skill_texts.append(text)

            if section and not skill_texts:
                for line in section.get_text("\n", strip=True).splitlines():
                    text = clean_skill_text(line)
                    if text and text.lower() not in {"skills", "additional skills"} and "[" not in text:
                        skill_texts.append(text)

            if section and skill_texts:
                seen = set()
                regular_items = []
                language_item = ""
                for text in skill_texts:
                    key = re.sub(r"\s+", " ", text).strip().lower()
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    if key.startswith("languages:"):
                        language_item = text
                    else:
                        regular_items.append(text)

                skills_list = soup.new_tag("ul")
                skills_list["class"] = ["skills-list"]
                for text in regular_items + ([language_item] if language_item else []):
                    li = soup.new_tag("li")
                    if ":" in text:
                        label, values = text.split(":", 1)
                        strong = soup.new_tag("strong")
                        strong.string = f"{label.strip()}:"
                        li.append(strong)
                        li.append(f" {values.strip()}")
                    else:
                        li.string = text
                    skills_list.append(li)

                for child in list(section.contents):
                    if getattr(child, "name", None) != "h2":
                        child.extract()
                section.append(skills_list)
            return str(soup)
        except Exception:
            return output

    @staticmethod
    def _html_to_visible_text(html_text: str) -> str:
        if not isinstance(html_text, str) or not html_text.strip():
            return ""
        try:
            soup = BeautifulSoup(html_text, "html.parser")
            return soup.get_text(separator="\n", strip=True)
        except Exception:
            return re.sub(r"<[^>]+>", " ", html_text)


    def set_resume(self, resume) -> None:
        """
        Set the resume object to be used for generating the resume.
        Args:
            resume (Resume): The resume object to be used.
        """
        self.resume = resume

    def generate_header(self, data = None) -> str:
        """
        Generate the header section of the resume.
        Args:
            data (dict): The personal information to use for generating the header.
        Returns:
            str: The generated header section.
        """
        header_prompt_template = self._preprocess_template_string(
            self.strings.prompt_header
        )
        prompt = ChatPromptTemplate.from_template(header_prompt_template)
        chain = prompt | self.llm_cheap | StrOutputParser()
        input_data = {
            "personal_information": self.resume.personal_information
        } if data is None else data
        output = chain.invoke(input_data)
        return output
    
    def generate_education_section(self, data = None) -> str:
        """
        Generate the education section of the resume.
        Args:
            data (dict): The education details to use for generating the education section.
        Returns:
            str: The generated education section.
        """
        logger.debug("Starting education section generation")

        education_prompt_template = self._preprocess_template_string(self.strings.prompt_education)
        logger.debug(f"Education template: {education_prompt_template}")

        prompt = ChatPromptTemplate.from_template(education_prompt_template)
        logger.debug(f"Prompt: {prompt}")
        
        chain = prompt | self.llm_cheap | StrOutputParser()
        logger.debug(f"Chain created: {chain}")
        
        input_data = {
            "education_details": self.resume.education_details
        } if data is None else data
        output = chain.invoke(input_data)
        logger.debug(f"Chain invocation result: {output}")

        logger.debug("Education section generation completed")
        return output
    

    # in src/libs/resume_and_cover_builder/llm/llm_generate_resume.py

    def generate_work_experience_section(self, data=None) -> str:
        """
        Generate the work experience section of the resume.
        Returns:
            str: The generated (likely HTML) work experience section from the LLM.
        """

        logger.debug("Starting work experience section generation")

        # 1) Instruction prompt (LLM control text; not the pure HTML template)
        work_experience_prompt_template = self._preprocess_template_string(
            self.strings.prompt_working_experience
        )

        # 追加强制改写/对齐规则
        rewrite_rules = """
    # Rewrite Rules (MUST FOLLOW)
    - No copy-paste: Do NOT reuse any 3+ consecutive words from the input anywhere.
    - Targeting: Prioritize these skills/keywords: {target_skills}.
    - Must-have vocabulary (from JD): {must_include_terms}.
    - Bullet style: CAR framing (Challenge → Action → Result). Start with a strong verb; end with a quantified result.
    - Quantify: If exact numbers are missing, use clear approximations (~, ≈) and label them as such.
    - Length & Density: 3–6 bullets per role; for any role whose title contains "Freelance", use at most 2 bullets and keep only the strongest evidence.
    - Relevance: Remove details not aligned to the JD; surface tools/domains that match the JD.
    - Tense: Current role may be present tense; past roles in past tense.
    - Tone: Active, specific, measurable; avoid fluff such as "responsible for".
    - Complete sentences: every bullet must be a complete sentence or complete action phrase. Never end a bullet with "...", "…", or an unfinished clause.
    - Keep bullets concise by rewriting shorter, not by truncating.
    """
        full_template = work_experience_prompt_template + "\n" + rewrite_rules
        prompt = ChatPromptTemplate.from_template(full_template)

        # Sanity checks
        logger.debug("[CHK] work_exp prompt contains <section>? {}", "<section" in work_experience_prompt_template)
        needed_vars = set(re.findall(r"\{([a-zA-Z0-9_]+)\}", full_template))
        logger.debug("[CHK] work_exp expects vars: {}", needed_vars)

        # 2) Build input payload: serialize experiences and ensure JD/summary are present
        if data is None:
            raw_list = self.resume.experience_details or []
            exp_dicts = [
                (e.__dict__ if hasattr(e, "__dict__") else vars(e))
                if not isinstance(e, dict) else e
                for e in raw_list
            ]
            input_data = {
                "experience_details": exp_dicts,
                "job_description": getattr(self, "job_description", "") or "",
                "job_description_summary": getattr(self, "_job_description_summary", "") or "",
                "resume_data": {"experience_details": exp_dicts},
            }
        else:
            input_data = dict(data)
            input_data.setdefault("job_description", getattr(self, "job_description", "") or "")
            input_data.setdefault("job_description_summary", getattr(self, "_job_description_summary", "") or "")
            if "experience_details" in input_data:
                lst = input_data["experience_details"] or []
                input_data["experience_details"] = [
                    (e.__dict__ if hasattr(e, "__dict__") else vars(e))
                    if not isinstance(e, dict) else e
                    for e in lst
                ]

        logger.debug("[CHK] provided keys: {}", set(input_data.keys()))

        # Fallback: 用摘要替代过短/空的 JD
        jd = (input_data.get("job_description") or "").strip()
        jds = (input_data.get("job_description_summary") or "").strip()
        if len(jd) < 120 and len(jds) > len(jd):
            input_data["job_description"] = jds
        logger.debug("[CHK] final job_description chars={}", len(input_data["job_description"]))

        # 3) 提取 target_skills（若子类已传入，则保留）
        def _extract_target_skills(text: str, top_k: int = 12) -> str:
            text_l = (text or "").lower()
            known_terms = [
                "c++20", "c++17", "c++", "oop", "solid", "design patterns",
                "linux", "linux networking", "tcp", "udp", "sockets", "posix",
                "jenkins", "gitlab ci", "ci/cd", "cmake", "gtest", "gmock",
                "clang-tidy", "cppcheck", "gerrit", "jira",
                "bash", "python", "devops", "ros", "ros2"
            ]
            hits = []
            for term in known_terms:
                pat = re.escape(term)
                if re.search(rf"\b{pat}\b", text_l):
                    hits.append(term)

            tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-\+\.#/]{1,19}", text or "")
            stop = {
                "the","and","with","for","you","your","our","to","of","in","on","a","an",
                "ability","experience","skills","work","team","strong","good","including",
                "preferred","required","responsibilities","requirements","role","description",
                "knowledge","using","familiarity","understanding","excellent","proven"
            }
            alias = {"js":"javascript","ts":"typescript","py":"python","db":"database","ml":"machine-learning"}
            tokens = [alias.get(t.lower(), t.lower()) for t in tokens if t.lower() not in stop]

            ordered, seen = [], set()
            for t in hits:
                if t not in seen:
                    seen.add(t); ordered.append(t)
            for w, _ in Counter(tokens).most_common(64):
                if w not in seen:
                    seen.add(w); ordered.append(w)

            if not ordered:
                ordered = ["c++20","oop","design patterns","linux","linux networking",
                        "jenkins","gerrit","jira","bash","python","cmake","ci/cd"]
            return ", ".join(ordered[:top_k])

        if not input_data.get("target_skills"):
            try:
                input_data["target_skills"] = _extract_target_skills(input_data["job_description"])
            except Exception as e:
                logger.warning("target_skills extraction failed: %s", e)
                input_data["target_skills"] = ""
        logger.debug("[CHK] target_skills: {}", input_data["target_skills"])

        # 4) must_include_terms（若子类已传入，则保留）
        if not input_data.get("must_include_terms"):
            must_pool = ["c++20","oop","design patterns","linux networking","jenkins",
                        "gerrit","jira","bash","python","cmake","ci/cd","linux"]
            jd_lower = (input_data["job_description"] or "").lower()
            must_include = [w for w in must_pool if w in jd_lower]
            input_data["must_include_terms"] = ", ".join(must_include)

        # Prompt 预览
        try:
            preview = prompt.format(**{k: (str(v)[:1000]) for k, v in input_data.items()})
            logger.debug("[EXP] WorkExp prompt preview:\n{}", preview[:1800])
        except Exception as e:
            logger.debug("[EXP] Preview skipped: %s", e)

        # 5) 调用 LLM
        chain = prompt | self.llm_cheap | StrOutputParser()

        # 构造相似度防护基线
        raw_concat = ""
        for e in (self.resume.experience_details or []):
            d = e if isinstance(e, dict) else (e.__dict__ if hasattr(e, "__dict__") else vars(e))
            pieces = [
                d.get("summary", ""),
                " ".join(d.get("highlights", []) or []),
                d.get("description", ""),
                d.get("title", ""),
            ]
            raw_concat += " ".join([p for p in pieces if p]) + "\n"

        def _ngram_set(text: str, n: int = 3) -> set[tuple[str, ...]]:
            toks = re.findall(r"[a-zA-Z0-9\-\+\./%]+", (text or "").lower())
            return set(tuple(toks[i:i+n]) for i in range(0, max(0, len(toks)-n+1)))

        def _violates_3gram_guard(source_texts: list[str], generated_text: str, n: int = 3) -> bool:
            src = set()
            for s in source_texts:
                src |= _ngram_set(s, n)
            gen = _ngram_set(generated_text, n)
            return len(src.intersection(gen)) > 0

        def _too_similar(src: str, gen: str, thresh: float = 0.82) -> bool:
            norm = lambda s: re.sub(r"\s+", " ", (s or "").lower()).strip()
            return SequenceMatcher(None, norm(src), norm(gen)).ratio() >= thresh

        output = chain.invoke(input_data).strip()

        # Guard 1: 反抄袭
        source_blobs = []
        for e in (self.resume.experience_details or []):
            d = e if isinstance(e, dict) else (e.__dict__ if hasattr(e, "__dict__") else vars(e))
            source_blobs.append((d.get("summary") or "") + " " + " ".join(d.get("highlights") or []))

        if _violates_3gram_guard(source_blobs, output, n=3) or _too_similar(raw_concat, output):
            logger.debug("[GUARD] Output too similar to input. Regenerating with stronger rewrite rule.")
            prompt2 = ChatPromptTemplate.from_template(
                full_template + """
    - Anti-plagiarism reinforcement: Rephrase more aggressively; vary syntax and verbs; restructure bullets.
    - Absolutely avoid any 3+ consecutive words identical to the input anywhere.
    """
            )
            output = (prompt2 | self.llm_cheap | StrOutputParser()).invoke(input_data).strip()

        # Guard 2: 必备术语覆盖（温和修复）
        missing_terms = []
        out_lower = output.lower()
        for t in (input_data.get("must_include_terms", "") or "").split(","):
            t = t.strip()
            if t and t not in out_lower:
                missing_terms.append(t)
        if missing_terms:
            logger.debug("[GUARD] Missing must-have terms; attempting coverage fix: {}", missing_terms[:5])
            prompt3 = ChatPromptTemplate.from_template(
                full_template + f"""
    - Coverage fix: Naturally incorporate at least one of these terms where relevant: {missing_terms[:5]}.
    - Do NOT add irrelevant fluff; integrate terms only where they make sense.
    """
            )
            output2 = (prompt3 | self.llm_cheap | StrOutputParser()).invoke(input_data).strip()
            if not _violates_3gram_guard(source_blobs, output2, n=3) and not _too_similar(raw_concat, output2):
                output = output2

        # 6) 归一化
        output = _normalize_work_section(output)

        # 7) 兜底：包成 HTML
        if not output.lstrip().startswith("<"):
            output = f"<section id='work-experience'><h2>Work Experience</h2><div>{output}</div></section>"

        logger.info(
            "[RESULT][WORK-EXP] ===== GENERATED HTML START =====\n{}\n[RESULT][WORK-EXP] ===== GENERATED HTML END =====",
            output
        )
        logger.debug("Work experience section generation completed")
        return output



    def generate_projects_section(self, data = None) -> str:
        """
        Generate the side projects section of the resume.
        Args:
            data (dict): The side projects to use for generating the side projects section.
        Returns:
            str: The generated side projects section.
        """
        logger.debug("Starting side projects section generation")

        projects_prompt_template = self._preprocess_template_string(self.strings.prompt_projects)
        logger.debug(f"Side projects template: {projects_prompt_template}")

        prompt = ChatPromptTemplate.from_template(projects_prompt_template)
        logger.debug(f"Prompt: {prompt}")
        
        chain = prompt | self.llm_cheap | StrOutputParser()
        logger.debug(f"Chain created: {chain}")
        
        input_data = {
            "projects": self.resume.projects
        } if data is None else data
        output = chain.invoke(input_data)
        logger.debug(f"Chain invocation result: {output}")

        logger.debug("Side projects section generation completed")
        return output

    def generate_achievements_section(self, data = None) -> str:
        """
        Generate the achievements section of the resume.
        Args:
            data (dict): The achievements to use for generating the achievements section.
        Returns:
            str: The generated achievements section.
        """
        logger.debug("Starting achievements section generation")

        achievements_prompt_template = self._preprocess_template_string(self.strings.prompt_achievements)
        logger.debug(f"Achievements template: {achievements_prompt_template}")

        prompt = ChatPromptTemplate.from_template(achievements_prompt_template)
        logger.debug(f"Prompt: {prompt}")

        chain = prompt | self.llm_cheap | StrOutputParser()
        logger.debug(f"Chain created: {chain}")

        input_data = {
            "achievements": self.resume.achievements,
        } if data is None else data
        logger.debug(f"Input data for the chain: {input_data}")

        output = chain.invoke(input_data)
        logger.debug(f"Chain invocation result: {output}")

        logger.debug("Achievements section generation completed")
        return output
    
    def generate_additional_skills_section(self, work_experience_html: str = "", data = None) -> str:
        """
        Generate the additional skills section of the resume.
        Returns:
            str: The generated additional skills section.
        """
        additional_skills_prompt_template = self._preprocess_template_string(self.strings.prompt_additional_skills)
        
        skills = set()
        if self.resume.experience_details:
            for exp in self.resume.experience_details:
                if exp.skills_acquired:
                    skills.update(exp.skills_acquired)

        if self.resume.education_details:
            for edu in self.resume.education_details:
                if edu.exam:
                    for exam in edu.exam:
                        skills.update(exam.keys())
        prompt = ChatPromptTemplate.from_template(additional_skills_prompt_template)
        chain = prompt | self.llm_cheap | StrOutputParser()
        input_data = {
            "work_experience": self._html_to_visible_text(work_experience_html),
            "languages": self.resume.languages,
            "skills": skills,
        } if data is None else data
        output = chain.invoke(input_data)
        return self._normalize_additional_skills_html(output)

    def generate_html_resume(self) -> str:
        """
        Generate the full HTML resume based on the resume object.
        Returns:
            str: The generated HTML resume.
        """
        def header_fn():
            if self.resume.personal_information:
                return self.generate_header()
            return ""

        def education_fn():
            if self.resume.education_details:
                return self.generate_education_section()
            return ""

        def work_experience_fn():
            if self.resume.experience_details:
                return self.generate_work_experience_section()
            return ""

        def achievements_fn():
            if self.resume.achievements:
                return self.generate_achievements_section()
            return ""
        

        # Create a dictionary to map the function names to their respective callables
        functions = {
            "header": header_fn,
            "education": education_fn,
            "work_experience": work_experience_fn,
            "achievements": achievements_fn,
        }

        # Use ThreadPoolExecutor to run the functions in parallel
        with ThreadPoolExecutor() as executor:
            future_to_section = {executor.submit(fn): section for section, fn in functions.items()}
            results = {}
            for future in as_completed(future_to_section):
                section = future_to_section[future]
                try:
                    result = future.result()
                    if result:
                        results[section] = result
                except Exception as exc:
                    logger.error(f'{section} raised an exception: {exc}')

        if results.get("header"):
            results["header"] = _inject_profile_photo_in_header(results["header"])

        if (results.get("work_experience") or self.resume.languages):
            results["additional_skills"] = self.generate_additional_skills_section(
                work_experience_html=results.get("work_experience", "")
            )

        full_resume = "<body>\n"
        full_resume += f"  {results.get('header', '')}\n"
        full_resume += "  <main>\n"
        full_resume += f"    {results.get('education', '')}\n"
        full_resume += f"    {results.get('work_experience', '')}\n"
        full_resume += f"    {results.get('achievements', '')}\n"
        full_resume += f"    {results.get('additional_skills', '')}\n"
        full_resume += "  </main>\n"
        full_resume += "</body>"
        return self._localize_resume_html_if_needed(full_resume)
