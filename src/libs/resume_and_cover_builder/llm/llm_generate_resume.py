"""
Create a class that generates a resume based on a resume and a resume template.
"""
# app/libs/resume_and_cover_builder/gpt_resume.py
import textwrap
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
                            max_bullets: int = 6,
                            max_en_words: int = 22,
                            max_cn_chars: int = 18) -> str:
    """
    Enforce output hygiene after generation:
    - Each role has 3–6 bullets (if too many, keep the most informative; if too few, leave as-is).
    - Each bullet length limit (EN by words, CN by chars). Overlong bullets are gracefully truncated with an ellipsis.
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

            # If too many bullets, keep top-K by text length (proxy for informativeness)
            if len(li_nodes) > max_bullets:
                li_nodes_sorted = sorted(li_nodes, key=lambda li: len(li.get_text(" ", strip=True)), reverse=True)
                keep = set(li_nodes_sorted[:max_bullets])
                for li in li_nodes:
                    if li not in keep:
                        li.decompose()
                li_nodes = [li for li in ul.find_all("li")]

            # If too few bullets (< min_bullets), do NOT fabricate content; leave as-is.

            # Enforce length per bullet
            for li in li_nodes:
                text = li.get_text(" ", strip=True)
                n, is_cn = _count_len(text)
                over_en = (not is_cn and n > max_en_words)
                over_cn = (is_cn and n > max_cn_chars)
                if over_en or over_cn:
                    if not is_cn:
                        # Cut by word tokens while preserving separators
                        tokens = re.findall(r"[A-Za-z0-9\-\+\./%]+|\W+", text)
                        kept, count = [], 0
                        for tok in tokens:
                            if re.match(r"[A-Za-z0-9\-\+\./%]+", tok):
                                count += 1
                            kept.append(tok)
                            if count >= max_en_words:
                                break
                        new_text = "".join(kept).strip(",;: .") + " …"
                    else:
                        new_text = text[:max_cn_chars].rstrip("，、；。.:,; ") + "…"
                    li.clear()
                    li.append(html.escape(new_text))

        return str(section)
    except Exception as e:
        try:
            from loguru import logger
            logger.debug("[POST] normalize skip due to error: {}", e)
        except Exception:
            pass
        return html_text

# Load environment variables from .env file
load_dotenv()


class LLMResumer:

    def __init__(self, openai_api_key, strings):
        self.llm_cheap = LoggerChatModel(
            ChatOpenAI(
                model_name="gpt-4.1", openai_api_key=openai_api_key, temperature=0.4
            )
        )
        self.strings = strings

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
        """Normalize Additional Skills HTML to avoid accidental bold bleed from malformed tags."""
        if not isinstance(output, str) or not output.strip():
            return output
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(output, "html.parser")
            for li in soup.find_all("li"):
                text = li.get_text(" ", strip=True)
                if text.lower().startswith("languages:"):
                    lang_text = text.split(":", 1)[1].strip() if ":" in text else text
                    li.clear()
                    strong = soup.new_tag("strong")
                    strong.string = "Languages:"
                    li.append(strong)
                    if lang_text:
                        li.append(" " + lang_text)
            return str(soup)
        except Exception:
            return output


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
    - Length & Density: 3–6 bullets per role; each bullet concise (avoid run-ons).
    - Relevance: Remove details not aligned to the JD; surface tools/domains that match the JD.
    - Tense: Current role may be present tense; past roles in past tense.
    - Tone: Active, specific, measurable; avoid fluff such as "responsible for".
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
    
    def generate_additional_skills_section(self, data = None) -> str:
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
            "languages": self.resume.languages,
            "interests": self.resume.interests,
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

        def projects_fn():
            if self.resume.projects:
                return self.generate_projects_section()
            return ""

        def achievements_fn():
            if self.resume.achievements:
                return self.generate_achievements_section()
            return ""
        

        def additional_skills_fn():
            if (self.resume.experience_details or self.resume.education_details or
                self.resume.languages or self.resume.interests):
                return self.generate_additional_skills_section()
            return ""

        # Create a dictionary to map the function names to their respective callables
        functions = {
            "header": header_fn,
            "education": education_fn,
            "work_experience": work_experience_fn,
            "projects": projects_fn,
            "achievements": achievements_fn,
            "additional_skills": additional_skills_fn,
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
        full_resume = "<body>\n"
        full_resume += f"  {results.get('header', '')}\n"
        full_resume += "  <main>\n"
        full_resume += f"    {results.get('education', '')}\n"
        full_resume += f"    {results.get('work_experience', '')}\n"
        full_resume += f"    {results.get('projects', '')}\n"
        full_resume += f"    {results.get('achievements', '')}\n"
        full_resume += f"    {results.get('additional_skills', '')}\n"
        full_resume += "  </main>\n"
        full_resume += "</body>"
        return full_resume
