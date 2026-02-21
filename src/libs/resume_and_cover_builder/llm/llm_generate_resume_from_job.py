"""
Create a class that generates a job description based on a resume and a job description template.
"""
# app/libs/resume_and_cover_builder/llm_generate_resume_from_job.py
import re
from dotenv import load_dotenv
from loguru import logger
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.libs.resume_and_cover_builder.llm.llm_generate_resume import LLMResumer

# 放在 imports 后面
import json


# Load environment variables from .env file
load_dotenv()


class LLMResumeJobDescription(LLMResumer):
    def __init__(self, openai_api_key, strings):
        super().__init__(openai_api_key, strings)
        # 显式初始化，外部可读取
        self._job_description_raw = ""
        self._job_description_summary = ""   # ✅ 存“JD SUMMARY (full)”全文（Markdown/段落）
        self.job_description_summary = ""    # 兼容无下划线读取
        # 可选：外部若想手动设置“职责要点”，可赋值到该属性
        self.job_responsibilities = ""

    # ---- JD 摘要 ----
    def set_job_description_from_text(self, job_description_text) -> None:
        """
        接收纯文本 JD，生成“JD SUMMARY (full)”（段落版），并把【原始 JD】和【摘要全文】都挂载到实例属性。
        同时把渲染后的 Prompt、输入与模型原始输出落盘，方便排查。
        """
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        if not job_description_text or not str(job_description_text).strip():
            raise ValueError("Job description text is empty.")

        # 构建链
        prompt = ChatPromptTemplate.from_template(self.strings.summarize_prompt_template)
        chain = prompt | self.llm_cheap | StrOutputParser()

        logger.info(f"[JD] summarize_prompt_template exists? {hasattr(self.strings, 'summarize_prompt_template')}")
        logger.debug(f"[JD] summarize_prompt_template preview:\n{getattr(self.strings, 'summarize_prompt_template', '')[:300]}")

        summary = chain.invoke({"text": job_description_text}) or ""
        logger.debug(f"[OUTPUT summarize] ===== START =====\n{summary[:2000]}\n===== END =====")

        # 属性持久化
        self._job_description_raw = (job_description_text or "").strip()
        self._job_description_summary = (summary or "").strip()
        self.job_description_summary = self._job_description_summary
        self.job_description = self._job_description_raw  # 兼容旧逻辑

        # 兜底
        if not self._job_description_summary:
            logger.warning("[JD] summary is EMPTY after summarize. Using RAW as fallback for downstream.")
            self._job_description_summary = self._job_description_raw
            self.job_description_summary = self._job_description_summary

        logger.info(f"[JD] summary length: {len(self._job_description_summary)}")
        logger.debug(f"[JD] summary preview:\n{self._job_description_summary[:1200]}")



    def get_job_description_summary(self) -> str:
        return getattr(self, "_job_description_summary", "")

    def get_job_description_raw(self) -> str:
        return getattr(self, "_job_description_raw", "")

    # ---- 其它段落直接转发到父类实现 ----
    def generate_header(self) -> str:
        # header 不需要强制 JD SUMMARY，保留旧行为
        return super().generate_header(data={
            "personal_information": self.resume.personal_information,
            "job_description": self.job_description
        })

    def generate_education_section(self) -> str:
        return super().generate_education_section(data={
            "education_details": self.resume.education_details,
            "job_description": self.job_description_summary or self.job_description
        })

    def generate_projects_section(self) -> str:
        return super().generate_projects_section(data={
            "projects": self.resume.projects,
            "job_description": self.job_description_summary or self.job_description
        })

    def generate_achievements_section(self) -> str:
        return super().generate_achievements_section(data={
            "achievements": self.resume.achievements,
            "job_description": self.job_description_summary or self.job_description
        })

    def generate_additional_skills_section(self) -> str:
        """
        额外技能段生成（带分层调试日志）：
        [A0]/[A1] 数据探针  → 看源数据/合并后的 skills_list 是否为空
        [B0]/[B1] 生成探针  → 看 Prompt 模板与 LLM 原样输出是否只剩 Languages
        """
        import json
        try:
            # ---------- [A0] 源数据快照 ----------
            langs0 = getattr(self.resume, "languages", None)
            inter0 = getattr(self.resume, "interests", None)
            add0   = getattr(self.resume, "additional_skills", None)
            skills0= getattr(self.resume, "skills", None)
            exp0   = getattr(self.resume, "experience_details", None)
            edu0   = getattr(self.resume, "education_details", None)

            logger.debug(f"[A0] resume.languages={langs0!r}")
            logger.debug(f"[A0] resume.interests={inter0!r}")
            logger.debug(f"[A0] resume.additional_skills={add0!r}")
            logger.debug(f"[A0] resume.skills={skills0!r}")
            if exp0:
                fields_map = [{k: bool(getattr(exp, k, None)) for k in ("skills_acquired","tools","technologies","keywords")} for exp in exp0]
                logger.debug(f"[A0] experience_details fields={fields_map}")
            if edu0:
                logger.debug(f"[A0] education_details sample={edu0[0] if len(edu0)>0 else None!r}")

            # ---------- 1) 预处理模板 ----------
            additional_skills_prompt_template = self._preprocess_template_string(
                self.strings.prompt_additional_skills
            )

            # ---------- 2) 收集技能（多源兜底） ----------
            skills_set = set()

            # 2.1 从工作经历抓
            if exp0:
                for exp in exp0:
                    for attr in ("skills_acquired", "tools", "technologies", "keywords"):
                        vals = getattr(exp, attr, None)
                        if isinstance(vals, (list, tuple, set)):
                            skills_set.update(map(str, vals))

            # 2.2 从教育信息抓
            if edu0:
                for edu in edu0:
                    for attr in ("skills", "courses", "coursework"):
                        vals = getattr(edu, attr, None)
                        if isinstance(vals, (list, tuple, set)):
                            skills_set.update(map(str, vals))

            # 2.3 从 additional_skills 子类兜底
            add = add0 or {}
            for k in ("tools", "frameworks", "devops", "soft_skills", "other"):
                vals = add.get(k)
                if isinstance(vals, (list, tuple, set)):
                    skills_set.update(map(str, vals))

            # 2.4 从 resume.skills 兜底
            if isinstance(skills0, (list, tuple, set)):
                skills_set.update(map(str, skills0))

            # 2.5 排序+清洗
            skills_list = sorted({s.strip() for s in skills_set if s and isinstance(s, str)})

            # ---------- 3) 语言/兴趣 统一为列表 ----------
            languages = langs0 or []
            interests = inter0 or []
            if isinstance(languages, str):
                languages = [languages]
            if isinstance(interests, str):
                interests = [interests]

            # ---------- 4) JD 文本化 ----------
            jd_input = self.job_description_summary or self.job_description or ""
            if not isinstance(jd_input, str):
                try:
                    jd_input = " ; ".join(
                        f"{k}: {', '.join(v) if isinstance(v, list) else v}"
                        for k, v in jd_input.items()
                    )
                except Exception:
                    jd_input = str(jd_input)

            # ---------- [A1] 合并后探针 ----------
            logger.debug(f"[A1] merged skills_list(len={len(skills_list)}) sample={skills_list[:8]}")
            logger.debug(f"[A1] languages(list)={languages}")
            logger.debug(f"[A1] interests(list)={interests}")
            jd_preview = jd_input[:300] if isinstance(jd_input, str) else jd_input
            jd_len = len(jd_input) if isinstance(jd_input, str) else "NA"
            logger.debug(f"[A1] jd_input(type={type(jd_input).__name__}, len≈{jd_len}) preview={jd_preview}")

            # ---------- 5) 组链并调用 ----------
            prompt = ChatPromptTemplate.from_template(additional_skills_prompt_template)
            chain = prompt | self.llm_strong if hasattr(self, "llm_strong") else (prompt | self.llm_cheap)

            # ---------- [B0] Prompt与入参探针 ----------
            tpl_preview = (additional_skills_prompt_template[:220] if additional_skills_prompt_template else None)
            logger.debug(f"[B0] prompt_additional_skills (first 220)={tpl_preview!r}")
            logger.debug(f"[B0] LLM inputs: langs={languages} | interests={interests} | skills_len={len(skills_list)} sample={skills_list[:5]}")

            output = (chain | StrOutputParser()).invoke({
                "languages": languages,
                "interests": interests,
                "skills": skills_list,
                "job_description": jd_input
            })

            # ---------- [B1] LLM原样输出探针 ----------
            if isinstance(output, str):
                logger.debug(f"[B1] LLM raw output (first 500)={output[:500]}")
            else:
                logger.debug(f"[B1] LLM raw output (non-str)={output!r}")

            return self._normalize_additional_skills_html(output)

        except Exception as exc:
            logger.exception(f"[ERR] generate_additional_skills_section failed: {exc}")
            return ""





    def generate_work_experience_section(self) -> str:
        """
        Edit-only 生成“工作经历”（以 JD ANALYSIS + 個人經歷為唯一依據，要求≥80%覆蓋，但不做程序校驗）。
        - 禁止捏造未出現在 SOURCE 的品牌/技術/數據；
        - 返回 <section id="work-experience"> 片段。
        - 僅落盤 10/11/20/21 四個文件，無 12/22。
        """
        import re
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        logger.debug("Starting work experience section generation (edit-only, no responsibilities, no coverage parsing)")

        # ---------- 1) 聚合输入 ----------
        exp_list = self.resume.experience_details or []
        jd_text = (
            getattr(self, "job_description_summary", "")
            or getattr(self, "_job_description_summary", "")
            or getattr(self, "job_description", "")
            or ""
        ).strip()

        # 统一 dict
        def _as_dict(e):
            if isinstance(e, dict):
                return e
            if hasattr(e, "__dict__"):
                return e.__dict__
            try:
                return vars(e)
            except Exception:
                return {}

        exp_dicts = [_as_dict(e) for e in exp_list]

        # 展平成可编辑源文本
        def _flatten_exp(d: dict) -> str:
            parts = []
            for k in ("position", "title", "company", "employment_period", "location",
                    "industry", "summary", "description"):
                v = d.get(k)
                if v:
                    parts.append(str(v))
            for item in (d.get("key_responsibilities") or []):
                parts.append(str(item))
            for x in (d.get("highlights") or []):
                parts.append(str(x))
            return "\n".join(parts)

        source_blocks = [_flatten_exp(d) for d in exp_dicts]
        source_text = "\n\n---\n\n".join(source_blocks).strip()

        if len(source_text) < 50:
            raise ValueError("[WorkExp] Source experience is too short/empty; nothing to edit.")

        # ---------- 2) Prompt：僅 JD ANALYSIS + SOURCE EXPERIENCE ----------
        base_prompt = r"""
        You are a senior HR/ATS resume editor. Transform the user’s work experience so it strongly aligns with the target Job **Analysis** (below). Aim for ≥80% theme coverage and high keyword resonance while keeping content plausible and professional.

        ### GUIDING PRINCIPLES
        - Alignment first: emphasize responsibilities, skills, methods, and tools that best match the JD themes.
        - Reasonable augmentation: you may infer and expand on typical duties and outcomes for similar roles when the context clearly supports them.
        - Brand/tool hygiene: feel free to introduce **brand-agnostic** capabilities (e.g., “version control branching and review,” “CI/CD automation,” “monitoring & logging”) even if the specific vendor name is not provided. Only name a specific brand/tool if it is already known or unambiguously implied.
        - Impact language: prefer CAR-style bullets (Challenge → Action → Result). Use strong verbs and clear outcomes; where numbers are not available, use qualitative impact (stability, reliability, latency, scalability, maintainability, cycle time).
        - Tense: current role → present; prior roles → past.
        - Brevity & ATS: 3–6 bullets per role; concise, skimmable, and keyword-rich without sounding stuffed.

        ### PRIORITIZATION
        - Cover the JD’s highest-weight themes first (domain, core responsibilities, required skills, must-have tools or their brand-agnostic equivalents).
        - Merge overlap, remove low-value details, and elevate scope to reflect professional-level accountability (ownership, cross-team collaboration, delivery constraints, quality/security considerations).
        - If a JD concept is absent, introduce a **short bridge bullet** phrased generically (capability-level) that fits the candidate’s context.

        ### INPUTS
        [JD ANALYSIS]
        {jd_analysis}

        [SOURCE EXPERIENCE (verbatim)]
        {source_experience}

        ### HTML TEMPLATE (return exactly this structure)
        <section id="work-experience">
        <h2>Work Experience</h2>
        <!-- Repeat one .entry per role (keep original order) -->
        <div class="entry">
        <div class="entry-header">
            <span class="entry-name">[Company Name]</span>
            <span class="entry-location">[Location]</span>
        </div>
        <div class="entry-details">
            <span class="entry-title">[Job Title]</span>
            <span class="entry-year">[Start – End]</span>
        </div>
        <ul class="compact-list">
            <li>[Edited bullet 1]</li>
            <li>[Edited bullet 2]</li>
            <li>[Edited bullet 3]</li>
            <li>[Edited bullet 4]</li>
            <li>[Edited bullet 5]</li>
        </ul>
        </div>
        </section>
        """


        prompt_obj = ChatPromptTemplate.from_template(base_prompt)
        inputs = {
            "jd_analysis": jd_text[:8000],
            "source_experience": source_text[:8000],
        }

        # ---------- 3) 調用模型（單次） ----------
        chain_local = prompt_obj | self.llm_cheap | StrOutputParser()
        html = (chain_local.invoke(inputs) or "").strip()
        logger.debug(f"[OUTPUT pass1] ===== START =====\n{html}\n===== END =====")

        # ---------- 4) 兜底 + 清理 ----------
        m_final = re.search(
            r"<section[^>]*id=['\"]work-experience['\"][\s\S]*?</section>",
            html, flags=re.IGNORECASE
        )
        if m_final:
            html = m_final.group(0).strip()

        if not html.lstrip().startswith("<"):
            html = f"<section id='work-experience'><h2>Work Experience</h2><div>{html}</div></section>"

        logger.info("[RESULT][WORK-EXP][EDIT-ONLY, no coverage] === HTML START ===\n{}\n=== HTML END ===", html)
        return html








