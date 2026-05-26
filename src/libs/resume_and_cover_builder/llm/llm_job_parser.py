import textwrap
import re  # For email validation
import json
import html as html_lib
from src.libs.resume_and_cover_builder.utils import LoggerChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser  # 严格 JSON 解析
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from loguru import logger
from langchain_text_splitters import TokenTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from config import LLM_MODEL

# Load environment variables from the .env file
load_dotenv()



class LLMParser:
    def __init__(self, openai_api_key):
        self.llm = LoggerChatModel(
            ChatOpenAI(
                model_name=LLM_MODEL,
                openai_api_key=openai_api_key,
                temperature=0.2
            )
        )
        self.llm_embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.vectorstore = None  # Will be initialized after document loading
        self._full_text = ""        # 保存（裁剪+清洗后）的全文纯文本，用于正则/兜底
        self._trimmed_html = ""     # 保存域内裁剪后的 HTML，供分节抽取
        self._extracted_cache = None  # 一次性抽取结果缓存（dict）

    @staticmethod
    def _preprocess_template_string(template: str) -> str:
        """
        Preprocess the template string by removing leading whitespaces and indentation.
        Args:
            template (str): The template string to preprocess.
        Returns:
            str: The preprocessed template string.
        """
        return textwrap.dedent(template)

    # -------------------------
    # 域内裁剪 & 可见文本清洗
    # -------------------------
    def _domain_specific_trim(self, raw_html: str) -> str:
        """
        将整页 HTML 裁剪为“职位描述主容器”区域（针对 LinkedIn 做多选择器兜底）。
        匹配不到则返回原始 HTML。
        """
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(raw_html, "html.parser")

            candidates = [
                # 典型职位描述正文容器
                "div.show-more-less-html__markup",
                "section.show-more-less-html",
                "div.jobs-description__content",
                "div.jobs-box__html-content",
                # Indeed 常见容器
                "div#jobDescriptionText",
                "div.jobsearch-JobComponent-description",
                "div[data-testid='jobsearch-JobComponent-description']",
                # 通用主内容
                "main",
                "article",
                "div[role='main']",
                # 次级容器（仍以正文为主）
                "div.jobs-details__main-content",
                "section.core-section-container",
            ]
            for css in candidates:
                node = soup.select_one(css)
                if node and node.get_text(strip=True):
                    logger.debug(f"Domain trim matched container: {css}")
                    return str(node)
            return raw_html
        except Exception as e:
            logger.warning(f"Domain-specific trim failed; fallback to raw HTML. Error: {e}")
            return raw_html

    def _extract_jobposting_from_jsonld(self, raw_html: str) -> str:
        """
        从页面 JSON-LD 中抽取 JobPosting。Indeed 等站点常把完整 JD 放在这里。
        """
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(raw_html, "html.parser")
            scripts = soup.find_all("script", attrs={"type": "application/ld+json"})
            if not scripts:
                return ""

            def _iter_nodes(node):
                if isinstance(node, dict):
                    yield node
                    for v in node.values():
                        yield from _iter_nodes(v)
                elif isinstance(node, list):
                    for item in node:
                        yield from _iter_nodes(item)

            def _extract_location(loc_obj):
                if not isinstance(loc_obj, dict):
                    return ""
                addr = loc_obj.get("address", {}) if isinstance(loc_obj.get("address"), dict) else {}
                parts = [
                    addr.get("addressLocality", ""),
                    addr.get("addressRegion", ""),
                    addr.get("addressCountry", ""),
                ]
                return ", ".join([p for p in parts if p])

            for sc in scripts:
                raw = (sc.string or sc.get_text() or "").strip()
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                except Exception:
                    continue

                for node in _iter_nodes(data):
                    if not isinstance(node, dict):
                        continue
                    ntype = str(node.get("@type", "")).lower()
                    if "jobposting" not in ntype:
                        continue

                    title = str(node.get("title", "") or "").strip()
                    org = node.get("hiringOrganization", {})
                    company = (org.get("name", "") if isinstance(org, dict) else "") or ""
                    location = _extract_location(node.get("jobLocation", {}))
                    desc_raw = str(node.get("description", "") or "")
                    desc_text = self._visible_text_clean(desc_raw) if desc_raw else ""

                    merged = (
                        f"Job Title: {title}\n"
                        f"Company: {company}\n"
                        f"Location: {location}\n\n"
                        f"Description:\n{desc_text}"
                    ).strip()
                    if len(merged) >= 120:
                        logger.debug("JobPosting JSON-LD extracted successfully.")
                        return merged
            return ""
        except Exception as e:
            logger.warning(f"JSON-LD JobPosting extraction failed: {e}")
            return ""

    def _visible_text_clean(self, html: str) -> str:
        """
        去除脚本/样式/导航/页脚等噪声，返回可见文本。
        """
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "iframe", "svg"]):
            tag.decompose()

        noisy_classes = ("nav", "menu", "footer", "header", "breadcrumbs", "subscribe", "cookie")
        for div in soup.find_all(True):
            cls = " ".join(div.get("class", [])).lower()
            if any(x in cls for x in noisy_classes):
                div.decompose()

        text = soup.get_text(separator="\n", strip=True)
        return text

    def _sanitize_text_for_llm(self, text: str, max_chars: int = 18000) -> str:
        """
        强制将输入压缩为可控大小，避免把整页脚本/导航噪声送给 LLM 导致 429。
        """
        if not text:
            return ""

        text = html_lib.unescape(text)
        # 防御性去标签（即使上游清洗失败，也尽量还原成纯文本）
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()

        if len(text) <= max_chars:
            return text

        keywords = [
            "responsibil", "requirement", "qualification", "experience", "skills",
            "key responsibilities", "knowledge", "what we offer", "benefits",
            "岗位职责", "工作职责", "职位描述", "任职要求", "技能要求", "加分项",
            "aufgaben", "anforderungen", "qualifikation", "kenntnisse", "voraussetzungen",
        ]

        segments = []
        # 保留开头（通常有职位名、公司、摘要）
        segments.append(text[:4000])
        for kw in keywords:
            for m in re.finditer(re.escape(kw), text, flags=re.IGNORECASE):
                start = max(0, m.start() - 1200)
                end = min(len(text), m.end() + 2600)
                segments.append(text[start:end])
                if len(segments) >= 10:
                    break
            if len(segments) >= 10:
                break
        # 再保留结尾（常见“requirements/offer”落在后半段）
        segments.append(text[-3000:])

        compact = []
        seen = set()
        for seg in segments:
            s = seg.strip()
            if not s:
                continue
            key = s[:200]
            if key in seen:
                continue
            seen.add(key)
            compact.append(s)

        merged = "\n\n".join(compact).strip()
        return merged[:max_chars]

    def _extract_section_from_html(
        self,
        html: str,
        header_keywords: list[str],
        stop_keywords: list[str],
        other_section_headers: list[str] | None = None,
        max_chars: int = 8000
    ) -> str:
        """
        从裁剪后的职位 HTML 中，按标题关键字提取某个分节（Responsibilities / Requirements）。
        改进点：
          - 将“另一节”的标题也视为 stop（避免把两节混在一起）
          - 只遍历标题节点的“后续兄弟节点”（next_siblings），避免跨层游走
          - 只收集 bullet 列表与段落，直到遇到 stop 或新标题
        """
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")

            title_tags = ["h1", "h2", "h3", "h4", "strong", "b", "p", "span"]
            kw_lower = [k.lower() for k in header_keywords]
            stop_lower = [k.lower() for k in stop_keywords]
            other_lower = [k.lower() for k in (other_section_headers or [])]

            def _norm(s: str) -> str:
                return (s or "").strip().lower()

            def is_header_text(t: str) -> bool:
                t = _norm(t)
                return any(k in t for k in kw_lower)

            def is_stop_text(t: str) -> bool:
                t = _norm(t)
                return any(k in t for k in stop_lower) or any(k in t for k in other_lower)

            # 1) 找标题节点（优先 h1~h4/strong/b/p/span 中的匹配）
            header_node = None
            for tag in title_tags:
                for node in soup.find_all(tag):
                    txt = node.get_text(separator=" ", strip=True)
                    if is_header_text(txt):
                        header_node = node
                        break
                if header_node:
                    break
            if not header_node:
                return ""

            # 2) 只遍历“后续同级兄弟节点”
            collected = []
            total = 0
            for sib in header_node.next_siblings:
                if getattr(sib, "name", None) in title_tags:
                    txt = sib.get_text(separator=" ", strip=True)
                    if is_header_text(txt) or is_stop_text(txt):
                        break

                if getattr(sib, "name", None) in ("ul", "ol"):
                    for li in sib.find_all("li"):
                        bullet = li.get_text(separator=" ", strip=True)
                        if bullet:
                            collected.append(f"• {bullet}")
                            total += len(bullet)
                elif getattr(sib, "name", None) == "p":
                    para = sib.get_text(separator=" ", strip=True)
                    if para:
                        collected.append(para)
                        total += len(para)

                if total > max_chars:
                    break

            return "\n".join(collected).strip()
        except Exception as e:
            logger.warning(f"Section extraction failed: {e}")
            return ""

    def _normalize_bullets(self, text: str, max_bullets: int = 12) -> str:
        """
        规整为 bullets：
          - 丢弃空行和非信息性标题（如 "that's you", "your responsibilities"）
          - 只保留以 '•' / '-' / '–' / '*' 开头的行；若没有，则按换行切分再做前缀化
          - 限制条数
        """
        if not text:
            return ""

        drop_markers = [
            "that's you", "that´s you", "your responsibilities", "responsibilities",
            "what you will do", "what you'll do", "requirements", "qualifications",
            "must have", "nice to have", "about you",
            "岗位职责", "工作职责", "职位职责", "工作内容", "主要职责", "职责描述",
            "任职要求", "职位要求", "岗位要求", "资格要求", "加分项", "你将负责", "我们希望你"
        ]
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        kept = []
        for ln in lines:
            low = ln.lower()
            if any(m in low for m in drop_markers):
                continue
            if ln.startswith(("•", "-", "–", "*")):
                kept.append("• " + ln.lstrip("•-*– ").strip())
            else:
                kept.append(ln)

        if not any(k.startswith("• ") for k in kept):
            kept = [f"• {k}" for k in kept]

        return "\n".join(kept[:max_bullets]).strip()

    def set_body_html(self, body_html):
        """
        Retrieves the job description from HTML, processes it, and initializes the vectorstore.
        Args:
            body_html (str): The HTML content to process.
        """
        # A) 先做域内裁剪（关键：只保留“职位描述主容器”，避免推荐/其他岗位串扰）
        trimmed_html = self._domain_specific_trim(body_html)
        self._trimmed_html = trimmed_html  # 保存供分节抽取
        logger.debug("Domain-specific trim completed.")

        # B) 尝试优先从 JSON-LD JobPosting 获取结构化正文（Indeed 常见）
        jsonld_job_text = self._extract_jobposting_from_jsonld(body_html)

        # C) 清洗得到可见文本（供正则/回退使用）
        try:
            visible_text = self._visible_text_clean(trimmed_html)
        except Exception as e:
            logger.warning(f"Visible text cleaning failed, fallback to raw HTML text: {e}")
            visible_text = trimmed_html

        # D) 合并并强制压缩，确保不会把整页脏 HTML 直接送给 LLM
        if jsonld_job_text and len(jsonld_job_text) >= 120:
            merged_text = f"{jsonld_job_text}\n\n{visible_text}"
        else:
            merged_text = visible_text

        self._full_text = self._sanitize_text_for_llm(merged_text, max_chars=18000)
        logger.debug(f"Full text (cleaned & compacted) length: {len(self._full_text)}")

        # E) 构造单一文档供切块与向量化
        from langchain_core.documents import Document
        documents = [Document(page_content=self._full_text)]

        # F) 切块
        text_splitter = TokenTextSplitter(chunk_size=900, chunk_overlap=120)
        all_splits = text_splitter.split_documents(documents)
        logger.debug(f"Text split into {len(all_splits)} fragments (no truncation in logs).")

        # G) 向量库
        try:
            self.vectorstore = FAISS.from_documents(documents=all_splits, embedding=self.llm_embeddings)
            logger.debug("Vectorstore successfully initialized.")
        except Exception as e:
            logger.error(f"Error during vectorstore creation: {e}")
            raise

        # H) 重置缓存
        self._extracted_cache = None

    def _retrieve_context(self, query: str, top_k: int = 5) -> str:
        """
        Retrieves the most relevant text fragments using the retriever.
        Args:
            query (str): The search query.
            top_k (int): Number of fragments to retrieve.
        Returns:
            str: Concatenated text fragments.
        """
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized. Call set_body_html() first.")

        retriever = self.vectorstore.as_retriever()
        retrieved_docs = retriever.get_relevant_documents(query)[:top_k]
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        logger.debug(f"Context retrieved for query [{query}]:\n{context}\n--- END OF CONTEXT ---")
        return context

    def _extract_information(self, question: str, retrieval_query: str) -> str:
        """
        (兼容保留) 单字段抽取：保留原有行为，内部将优先使用一次性抽取缓存。
        """
        data = self._ensure_all_extracted()
        mapping = {
            "What is the job description of the company?": "job_description",
            "What is the company's name?": "company_name",
            "What is the role or title sought in this job description?": "role",
            "What is the location mentioned in this job description?": "location",
            "What is the recruiter's email address in this job description?": "recruiter_email",
        }
        key = mapping.get(question)
        if key and key in data:
            return data.get(key, "") or ""

        # 兜底一次（理论上不会走到这里）
        context = self._retrieve_context(retrieval_query)
        prompt = ChatPromptTemplate.from_template(
            template="""
You are an expert in extracting specific information from job descriptions.
Carefully read the job description context below and provide a clear and concise answer to the question.

Context:
{context}

Question: {question}
Answer:
"""
        )
        formatted_prompt = prompt.format(context=context, question=question)
        logger.debug(f"Formatted prompt for single-field extraction (no truncation):\n{formatted_prompt}")

        try:
            chain = prompt | self.llm | StrOutputParser()
            result = chain.invoke({"context": context, "question": question})
            extracted_info = result.strip()
            logger.debug(f"Single-field extracted information (no truncation):\n{extracted_info}")
            return extracted_info
        except Exception as e:
            logger.error(f"Error during information extraction: {e}")
            return ""

    # -------------------------
    # 一次性结构化抽取
    # -------------------------
    def _ensure_all_extracted(self) -> dict:
        """
        确保一次性抽取已完成；若无缓存则执行一次并缓存。
        """
        if self._extracted_cache is not None:
            return self._extracted_cache

        self._extracted_cache = self.extract_all()
        return self._extracted_cache

    def extract_all(self) -> dict:
        """
        一次性抽取所有字段，严格 JSON 输出，键名固定：
        ["company_name","role","location","recruiter_email","job_description","job_responsibilities","job_requirements"]
        """
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized. Call set_body_html() first.")

        # A) 更强语义的查询，扩大覆盖到正文/要求/职责/技术栈/福利等关键词；拉高 top_k
        context = self._retrieve_context(
            ("Full job description, responsibilities, requirements, qualifications, tech stack, benefits, "
             "about the role, about you, what you will do, what we offer, tasks"),
            top_k=10
        )

        # B) 统一 JSON 抽取（基础元字段 + job_description）
        prompt = ChatPromptTemplate.from_template("""
You will extract fields from a job description. 
Return STRICT JSON with keys exactly:
["company_name","role","location","recruiter_email","job_description"].
If unknown, use "" (empty string). Do not add extra keys. Do not add commentary.

Context:
{context}

JSON:
""")

        chain = prompt | self.llm | JsonOutputParser()
        result = chain.invoke({"context": context})

        logger.debug("Unified extraction raw JSON (no truncation):\n{}".format(result))

        # C) 邮箱：全文正则优先覆盖
        email_regex = r'[\w\.-]+@[\w\.-]+\.\w+'
        regex_emails = re.findall(email_regex, self._full_text) if self._full_text else []
        if regex_emails:
            result["recruiter_email"] = regex_emails[0]

        # D) 如果 job_description 为空，做一次“全文提炼”兜底（直接用清洗后的全文，不走检索）
        if not result.get("job_description"):
            fallback_prompt = ChatPromptTemplate.from_template("""
You are given the full plain text of a job post. 
Produce a concise but complete job description paragraph (5-10 sentences) covering: company intro (if present), role scope, core responsibilities, required skills, preferred/bonus skills, tools/tech stack, and any benefits or work arrangement details.
Do NOT invent facts not in the text. If something is not present, omit it.
Return plain text only.

FULL TEXT:
{full_text}

DESCRIPTION:
""")
            try:
                desc_chain = fallback_prompt | self.llm | StrOutputParser()
                jd_text = desc_chain.invoke({"full_text": self._full_text})
                result["job_description"] = jd_text.strip()
            except Exception as e:
                logger.error(f"Fallback description extraction failed: {e}")
                result["job_description"] = ""

        # E) 规则优先：从裁剪后的 HTML 中抓 Responsibilities / Requirements（互相作为 stop）
        resp = ""
        reqs = ""
        if self._trimmed_html:
            resp = self._extract_section_from_html(
                self._trimmed_html,
                header_keywords=[
                    "your responsibilities","responsibilities","what you will do","what you'll do",
                    "tasks","role responsibilities","about the role",
                    "岗位职责","工作职责","职位职责","工作内容","主要职责","职责描述","你将负责","你将做什么"
                ],
                stop_keywords=[
                    "benefits","what we offer","perks","why us","about us","company","culture",
                    "福利","我们提供","薪酬福利","关于我们","公司介绍","企业文化","团队介绍"
                ],
                other_section_headers=[
                    "that's you", "that´s you", "requirements","what you bring",
                    "what we are looking for","qualifications","skills","must have","nice to have","about you",
                    "任职要求","职位要求","岗位要求","资格要求","我们希望你","你需要具备","加分项","你是谁"
                ]
            )
            reqs = self._extract_section_from_html(
                self._trimmed_html,
                header_keywords=[
                    "that's you","that´s you","requirements","what you bring","what we are looking for",
                    "qualifications","skills","must have","nice to have","about you",
                    "任职要求","职位要求","岗位要求","资格要求","我们希望你","你需要具备","加分项","你是谁"
                ],
                stop_keywords=[
                    "benefits","what we offer","perks","why us","about us","company","culture",
                    "福利","我们提供","薪酬福利","关于我们","公司介绍","企业文化","团队介绍"
                ],
                other_section_headers=[
                    "your responsibilities","responsibilities","what you will do","what you'll do",
                    "tasks","role responsibilities","about the role",
                    "岗位职责","工作职责","职位职责","工作内容","主要职责","职责描述","你将负责","你将做什么"
                ]
            )

        # F) 兜底：LLM 分别抽 bullets（如果规则没抓到）
        RESP_PROMPT = ChatPromptTemplate.from_template("""
You are given the cleaned text of a single job post.
Extract ONLY the **job responsibilities/duties** as bullet points.
Rules:
- Derive bullets strictly from the text (no invention, no generalization).
- Keep at most {max_bullets} bullets; merge duplicates; normalize style.
- Output plain text, one bullet per line, each starting with "• ".

TEXT:
{txt}
""")

        REQ_PROMPT = ChatPromptTemplate.from_template("""
You are given the cleaned text of a single job post.
Extract ONLY the **requirements/qualifications** (must-have & nice-to-have) as bullet points.
Rules:
- Derive bullets strictly from the text (no invention).
- Keep at most {max_bullets} bullets; merge duplicates; normalize style.
- Output plain text, one bullet per line, each starting with "• ".

TEXT:
{txt}
""")

        MAX_BULLETS = 12
        if not resp:
            try:
                resp_chain = RESP_PROMPT | self.llm | StrOutputParser()
                resp = resp_chain.invoke({"txt": self._full_text, "max_bullets": MAX_BULLETS}).strip()
            except Exception as e:
                logger.warning(f"LLM responsibilities fallback failed: {e}")
                resp = ""
        if not reqs:
            try:
                req_chain = REQ_PROMPT | self.llm | StrOutputParser()
                reqs = req_chain.invoke({"txt": self._full_text, "max_bullets": MAX_BULLETS}).strip()
            except Exception as e:
                logger.warning(f"LLM requirements fallback failed: {e}")
                reqs = ""

        # —— 规整为 bullets，清除标题行，限制条数 ——
        resp = self._normalize_bullets(resp, max_bullets=MAX_BULLETS)
        reqs = self._normalize_bullets(reqs, max_bullets=MAX_BULLETS)

        # G) 兜底确保所有键存在 + 写入 responsibilities / requirements
        for k in ["company_name", "role", "location", "recruiter_email", "job_description"]:
            if k not in result or result[k] is None:
                result[k] = ""
        result["job_responsibilities"] = resp or ""
        result["job_requirements"] = reqs or ""

        # H) 全量打印（不省略）
        logger.info(
            "Unified extraction completed (no truncation):\n"
            f"company_name: {result['company_name']}\n"
            f"role: {result['role']}\n"
            f"location: {result['location']}\n"
            f"recruiter_email: {result['recruiter_email']}\n"
            "job_description:\n"
            f"{result['job_description']}\n"
            "job_responsibilities:\n"
            f"{result['job_responsibilities']}\n"
            "job_requirements:\n"
            f"{result['job_requirements']}\n"
            "--- END OF FIELDS ---"
        )

        return result

    # -------------------------
    # 兼容：保留原有方法名，对外接口不变
    # -------------------------
    def extract_job_description(self) -> str:
        """
        Backward-compatible alias.
        Prefer 'job_responsibilities' content; fall back to 'job_description'.
        """
        data = self._ensure_all_extracted()
        return data.get("job_responsibilities", "") or data.get("job_description", "") or ""

    def extract_company_name(self) -> str:
        """
        Extracts the company name from the job description.
        Returns:
            str: The extracted company name.
        """
        data = self._ensure_all_extracted()
        return data.get("company_name", "") or ""

    def extract_role(self) -> str:
        """
        Extracts the sought role/title from the job description.
        Returns:
            str: The extracted role/title.
        """
        data = self._ensure_all_extracted()
        return data.get("role", "") or ""

    def extract_location(self) -> str:
        """
        Extracts the location from the job description.
        Returns:
            str: The extracted location.
        """
        data = self._ensure_all_extracted()
        return data.get("location", "") or ""

    def extract_recruiter_email(self) -> str:
        """
        Extracts the recruiter's email from the job description.
        Returns:
            str: The extracted recruiter's email.
        """
        # 先用全文正则（强模式优先）
        email_regex = r'[\w\.-]+@[\w\.-]+\.\w+'
        if self._full_text:
            found = re.findall(email_regex, self._full_text)
            if found:
                logger.debug(f"Recruiter email found by regex (no truncation): {found[0]}")
                return found[0]

        # 回退到一次性抽取
        data = self._ensure_all_extracted()
        email = data.get("recruiter_email", "") or ""
        if re.match(email_regex, email):
            return email
        return ""

    # 新增：职责 & 要求
    def extract_job_responsibilities(self) -> str:
        """
        Extracts job responsibilities bullet list (rules/LLM extracted).
        """
        data = self._ensure_all_extracted()
        return data.get("job_responsibilities", "") or data.get("job_description", "") or ""

    def extract_job_requirements(self) -> str:
        """
        Extracts job requirements/qualifications bullet list (rules/LLM extracted).
        """
        data = self._ensure_all_extracted()
        return data.get("job_requirements", "") or ""

    # -------------------------
    # 便捷打印：无省略输出
    # -------------------------
    def print_all_extracted_info(self):
        """
        打印所有抽取结果（不省略）。方便调试与日志留痕。
        """
        data = self._ensure_all_extracted()
        text = (
            "=== EXTRACTED FIELDS (NO TRUNCATION) ===\n"
            f"company_name: {data.get('company_name','')}\n"
            f"role: {data.get('role','')}\n"
            f"location: {data.get('location','')}\n"
            f"recruiter_email: {data.get('recruiter_email','')}\n"
            "job_description:\n"
            f"{data.get('job_description','')}\n"
            "job_responsibilities:\n"
            f"{data.get('job_responsibilities','')}\n"
            "job_requirements:\n"
            f"{data.get('job_requirements','')}\n"
            "=== END ==="
        )
        print(text)
        logger.info(text)
