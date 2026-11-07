"""引擎模块：对话总结、用户画像更新、关键词提取。"""
import asyncio
import json
import re
import time
import uuid

from astrbot.api import logger

# ===== 话题边界（总结触发）=====
# 对话收尾语：出现即视为天然的话题边界，此刻总结不会把话题切两半。
_CLOSURE_RE = re.compile(
    r"就这样|先这样|没事了|没别的了|没有了|先忙|去忙|改天|下次再聊|"
    r"晚安|睡了|明天见|拜拜|再见|不聊了|先下了"
)
# 语义连贯阈值：最新一轮与前几轮的余弦相似度 ≥ 此值 → 判定"同一话题仍在继续"
_TOPIC_CONTINUATION_SIM = 0.7

# ===== 回溯/补充语（"对了""之前说的""更正一下"）=====
# 用户在给**已结束的话题**补充信息时，新内容的措辞往往与原记忆差异较大
# （只是共享几个实体词），用常规阈值会漏检。检测到这类信号时：
#   · 扩大喂给总结器的"已有记忆"窗口（8 → 15 条）
#   · 把冲突合并的相似度阈值从 0.85 放宽到 0.55
_BACKREF_RE = re.compile(
    r"对了|补充一下|补充个|再补充|补充点|更正|修正一下|说错|记错|纠正|"
    r"之前(?:说|提|聊|讲)|上次(?:说|提|聊|讲)|前面(?:说|提)|刚才(?:说|提)|"
    r"忘了说|还没说|另外(?:还有|那个)|顺便说|接着刚才|继续刚才|"
    r"不是.{1,12}是|其实(?:是|叫)|那个.{0,6}叫"
)
# 连续性上下文窗口：常规 8 条，检测到回溯语时放宽到 15 条
_CONTINUITY_WINDOW = 8
_CONTINUITY_WINDOW_BACKREF = 15

# 画像更新失败退避：超时/异常后按指数退避，避免"每条消息都重试同一个大 prompt"。
# 实测一次画像更新 prompt 约 3~5k 字符（≈3.5k tokens），Provider 忙时必然超时，
# 若不做退避会形成死循环——每次 60s 超时 + 全量重发，纯烧 token。
_PROFILE_BACKOFF_BASE = 120.0
_PROFILE_BACKOFF_MAX = 1800.0
# 冲突合并的相似度阈值
_CONFLICT_SIM_DEFAULT = 0.85
_CONFLICT_SIM_BACKREF = 0.55
# 因话题未结束而延后总结的最大轮数（硬上限，避免迟迟不总结）
_TOPIC_DEFER_MAX_ROUNDS = 3


class EngineMixin:
    """LLM 调用相关的高阶逻辑。"""

    # ========== 结构化总结解析 ==========
    @staticmethod
    def _extract_json_span(text: str):
        """从可能夹带说明文字的回复中，抠出第一个**配对完整**的 JSON 对象/数组。"""
        start = -1
        for i, ch in enumerate(text):
            if ch in "{[":
                start = i
                break
        if start < 0:
            return None
        open_ch = text[start]
        close_ch = "}" if open_ch == "{" else "]"
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == open_ch:
                depth += 1
            elif c == close_ch:
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        return None

    @classmethod
    def _parse_summary_items(cls, text: str) -> list:
        """解析总结输出，返回条目列表（支持一条总结里包含多个话题）。

        兼容形态：
          {"memories": [{...}, ...]}  新格式：按话题拆分
          [{...}, ...]                裸数组
          {"summary": "..."}          旧格式：单条
          纯文本                       回退为单条
        每条包含 summary / importance / tags / narrative_time。
        """
        def norm(d):
            if not isinstance(d, dict):
                return None
            s = str(d.get("summary", "") or "").strip()
            if not s:
                return None
            try:
                imp = int(d.get("importance", 3) or 3)
            except (TypeError, ValueError):
                imp = 3
            tags = d.get("tags")
            topic = str(d.get("topic", "") or "").strip()
            return {
                "summary": s,
                "importance": max(1, min(5, imp)),
                "tags": tags if isinstance(tags, list) else ([topic] if topic else []),
                "narrative_time": str(d.get("narrative_time", "") or "").strip(),
            }

        parsed_list = []
        try:
            parsed_list.append(json.loads(text))
        except (json.JSONDecodeError, ValueError, TypeError):
            span = cls._extract_json_span(text)
            if span:
                try:
                    parsed_list.append(json.loads(span))
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass

        for parsed in parsed_list:
            # 是否解析出了"结构化容器"——只要容器合法，就以它为准（**即使为空列表**），
            # 否则 LLM 明确表示"无信息"时（{"memories": []}）会被下面的纯文本兜底
            # 当成一条记忆存进去。
            if isinstance(parsed, dict) and isinstance(parsed.get("memories"), list):
                raw_items = parsed["memories"]
            elif isinstance(parsed, list):
                raw_items = parsed
            elif isinstance(parsed, dict) and "summary" in parsed:
                raw_items = [parsed]
            else:
                continue
            return [x for x in (norm(d) for d in raw_items) if x]

        # 没有任何结构化容器 → 回退为整段文本作为一条
        t = (text or "").strip()
        return [{"summary": t, "importance": 3, "tags": [], "narrative_time": ""}] if t else []

    def _build_enhanced_prompt(self, template: str, dialog_text: str, known: list = None,
                               backref: bool = False) -> str:
        """构建总结 prompt：注入已有记忆（用于话题衔接）并附加强制按话题拆分的输出要求。

        为什么需要"已有记忆"：模型看重的是"避免重复"和"续写同一话题"。
        缺少它时，一个跨越多轮的话题会被切成前后两条互不相干的总结。
        """
        prompt = template.replace("{对话历史}", dialog_text)

        if known:
            ctx = "\n".join(f"- {t}" for t in known)
            prompt += (
                "\n\n【已有记忆（重要）】\n"
                + ctx
                + "\n若本次对话是上述某条记忆的**同一话题**的延续、补充或修正，"
                  "请把新旧信息**融合成一条完整表述**；若某条已被本次对话完整覆盖且无新增信息，则不要重复输出。"
            )

        if backref:
            prompt += (
                "\n\n【注意】用户本轮很可能在**补充或更正较早提到过的话题**。"
                "请优先检查【已有记忆】中是否有相关条目：如果有，把新旧信息融合成一条完整表述"
                "（不要新建一条平行记忆）；如果旧记忆有误，以用户本次的说法为准。"
            )

        imp_on = self._get_config("memory_importance_enabled", True)
        fields = ['"summary": "该话题的完整信息"', '"topic": "话题标签(不超过8字)"']
        if imp_on:
            fields.append('"importance": 3')
        fields.append('"narrative_time": ""')

        prompt += (
            "\n\n请**按话题拆分**后输出 JSON，一个话题一条，不要把不相关的话题合并：\n"
            '{"memories": [' + "{" + ", ".join(fields) + "}" + "]}\n"
            "硬性要求：\n"
            "1. 不同话题（不同人物 / 不同事件 / 不同领域）必须拆成不同条目；\n"
            "2. 每条必须**自包含**、可独立理解，禁止出现「同上」「该话题」「上述」等指代；\n"
            "3. 同一话题若已存在于【已有记忆】，请融合成一条，不要产生两条相近记忆；\n"
            "4. 没有值得长期记住的信息时，输出 {\"memories\": []}。"
        )
        if imp_on:
            prompt += "\nimportance: 1-5 整数（5=极其重要如姓名/联系方式/重大事件，3=一般偏好/习惯，1=琐碎信息）"
        prompt += "\nnarrative_time: 对话中角色扮演/故事所处的叙事时间。优先提取用户明确提到的时间（如「7月21日周一下午」「圣诞夜」「2024年3月」），若只有相对时间（如「下周五」）则照写，没有特殊时间上下文则留空"
        return prompt
    # ========== 对话总结 ==========
    async def _summarize_buffer(self, scope_id: str, event=None, buffer: list = None) -> bool:
        """对话总结入口：带防重入保护，同一 scope 同时只允许一个总结任务。
        （时间/轮数/信息密度/手动触发并发时防止同一批对话被重复总结）
        """
        guard = getattr(self, "_summarizing_scopes", None)
        if guard is None:
            self._summarizing_scopes = set()
            guard = self._summarizing_scopes
        if scope_id in guard:
            logger.info(f"[长期记忆] 总结已在进行中，跳过本次触发 (scope={scope_id})")
            return False
        guard.add(scope_id)
        try:
            return await self._summarize_buffer_locked(scope_id, event, buffer)
        finally:
            guard.discard(scope_id)

    async def _summarize_buffer_locked(self, scope_id: str, event=None, buffer: list = None) -> bool:
        if buffer is None:
            buffer = await self._load_buffer(scope_id)
        if not buffer:
            return False

        # 过滤错误消息：排除包含 Python traceback / 异常信息的对话条目
        from .core import _looks_like_error
        dialog_items = []
        # 不可总结的条目（/ 命令对、错误消息）：这些永远不会进入任何总结，
        # 必须从缓冲区中剔除，否则它们会永久驻留并反复触发"无内容可总结"的空转。
        junk_ids = set()
        skip = False
        for item in buffer:
            role = item.get("role", "user")
            content = item.get("content", "")
            # 跳过以 / 开头的用户指令
            if role == "user" and content.strip().startswith("/"):
                skip = True
                if item.get("buf_id"):
                    junk_ids.add(item["buf_id"])
                continue
            if skip and role == "assistant":
                skip = False
                if item.get("buf_id"):
                    junk_ids.add(item["buf_id"])
                continue
            # 跳过包含错误信息的条目
            if _looks_like_error(content):
                logger.info(f"[长期记忆] 总结时跳过错误消息: {content[:80]}")
                if item.get("buf_id"):
                    junk_ids.add(item["buf_id"])
                # 如果助理消息是错误，则前一条用户消息也跳过（该轮对话无效）
                if role == "assistant" and dialog_items and dialog_items[-1]["role"] == "user":
                    removed = dialog_items.pop()
                    if removed.get("buf_id"):
                        junk_ids.add(removed["buf_id"])
                    logger.info(f"[长期记忆] 同时移除对应的用户消息: {removed.get('content', '')[:80]}")
                continue
            dialog_items.append(item)
            skip = False

        # 立即剔除垃圾条目（即使本次不产生总结），避免缓冲区被无法总结的内容占满
        if junk_ids:
            async with self._get_buffer_lock(scope_id):
                buf_now = await self._load_buffer_internal(scope_id)
                cleaned = [i for i in buf_now if i.get("buf_id") not in junk_ids]
                if len(cleaned) != len(buf_now):
                    await self._save_buffer_internal(scope_id, cleaned)
                    logger.debug(
                        f"[长期记忆] 已从缓冲区剔除 {len(buf_now) - len(cleaned)} 条不可总结条目"
                    )

        if not dialog_items:
            return False

        # 修复: 总结内容太短（总字符数 < 100）时跳过，避免低质量总结
        total_chars = sum(len(i.get("content", "")) for i in dialog_items)
        if total_chars < 100:
            logger.info(f"[长期记忆] 对话总长度仅 {total_chars} 字符，跳过总结")
            return False

        summarized_ids = {i["buf_id"] for i in dialog_items if "buf_id" in i}
        dialog_text = "\n".join(
            f"{'用户' if i['role']=='user' else '助手'}: {i['content']}" for i in dialog_items
        )

        template = self._get_config(
            "summary_prompt_template",
            "请总结以下对话中用户提供的关键信息、偏好、事实或重要上下文。输出简短段落。\n对话内容：\n{对话历史}"
        )
        if "{对话历史}" not in template:
            template = "请总结以下对话...\n{对话历史}"

        # 是否为"给已结束的话题补充信息"：这类内容措辞差异大，需要放宽匹配
        backref = self._has_backreference(dialog_items)
        if backref:
            logger.info("[长期记忆] 检测到回溯/补充语，放宽已有记忆窗口与合并阈值")

        # 取最近几条已有总结，供 LLM 做话题衔接：避免同一话题被切成前后两条
        known_texts = []
        try:
            memories = await self._load_memories(scope_id)
            recent = [m.get("text", "") for m in memories
                      if m.get("type") == "summary" and m.get("text")]
            window = _CONTINUITY_WINDOW_BACKREF if backref else _CONTINUITY_WINDOW
            known_texts = [t[:200] for t in recent[-window:]]
        except Exception:
            pass
        prompt = self._build_enhanced_prompt(template, dialog_text, known_texts, backref)

        umo = scope_id
        if event:
            try:
                umo = event.unified_msg_origin
            except Exception:
                pass

        timeout = self._get_config("llm_timeout_seconds", 60)
        max_retries = self._get_config("llm_max_retries", 2)
        provider_id = await self._get_llm_provider_id(umo)
        if not provider_id:
            logger.warning("[长期记忆] 无可用 LLM，取消总结")
            return False

        for attempt in range(max_retries + 1):
            try:
                resp = await asyncio.wait_for(
                    self.context.llm_generate(chat_provider_id=provider_id, prompt=prompt),
                    timeout=timeout
                )
                raw_text = resp.completion_text.strip().strip('"').strip("'")
                if not raw_text:
                    logger.warning("[长期记忆] LLM 返回空总结")
                    return False

                # 解析结构化输出：可能包含多个话题，逐条存储
                items = self._parse_summary_items(raw_text)
                dialog_ids = [i.get("message_id", "") for i in dialog_items if i.get("message_id")]
                conflict_on = self._get_config("memory_conflict_detection_enabled", True)
                saved = 0
                summary = ""
                for it in items:
                    text_i = it["summary"]
                    if conflict_on:
                        # 与已有记忆冲突/延续时，可能被合并（返回空串表示不再新增）
                        text_i = await self._detect_and_handle_conflict(
                            scope_id, text_i, provider_id, timeout, event, backref=backref
                        )
                    if not text_i:
                        continue
                    await self._add_memory(
                        scope_id, text_i, "summary", dialog_ids,
                        importance=it["importance"], tags=it["tags"],
                        narrative_time=it["narrative_time"],
                    )
                    saved += 1
                    if not summary:
                        summary = text_i
                if not items:
                    logger.info(f"[长期记忆] 本次对话无值得长期记忆的信息 ({scope_id})")
                elif not saved:
                    logger.info(f"[长期记忆] 总结内容已被既有记忆覆盖，未新增 ({scope_id})")

                if self._get_config("keep_raw_messages", False):
                    for item in dialog_items:
                        await self._add_memory(
                            scope_id,
                            f"[{item['role']}] {item['content']}",
                            "raw",
                            [item.get("message_id", "")]
                        )

                async with self._get_buffer_lock(scope_id):
                    buf = await self._load_buffer_internal(scope_id)
                    remain = [i for i in buf if i.get("buf_id") not in summarized_ids]
                    await self._save_buffer_internal(scope_id, remain)

                await self._save_last_summary_time(scope_id, time.time())
                logger.info(
                    f"[长期记忆] 总结成功 ({scope_id}): 新增 {saved} 条记忆"
                    + (f" | 首条: {summary[:60]}..." if summary else "")
                )

                # 人设漂移检测（总结后异步触发）
                if self._get_config("persona_drift_enabled", False):
                    asyncio.create_task(self._check_persona_drift(scope_id, event))

                return True

            except asyncio.TimeoutError:
                # 超时说明 Provider 忙或 prompt 过大：立即重试只会重复烧掉同样的 token。
                # 缓冲区保留，等下次触发自然重试即可。
                logger.warning(
                    f"[长期记忆] 总结超时 (attempt {attempt+1}/{max_retries+1}，"
                    f"prompt {len(prompt)/1000.0:.1f}k 字符)，放弃本轮重试"
                )
                break
            except Exception as e:
                logger.warning(f"[长期记忆] 总结失败: {e}")
                if attempt < max_retries:
                    await asyncio.sleep(min(2 ** attempt, 8))

        logger.warning("[长期记忆] 总结最终失败，缓冲区保留")
        return False

    async def _detect_and_handle_conflict(
        self, scope_id: str, new_summary: str,
        provider_id: str, timeout: int, event=None, backref: bool = False
    ) -> str:
        """检测新总结是否与已有记忆冲突，返回最终应存储的文本。

        返回空串表示"已就地合并进旧记忆，调用方不要再新增"。
        backref=True（检测到回溯/补充语）时把相似度阈值从 0.85 放宽到 0.55——
        用户给旧话题补充信息时，新表述与原记忆往往只共享少量实体词。
        """
        try:
            memories = await self._load_memories(scope_id)
            searchable = [m for m in memories if m.get("type") == "summary" and not m.get("_embed_failed")]
            if not searchable:
                return new_summary

            query_vec = await self._embed_text(new_summary)
            if not query_vec:
                return new_summary

            valid, valid_vecs = [], []
            for m in searchable:
                emb = m.get("embedding")
                if emb and isinstance(emb, (list, tuple)):
                    valid.append(m)
                    valid_vecs.append(emb)
            if not valid_vecs:
                return new_summary

            sims = self._batch_dot_product(query_vec, valid_vecs)
            # 找相似度最高的记忆
            best_idx = max(range(len(sims)), key=lambda i: sims[i])
            best_sim = sims[best_idx]
            threshold = _CONFLICT_SIM_BACKREF if backref else _CONFLICT_SIM_DEFAULT
            if best_sim < threshold:
                return new_summary

            best_mem = valid[best_idx]
            old_text = best_mem.get("text", "")
            if not old_text or len(old_text) < 10:
                return new_summary

            conflict_prompt = (
                "以下是一条已有的用户记忆和一条新的对话总结。请判断新总结是否是对已有记忆的更新/修正/细化。\n\n"
                f"已有记忆：{old_text}\n\n"
                f"新总结：{new_summary}\n\n"
                "请以 JSON 格式输出：\n"
                '{"is_conflict": true/false, "action": "update|keep_both", "merged_text": "合并后文本(仅 action=update 时填写)"}\n\n'
                "判断标准：\n"
                "- 如果新总结只是复述旧信息 → is_conflict=false, action=keep_both\n"
                "- 如果新总结修正/更新/细化了旧信息 → is_conflict=true, action=update\n"
                "- 如果新旧信息部分重叠但有新的独立信息 → is_conflict=false, action=keep_both"
            )

            resp = await asyncio.wait_for(
                self.context.llm_generate(chat_provider_id=provider_id, prompt=conflict_prompt),
                timeout=min(timeout, 30),
            )
            result_text = resp.completion_text.strip()

            try:
                result = json.loads(result_text)
            except json.JSONDecodeError:
                m = re.search(r'\{[^{}]*\}', result_text, re.DOTALL)
                if m:
                    try:
                        result = json.loads(m.group())
                    except json.JSONDecodeError:
                        return new_summary
                else:
                    return new_summary

            if result.get("is_conflict") and result.get("action") == "update":
                merged = str(result.get("merged_text", "")).strip()
                if merged and len(merged) >= 10:
                    # 锁外重新生成 embedding（避免持锁做 LLM 调用）
                    new_emb = await self._embed_text(merged)
                    meta = None
                    async with self._get_memory_lock(scope_id):
                        mems = await self._load_memories_internal(scope_id)
                        for m in mems:
                            if m.get("id") == best_mem["id"]:
                                old = m["text"]
                                m["text"] = merged
                                if new_emb:
                                    m["embedding"] = new_emb
                                    m.pop("_embed_failed", None)
                                else:
                                    m["_embed_failed"] = True
                                await self._save_memories_internal(scope_id, mems)
                                meta = {
                                    "type": m.get("type", "summary"),
                                    "importance": m.get("importance", 3),
                                    "tags": m.get("tags", []) or [],
                                    "timestamp": m.get("timestamp", time.time()),
                                    "narrative_time": m.get("narrative_time", ""),
                                }
                                logger.info(
                                    f"[长期记忆] 冲突合并: '{old[:40]}...' → '{merged[:40]}...' "
                                    f"(相似度={best_sim:.2f})"
                                )
                                break
                    # 同步 ChromaDB：有向量则更新，无向量则移除旧记录
                    if self._get_config("chromadb_enabled", False) and meta:
                        if new_emb:
                            await self._chroma_upsert_memory(
                                scope_id, best_mem["id"], merged, new_emb,
                                meta["type"], meta["importance"], meta["tags"],
                                meta["timestamp"], meta["narrative_time"],
                            )
                        else:
                            await self._chroma_delete_memory(scope_id, best_mem["id"])
                    return ""  # 返回空表示已合并，不再新增

            return new_summary

        except asyncio.TimeoutError:
            logger.debug("[长期记忆] 冲突检测超时，保留新总结")
        except Exception as e:
            logger.debug(f"[长期记忆] 冲突检测异常: {e}")
        return new_summary

    async def _check_and_trigger_summary(self, scope_id, event, buffer: list = None) -> bool:
        trigger_rounds = self._get_config("summary_trigger_rounds", 5)
        interval = self._get_config("summary_interval_seconds", 300)
        min_rounds = self._get_config("min_summary_rounds", 2)

        if buffer is None:
            buffer = await self._load_buffer(scope_id)
        user_msgs = sum(1 for i in buffer if i.get("role") == "user")

        # ---- 触发条件0：话题收尾语（优先级最高）----
        # 用户明确收尾时（"就这样/晚安/改天聊"），此刻是天然的话题边界，
        # 立即总结可以避免同一话题被后续的轮数/时间条件切到下一次总结里。
        if self._get_config("summary_topic_boundary_enabled", True) and user_msgs >= min_rounds:
            if self._has_closure_signal(buffer):
                logger.info(f"[长期记忆] 话题收尾语触发总结（{user_msgs} 轮）")
                return await self._summarize_buffer(scope_id, event, buffer)

        # ---- 触发条件1：轮数达标 ----
        if user_msgs >= trigger_rounds:
            # 话题仍在延续时短暂延后，等这个话题说完再总结（硬上限：+N 轮）
            if (self._get_config("summary_topic_boundary_enabled", True)
                    and user_msgs < trigger_rounds + _TOPIC_DEFER_MAX_ROUNDS
                    and await self._topic_still_continuing(buffer)):
                logger.info(
                    f"[长期记忆] 轮数 {user_msgs} 已达标，但语义上话题仍在延续，延后总结"
                    f"（最迟 {trigger_rounds + _TOPIC_DEFER_MAX_ROUNDS} 轮）"
                )
                return False
            logger.info(f"[长期记忆] 轮数 {user_msgs} 触发总结")
            return await self._summarize_buffer(scope_id, event, buffer)

        last = await self._load_last_summary_time(scope_id)

        # ---- 触发条件2：时间间隔达标 ----
        if last > 0 and buffer and (time.time() - last) >= interval and user_msgs >= min_rounds:
            logger.info(f"[长期记忆] 时间触发总结 ({int(time.time()-last)}s)")
            return await self._summarize_buffer(scope_id, event, buffer)

        # ---- 触发条件3：信息密度高（用户消息总字符数 > 600） ----
        if user_msgs >= 2:
            total_chars = sum(len(i.get("content", "")) for i in buffer if i.get("role") == "user")
            if total_chars >= 600:
                logger.info(f"[长期记忆] 信息密度触发总结 (用户消息 {total_chars} 字符)")
                return await self._summarize_buffer(scope_id, event, buffer)

        # ---- 触发条件4：对话时间跨度大（缓冲区首尾消息间隔 > 10 分钟） ----
        if user_msgs >= 3 and len(buffer) >= 4:
            timestamps = [i.get("timestamp", 0) for i in buffer if i.get("timestamp")]
            if len(timestamps) >= 2 and max(timestamps) - min(timestamps) >= 600:
                span = int(max(timestamps) - min(timestamps))
                logger.info(f"[长期记忆] 时间跨度触发总结 ({span}s)")
                return await self._summarize_buffer(scope_id, event, buffer)

        return False

    # ========== 旧数据重整（合并同话题碎片 / 拆分混合话题） ==========
    _REORG_BATCH = 12

    @staticmethod
    def _build_reorg_prompt(items: list) -> str:
        numbered = "\n".join(f"{i + 1}. {t[:300]}" for i, t in enumerate(items))
        return (
            "下面是同一个人的历史记忆条目。它们由早期版本生成，可能存在两类问题：\n"
            "  A. 不同的话题被混在同一条里；\n"
            "  B. 同一个话题被拆成了多条零散记录。\n"
            "请把它们重整为一份干净的记忆列表：\n"
            "1. 同一话题（同一人物 / 事件 / 领域）的零散条目 → 合并成一条完整表述；\n"
            "2. 一条里混了互不相关的话题 → 拆成多条；\n"
            "3. **必须保留原有的全部关键信息**（人名、时间、数字、偏好、约定），不得凭空新增或推测；\n"
            "4. 每条自包含，禁止出现「同上」「该话题」「上述」等指代；\n"
            "5. 若某条已被另一条完全覆盖，可省略重复部分。\n\n"
            "输出 JSON：\n"
            '{"memories": [{"summary": "该话题的完整信息", "topic": "话题标签(不超过8字)", "importance": 3}]}\n\n'
            "原始条目：\n" + numbered
        )

    async def _reorganize_memories(self, scope_id: str, job: dict = None, resume: bool = False) -> dict:
        """用 LLM 重整既有记忆：合并同话题碎片、拆分混合话题。

        逐批处理（每批 _REORG_BATCH 条），**开始前先备份原数据**以便回滚。
        resume=True 时只重新处理"上次失败的批次"（从 ltm_reorg_pending_<scope> 读取），
        且**只替换这些条目**，不会动其它记忆。

        返回 {"before": n, "after": m, "failed_batches": k, "ok": bool}。
        """
        def _set(**kw):
            if isinstance(job, dict):
                job.update(kw)

        memories = await self._load_memories(scope_id)
        pending_texts = []
        if resume:
            try:
                pending = await self.get_kv_data(self._KV_REORG_PENDING_PREFIX + scope_id, None)
                if isinstance(pending, dict) and isinstance(pending.get("texts"), list):
                    pending_texts = [t for t in pending["texts"] if isinstance(t, str) and t.strip()]
            except Exception:
                pending_texts = []
            if not pending_texts:
                _set(error="", message="没有待重试的批次（上次重整没有失败项）")
                return {"before": 0, "after": 0, "failed_batches": 0, "ok": False}
            wanted = set(pending_texts)
            targets = [m for m in memories
                       if (m.get("text") or "").strip() in wanted]
        else:
            targets = [m for m in memories if m.get("type") == "summary" and (m.get("text") or "").strip()]

        if not targets:
            _set(error="", message="没有可重整的条目")
            return {"before": 0, "after": 0, "failed_batches": 0, "ok": False}
        if not resume and len(targets) < 2:
            _set(error="", message="可重整的总结记忆不足 2 条")
            return {"before": len(targets), "after": len(targets), "failed_batches": 0, "ok": False}

        # 备份（可回滚）
        try:
            await self.put_kv_data(self._KV_MEMORIES_BACKUP_PREFIX + scope_id, {
                "timestamp": time.time(),
                "memories": memories,
            })
        except Exception as e:
            logger.warning(f"[长期记忆] 重整前备份失败，已中止: {e}")
            _set(error=f"备份失败：{e}")
            return {"before": len(targets), "after": len(targets), "ok": False}

        provider_id = await self._get_llm_provider_id(scope_id)
        if not provider_id:
            _set(error="无可用 LLM")
            return {"before": len(targets), "after": len(targets), "ok": False}

        timeout = self._get_config("llm_timeout_seconds", 60)
        new_items = []
        texts = [m["text"] for m in targets]
        batches = [texts[i:i + self._REORG_BATCH] for i in range(0, len(texts), self._REORG_BATCH)]
        ok_batches = 0
        failed_batches = 0
        failed_texts = []
        total_b = len(batches)
        _set(total=total_b, done=0, ok=0, failed=0)

        async def _call_batch(batch):
            resp = await asyncio.wait_for(
                self.context.llm_generate(
                    chat_provider_id=provider_id,
                    prompt=self._build_reorg_prompt(batch),
                ),
                # 单批最多要产出 12 条结构化记忆，原 60 秒常不够
                timeout=max(timeout, 120),
            )
            return self._parse_summary_items(resp.completion_text.strip())

        async def _process(batch, depth=0):
            """返回 (items, ok)。超时/异常时若还能拆半，就拆半重试一次再放弃。"""
            try:
                items = await _call_batch(batch)
                if items:
                    return items, True
                # 模型明确表示无内容 → 原样保留，避免丢信息
                return [{"summary": t, "importance": 3} for t in batch if t.strip()], False
            except Exception as e:
                # 注意：asyncio.TimeoutError 的 str() 是空串，只打 {e} 会得到空白日志
                logger.warning(
                    f"[长期记忆] 重整失败（{len(batch)} 条，"
                    f"{type(e).__name__}: {e or '无错误信息'}）"
                )
                if depth < 1 and len(batch) > 1:
                    mid = len(batch) // 2
                    left, ok1 = await _process(batch[:mid], depth + 1)
                    right, ok2 = await _process(batch[mid:], depth + 1)
                    logger.info(
                        f"[长期记忆] 已拆半重试：{len(batch)} 条 → "
                        f"{mid} + {len(batch) - mid} 条"
                    )
                    return left + right, (ok1 and ok2)
                # 记录失败批次的原文，供「重试失败批次」精确续跑
                failed_texts.extend(t for t in batch if t.strip())
                return [{"summary": t, "importance": 3} for t in batch if t.strip()], False

        for bi, batch in enumerate(batches):
            # 先更新进度再调用：单批 LLM 可能要 20~60 秒，
            # 不提前刷进度会让页面看起来像卡死在 0/N。
            _set(message=f"正在重整第 {bi + 1}/{total_b} 批（{len(batch)} 条）…")
            items, ok = await _process(batch)
            new_items.extend(items)
            if ok:
                ok_batches += 1
            else:
                failed_batches += 1
            _set(
                done=bi + 1, ok=ok_batches, failed=failed_batches,
                message=f"已完成 {bi + 1}/{total_b} 批（失败 {failed_batches}）",
            )

        # 失败批次落盘：resume=True 时只重跑这些文本
        try:
            if failed_texts:
                await self.put_kv_data(self._KV_REORG_PENDING_PREFIX + scope_id, {
                    "timestamp": time.time(),
                    "texts": failed_texts,
                })
            else:
                await self.delete_kv_data(self._KV_REORG_PENDING_PREFIX + scope_id)
        except Exception as e:
            logger.warning(f"[长期记忆] 保存重整失败批次失败: {e}")

        if not new_items:
            _set(error="重整未产出任何条目，已保留原数据")
            return {"before": len(targets), "after": len(targets),
                    "failed_batches": failed_batches, "ok": False}

        # 新条目生成向量（失败则标记待补嵌，可稍后一键补嵌）
        _set(message=f"正在为 {len(new_items)} 条新记忆生成向量…")
        vecs = await self._embed_texts([it["summary"] for it in new_items], max_retries=1)

        rebuilt = []
        now = time.time()
        for idx, it in enumerate(new_items):
            text = (it.get("summary") or "").strip()
            if not text:
                continue
            entry = {
                "id": str(uuid.uuid4()),
                "text": text,
                "type": "summary",
                "timestamp": now,
                "importance": it.get("importance", 3),
                "tags": it.get("tags", []) or [],
                "narrative_time": it.get("narrative_time", ""),
                "dialog_ids": [],
            }
            vec = vecs[idx] if idx < len(vecs) else None
            if vec:
                entry["embedding"] = vec
            else:
                entry["_embed_failed"] = True
            rebuilt.append(entry)

        # 只替换**本次参与重整的条目**：
        # 全量模式下 = 全部 summary（raw 保留）；续跑模式下 = 那几批失败项，
        # 其它已合并好的记忆原样保留（否则续跑会把它们全删掉）。
        target_ids = {m.get("id") for m in targets}
        kept = [m for m in memories if m.get("id") not in target_ids]
        merged = kept + rebuilt

        limit = self._get_config("memory_limit", 500)
        try:
            limit = max(1, int(limit))
        except (TypeError, ValueError):
            limit = 500
        if len(merged) > limit and self._get_config("memory_eviction_strategy", "importance") != "fifo":
            merged = self._select_retained(merged, limit)

        await self._save_memories(scope_id, merged)
        tail = f"，{failed_batches} 批失败（已记录，可重试）" if failed_batches else ""
        logger.info(
            f"[长期记忆] 记忆重整完成 ({scope_id}): {len(targets)} 条 → {len(rebuilt)} 条"
            + (f"（mode={'resume' if resume else 'full'}）" if resume else "")
            + tail
        )
        return {"before": len(targets), "after": len(rebuilt),
                "failed_batches": failed_batches, "ok": True}

    # ========== 话题边界判定 ==========
    @staticmethod
    def _has_backreference(items: list) -> bool:
        """本轮对话中是否出现"回溯/补充已结束话题"的措辞。

        只看用户消息——这类信号由用户发出，助手复述不算。
        """
        for it in items:
            if it.get("role") != "user":
                continue
            text = (it.get("content") or "").strip()
            if text and _BACKREF_RE.search(text):
                return True
        return False

    @staticmethod
    def _has_closure_signal(buffer: list) -> bool:
        """缓冲区最新一条用户消息是否为收尾语。"""
        for item in reversed(buffer):
            if item.get("role") == "user":
                text = (item.get("content") or "").strip()
                return bool(text) and bool(_CLOSURE_RE.search(text))
        return False

    async def _topic_still_continuing(self, buffer: list) -> bool:
        """判断最新一轮是否仍在延续同一话题（语义层面）。

        仅在"轮数已达标、即将总结"时调用，且一次批量嵌入（带 5 分钟缓存），
        因此不会给每轮对话增加开销。取不到向量时返回 False（按原逻辑照常总结）。
        """
        try:
            users = [(i.get("content") or "").strip() for i in buffer if i.get("role") == "user"]
            users = [u for u in users if u]
            if len(users) < 3:
                return False
            newest = users[-1]
            if _CLOSURE_RE.search(newest):
                return False  # 已收尾 → 立即总结
            prev = users[-4:-1]
            if not prev:
                return False
            vecs = await self._embed_texts([newest] + prev, max_retries=0)
            if not vecs or vecs[0] is None:
                return False
            base = vecs[0]
            others = [v for v in vecs[1:] if v is not None]
            if not others:
                return False
            sims = self._batch_dot_product(base, others)
            return bool(sims) and max(sims) >= _TOPIC_CONTINUATION_SIM
        except Exception as e:
            logger.debug(f"[长期记忆] 话题延续判定失败，按原逻辑总结: {e}")
            return False

    # ========== 角色状态变化检测 ==========
    async def _check_persona_drift(self, scope_id: str, event=None) -> bool:
        """LLM 对比现有人设与最近记忆，检测角色关键状态变化。
        发现变化时保存待确认建议，通知用户二次确认。"""
        if not self._get_config("persona_drift_enabled", False):
            return False

        # 已存在待确认变更时跳过
        pending = await self._load_pending_persona(scope_id)
        if pending.get("changes"):
            return False

        memories = await self._load_memories(scope_id)
        recent = [m["text"] for m in memories if m.get("type") == "summary" and m.get("text")]
        if len(recent) < 3:
            return False

        # 获取当前人设文本
        persona_desc = ""
        try:
            pm = self.context.persona_manager
            persona_list = pm.get_all_personas() if callable(getattr(pm, "get_all_personas", None)) else []
            if persona_list and len(persona_list) > 0:
                persona_desc = getattr(persona_list[0], "system_prompt", "")[:800] or ""
        except Exception:
            pass
        if not persona_desc:
            persona_desc = self._get_config("summary_prompt_template", "")[:500]

        umo = scope_id
        if event:
            try:
                umo = event.unified_msg_origin
            except Exception:
                pass
        provider_id = await self._get_llm_provider_id(umo)
        if not provider_id:
            return False

        prompt = (
            "你是角色状态检测器。对比「当前人设」与「最近记忆」，检测角色是否发生了关键状态变化。\n\n"
            f"当前人设：\n{persona_desc or '（无）'}\n\n"
            f"最近记忆：\n" + "\n".join(f"- {r}" for r in recent[-10:]) + "\n\n"
            "请以 JSON 格式输出。若检测到角色关键状态变化（身份/年龄/职业/所在地/关系等转变），"
            "输出变化说明和建议修改。若无变化返回空 changes。\n"
            '{"changes": [{"old": "原描述", "new": "新描述", "reason": "变化原因"}], "summary": "一句话总结"}\n\n'
            "规则：只报告关键状态变化，忽略临时情绪和单个事件；old 引用人设原文，new 写修改后表述"
        )

        timeout = self._get_config("llm_timeout_seconds", 60)
        try:
            resp = await asyncio.wait_for(
                self.context.llm_generate(chat_provider_id=provider_id, prompt=prompt),
                timeout=timeout,
            )
            text = resp.completion_text.strip()
            m = re.search(r'\{[^{}]*"changes"[^{}]*\}', text, re.DOTALL)
            if m:
                result = json.loads(m.group())
                changes = result.get("changes", [])
                if changes and isinstance(changes, list) and len(changes) > 0:
                    await self._save_pending_persona(scope_id, {
                        "changes": changes[:5],
                        "summary": str(result.get("summary", "")),
                        "timestamp": time.time(),
                    })
                    logger.info(f"[人设漂移] 检测到 {len(changes)} 处变化 ({scope_id})")
                    # 通知用户
                    if event:
                        change_previews = [
                            f"「{c.get('old', '（新增）')}」→「{c.get('new', '')}」"
                            for c in changes[:3]
                        ]
                        try:
                            await event.send(event.plain_result(
                                f"🔄 检测到角色状态可能发生变化（{result.get('summary', '')}）：\n"
                                + "\n".join(change_previews)
                                + "\n\n发送 /ltm_persona_confirm 确认更新人设"
                                + "\n发送 /ltm_persona_reject 拒绝"
                                + "\n发送 /ltm_persona_pending 查看详情"
                            ))
                        except Exception:
                            pass
                    return True
        except asyncio.TimeoutError:
            pass
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
        except Exception as e:
            logger.debug(f"[人设漂移] 检测异常: {e}")

        return False

    # ========== 用户画像更新 ==========
    async def _check_and_trigger_profile_update(self, scope_id, event=None):
        """按时间间隔触发画像增量更新。
        仅当距离上次更新超过配置间隔、且存在新增对话总结时才执行。
        """
        if not self._get_config("profile_enabled", False):
            return

        # ---- 失败退避（指数）----
        # 注意：不能靠 profile_last_update 来节流——它只在**成功**时写入，
        # 超时后时间戳不变，下一条消息会立刻重试，形成死循环。
        fail_state = getattr(self, "_profile_fail_state", None)
        if fail_state is None:
            self._profile_fail_state = {}
            fail_state = self._profile_fail_state
        fails, next_try = fail_state.get(scope_id, (0, 0.0))
        if next_try and time.time() < next_try:
            logger.debug(
                f"[长期记忆] 画像更新退避中（已失败 {fails} 次，"
                f"剩余 {int(next_try - time.time())}s）: {scope_id}"
            )
            return

        interval_hours = self._get_config("profile_update_interval_hours", 12)
        # interval_hours == 0 表示每次有新总结都更新（旧行为），跳过时间检查
        if interval_hours > 0:
            last_update = await self._load_profile_last_update(scope_id)
            if (time.time() - last_update) < interval_hours * 3600:
                return

        anchor = await self._load_profile_anchor(scope_id)
        memories = await self._load_memories(scope_id)

        new_summaries = [
            m for m in memories
            if m.get("type") == "summary"
            and m.get("text")
            and (m.get("timestamp", 0) > anchor)
        ]

        if not new_summaries:
            logger.info(
                f"[长期记忆] 画像更新跳过：无新增对话总结 "
                f"(scope={scope_id}, anchor={anchor:.0f})"
            )
            return

        # 修复: 新总结太少（< 3 条）时跳过画像更新，避免从极少数据生成低质量画像
        min_profile_summaries = self._get_config("min_profile_summaries", 3)
        if len(new_summaries) < min_profile_summaries:
            logger.info(
                f"[长期记忆] 画像更新跳过：仅 {len(new_summaries)} 条新总结 "
                f"(至少需要 {min_profile_summaries} 条, scope={scope_id})"
            )
            return

        logger.info(
            f"[长期记忆] 触发画像更新：{len(new_summaries)} 条新总结 "
            f"(scope={scope_id}, anchor={anchor:.0f})"
        )
        # 重试时压缩单条长度（200→120 字符），降低单次请求体积与超时概率
        per_item_chars = 200 if fails == 0 else 120
        ok = await self._update_profile(
            scope_id, event, new_summaries, per_item_chars=per_item_chars
        )
        if ok:
            fail_state.pop(scope_id, None)
        else:
            fails += 1
            delay = min(_PROFILE_BACKOFF_BASE * (2 ** (fails - 1)), _PROFILE_BACKOFF_MAX)
            fail_state[scope_id] = (fails, time.time() + delay)
            logger.warning(
                f"[长期记忆] 画像更新第 {fails} 次失败，{int(delay)}s 后再试 "
                f"(scope={scope_id})"
            )

    async def _update_profile(self, scope_id, event=None, new_summaries=None,
                              max_items: int = 20, per_item_chars: int = 200) -> bool:
        """更新画像。返回 True=成功（或无需更新），False=本次失败（调用方据此退避）。

        per_item_chars：每条总结喂给 LLM 的截断长度。失败重试时调用方会调小它，
        以降低单次请求体积与耗时——超时多因 prompt 太大 + Provider 忙，而不是内容问题。
        """
        try:
            if new_summaries is None:
                # 兼容手动 /画像 更新：使用全部 summary
                memories = await self._load_memories(scope_id)
                new_summaries = [m for m in memories if m.get("type") == "summary" and m.get("text")]
            if not new_summaries:
                return True

            summary_texts = [m["text"] for m in new_summaries]
            existing = await self._load_profile(scope_id)
            template = self._get_config(
                "profile_prompt_template",
                # 关键：必须"先保留再增补"，否则每次更新都会把画像重写成寥寥数条
                "根据以下记忆总结，在【现有画像】的基础上做**增补与修正**："
                "1) 先保留现有画像中仍然有效的条目（本次记忆没提到也不要删除）；"
                "2) 再补充新特征，重复的合并成一条；"
                "3) 只有被新记忆明确推翻时才修改旧条目。"
                "只记录实际用户的真实信息，忽略角色扮演设定或虚构内容。"
                "每行一条、以 '- ' 开头，每条不超过 40 字，总共不超过 25 条。\n"
                "现有画像：{现有画像}\n\n记忆总结：\n{记忆列表}"
            )
            # max_items 只影响喂给 LLM 的记忆条数（重建画像时用更大的窗口）
            items_window = summary_texts[-max(1, int(max_items)):]
            _cap = max(40, int(per_item_chars))
            prompt = template.replace("{现有画像}", existing or "无").replace(
                "{记忆列表}", "\n".join(f"- {s[:_cap]}" for s in items_window)
            )
            umo = scope_id
            if event:
                try:
                    umo = event.unified_msg_origin
                except Exception: pass
            provider_id = await self._get_llm_provider_id(umo)
            if not provider_id:
                return False
            timeout = self._get_config("llm_timeout_seconds", 60)
            resp = await asyncio.wait_for(
                self.context.llm_generate(chat_provider_id=provider_id, prompt=prompt),
                timeout=timeout
            )
            profile = resp.completion_text.strip().strip('"').strip("'")
            if profile:
                # 注意：这里**不截断**。profile_max_chars 只用于"注入时"限长，
                # 若在保存时就截断，画像会被永久性破坏（管理页看到的也是残缺内容）。
                await self._save_profile(scope_id, profile)
                # 保存锚点为本批次所涵盖的最大 summary 时间戳
                max_ts = max(m.get("timestamp", 0) for m in new_summaries)
                await self._save_profile_anchor(scope_id, max_ts)
                await self._save_profile_last_update(scope_id, time.time())
                lines = profile.count("\n") + 1
                logger.info(
                    f"[长期记忆] 画像已更新 ({scope_id}): "
                    f"{len(profile)} 字符 / {lines} 行 | anchor={max_ts:.0f}"
                    f" | {profile[:60]}..."
                )
                return True
            return False
        except asyncio.TimeoutError:
            logger.warning("[长期记忆] 画像更新超时（prompt {:.1f}k 字符）".format(len(prompt) / 1000.0))
            return False
        except Exception as e:
            logger.warning(f"[长期记忆] 画像更新失败: {e}")
            return False

    # 画像注入（提供给主流程调用）
    async def _inject_profile(self, event, req, scope_id: str = ""):
        if not self._get_config("profile_enabled", False):
            return
        if not scope_id:  # 调用方已算好时直接复用，避免会话模式下重复查询会话 ID
            scope_id = await self._get_scope_id(event)
        profile = await self._load_profile(scope_id)
        if not profile or not profile.strip():
            return
        max_chars = self._get_config("profile_max_chars", 1200)
        original_len = len(profile)
        if len(profile) > max_chars:
            profile = profile[:max_chars-3] + "..."
        pos = self._get_config("profile_inject_position", "extra_user_content")
        ctx = f"以下是该用户的画像信息：\n{profile}"
        preview = profile[:80] + ("..." if len(profile) > 80 else "")

        log_data = {
            "scope_id": scope_id,
            "profile_length": original_len,
            "truncated": original_len > max_chars,
            "inject_position": pos,
            "profile_preview": preview,
        }

        if pos == "system_prompt":
            req.system_prompt += "\n\n" + ctx
            logger.info(
                f"[长期记忆] 📋 画像注入 | scope={scope_id} | "
                f"长度={original_len}字符{' (已截断)' if original_len > max_chars else ''} | "
                f"位置={pos} | 预览: {preview}"
            )
        else:
            # 官方推荐：extra_user_content_parts + mark_as_temp（本轮有效、不持久化到会话历史）
            ok = self._inject_extra_content(req, "[系统注入的用户画像]\n" + ctx)
            if not ok:
                logger.debug("[长期记忆] extra_user_content 不可用，画像已降级为 system_prompt 注入")
            logger.info(
                f"[长期记忆] 📋 画像注入 | scope={scope_id} | "
                f"长度={original_len}字符{' (已截断)' if original_len > max_chars else ''} | "
                f"位置={pos} | 预览: {preview}"
            )