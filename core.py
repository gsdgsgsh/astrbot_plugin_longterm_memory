"""核心模块：工具函数、存储管理、向量检索。"""
import asyncio
import hashlib
import math
import os
import re
import time
import uuid
from typing import Optional

import numpy as np

from astrbot.api import logger

# 错误消息特征模式，用于过滤不应存入缓冲区的 AstrBot 内部错误
_ERROR_PATTERNS = [
    r"object has no attribute",
    r"Error occurred while processing",
    r"Traceback \(most recent call last\)",
    r"File \".*\.py\", line \d+",
    r"TypeError:",
    r"AttributeError:",
    r"NameError:",
    r"ValueError:",
    r"KeyError:",
    r"ImportError:",
    r"SyntaxError:",
]
_ERROR_RE = re.compile("|".join(_ERROR_PATTERNS))

# 检索结果短期缓存（近义查询命中，避免重复 embedding API 调用 / 排序）
_RETRIEVE_CACHE_TTL = 60.0
_RETRIEVE_CACHE_MAX = 128
# on_llm_request 中记忆检索的超时（秒）：超时则跳过本次注入，绝不阻塞对话
_RETRIEVE_TIMEOUT = 2.0
# Embedding 单次请求的最大文本条数（分片上限）
_EMBED_CHUNK = 16
# 缓冲区硬上限（条，约 100 轮对话），防止 LLM 持续故障时无限累积
_MAX_BUFFER_ENTRIES = 200
# 每类 scope 数据缓存的最大条目数（防长期运行内存增长，超出淘汰最旧）
_MAX_SCOPE_CACHE_SIZE = 200


def _looks_like_error(text: str) -> bool:
    """检测文本是否像 AstrBot 内部错误信息（Python traceback / 异常消息）。
    返回 True 表示应该跳过，不存入缓冲区/不纳入总结。
    """
    if not text or not isinstance(text, str):
        return False
    return bool(_ERROR_RE.search(text))




class CoreMixin:
    """提供配置读取、锁、缓存、嵌入/重排 Provider 获取、记忆存取、向量检索。"""

    # ========== 基础工具 ==========
    def _get_config(self, key: str, default=None):
        return self.config.get(key, default)

    @staticmethod
    def _cache_put(cache: dict, key: str, value, max_size: int = _MAX_SCOPE_CACHE_SIZE):
        """写入带上限的缓存；超出上限时淘汰最旧条目（dict 保持插入顺序）。
        防止长期运行下按 scope 累积的缓存无限增长。
        """
        cache[key] = value
        while len(cache) > max_size:
            try:
                cache.pop(next(iter(cache)), None)
            except StopIteration:
                break

    def _inject_extra_content(self, req, text: str) -> bool:
        """将本轮动态上下文注入 LLM 请求（extra_user_content_parts + mark_as_temp）。

        官方推荐：动态内容（记忆/画像/好感度等）应通过 extra_user_content_parts 追加，
        而不是每轮改写 system_prompt——后者会破坏模型服务端 prompt 缓存，
        显著增加请求成本与首 token 延迟。mark_as_temp 使内容只参与本轮推理、
        不持久化到会话历史（避免注入内容被再次总结的"记忆套娃"）。
        旧版 AstrBot 无法使用该机制时降级为追加 system_prompt（不丢内容）。
        返回 True 表示注入成功。
        """
        try:
            from astrbot.core.agent.message import TextPart
            part = TextPart(text=text)
            if not hasattr(part, "mark_as_temp"):
                raise TypeError("mark_as_temp 不可用（旧版 AstrBot），降级 system_prompt 注入")
            part.mark_as_temp()
            parts = getattr(req, "extra_user_content_parts", None)
            if not isinstance(parts, list):
                raise TypeError("extra_user_content_parts 不可用，降级 system_prompt 注入")
            parts.append(part)
            return True
        except Exception:
            req.system_prompt += "\n\n" + text
            return False

    async def _detect_context_window(self, event) -> int:
        """带缓存的上下文窗口检测。
        检测结果按 provider_id 缓存：否则每条消息都会重新探测并重复打印
        "检测到上下文窗口" 日志（刷屏），也浪费 Provider 查询。
        """
        provider_id = ""
        try:
            provider_id = await self.context.get_current_chat_provider_id(
                umo=event.unified_msg_origin
            ) or ""
        except Exception:
            provider_id = ""
        if not provider_id:
            provider_id = self._get_config("summary_fallback_provider_id", "") or ""
        if provider_id:
            cache = getattr(self, "_ctx_window_cache", None)
            if cache is None:
                self._ctx_window_cache = {}
                cache = self._ctx_window_cache
            if provider_id in cache:
                return cache[provider_id]
            val = await self._detect_context_window_uncached(event)
            cache[provider_id] = val
            return val
        return await self._detect_context_window_uncached(event)

    async def _detect_context_window_uncached(self, event) -> int:
        """自动检测当前会话所用模型的上下文窗口大小（tokens）。
        优先从 Provider 对象属性读取，其次根据 model_name 映射已知值，
        最后回退到用户配置的 context_window_tokens。
        返回 0 表示无法检测，应始终启用记忆检索。
        """
        try:
            umo = event.unified_msg_origin
            provider_id = await self.context.get_current_chat_provider_id(umo=umo)
            if not provider_id:
                provider_id = self._get_config("summary_fallback_provider_id", "")
            if not provider_id:
                return 0

            p = self._try_get_provider_by_id(provider_id)
            if not p:
                return 0

            # 1. 尝试直接读取 Provider 的上下文长度属性
            for attr in ("max_tokens", "context_length", "max_input_tokens",
                         "model_max_length", "max_context_length", "max_context_tokens"):
                val = getattr(p, attr, None)
                if val is not None:
                    try:
                        n = int(val)
                        if n >= 1024:
                            logger.info(f"[长期记忆] 从 Provider.{attr} 检测到上下文窗口: {n} tokens")
                            return n
                    except (ValueError, TypeError):
                        pass

            # 2. 根据 model_name 映射已知上下文窗口
            model = (getattr(p, "model_name", "") or getattr(p, "model", "") or
                     getattr(p, "deployment_name", "") or "").lower()
            if model:
                KNOWN_WINDOWS = {
                    # OpenAI
                    "gpt-3.5-turbo": 4096, "gpt-3.5": 4096,
                    "gpt-4-turbo": 128000, "gpt-4o": 128000, "gpt-4o-mini": 128000,
                    "gpt-4-32k": 32768, "gpt-4": 8192,
                    "o1": 200000, "o1-mini": 128000, "o3": 200000, "o3-mini": 200000,
                    "o4-mini": 200000,
                    # Anthropic
                    "claude-3-opus": 200000, "claude-3-sonnet": 200000,
                    "claude-3-haiku": 200000, "claude-3.5": 200000,
                    "claude-4": 200000,
                    # DeepSeek
                    "deepseek-chat": 65536, "deepseek-reasoner": 65536,
                    "deepseek-v3": 65536, "deepseek-r1": 65536,
                    # Qwen
                    "qwen-turbo": 8192, "qwen-plus": 32768, "qwen-max": 32768,
                    "qwen3": 32768, "qwen2.5": 32768,
                    # GLM
                    "glm-4": 128000, "glm-4-flash": 128000,
                    # Gemini
                    "gemini-2.0-flash": 1048576, "gemini-1.5-pro": 2097152,
                    "gemini-1.5-flash": 1048576,
                    # Moonshot
                    "moonshot-v1": 32768, "moonshot-v1-8k": 8192,
                    "moonshot-v1-32k": 32768, "moonshot-v1-128k": 128000,
                    # Yi
                    "yi-large": 32768,
                }
                for key, window in KNOWN_WINDOWS.items():
                    if key in model:
                        logger.info(
                            f"[长期记忆] 通过 model_name '{model}' 识别上下文窗口: {window} tokens"
                        )
                        return window

            # 3. 检查 provider.extra 或 provider.config 中是否有提示
            extra = getattr(p, "extra", None) or getattr(p, "config", None)
            if isinstance(extra, dict):
                for k in ("max_tokens", "context_length", "max_input_tokens"):
                    val = extra.get(k)
                    if val is not None:
                        try:
                            n = int(val)
                            if n >= 1024:
                                logger.info(f"[长期记忆] 从 Provider.extra 检测到上下文窗口: {n} tokens")
                                return n
                        except (ValueError, TypeError):
                            pass

        except Exception as e:
            logger.debug(f"[长期记忆] 检测上下文窗口异常: {e}")

        # 4. 回退到用户手动配置
        fallback = self._get_config("context_window_tokens", 0)
        if fallback > 0:
            logger.info(f"[长期记忆] 使用用户配置的上下文窗口: {fallback} tokens")
        return fallback

    async def _get_scope_id(self, event):
        """计算当前事件所属的记忆作用域。
        memory_scope:
          - user:         按 unified_msg_origin（用户）隔离（向后兼容）
          - group_user:   按 群 + 用户 隔离
          - user_bot:     按 用户 + 机器人 隔离
          - conversation: 按 AstrBot 会话（conversation cid）隔离——同一用户的不同
                          对话各自拥有独立的记忆/总结/画像/好感度
        """
        scope = self._get_config("memory_scope", "user")
        if scope == "group_user":
            group_id = event.message_obj.group_id if event.message_obj else ""
            user_id = (
                event.message_obj.sender.user_id
                if event.message_obj and event.message_obj.sender
                else ""
            )
            return f"group_user:{group_id}:{user_id}"
        if scope == "user_bot":
            # 防御：message_obj 可能为 None（部分适配器），self_id 取不到时退化为 umo 本身
            self_id = getattr(event.message_obj, "self_id", "") if event.message_obj else ""
            return f"{event.unified_msg_origin}:{self_id}"
        if scope == "conversation":
            # 按会话隔离：取当前 AstrBot 会话 cid 作为作用域后缀
            cid = ""
            try:
                conv_mgr = self.context.conversation_manager
                cid = await conv_mgr.get_curr_conversation_id(event.unified_msg_origin) or ""
            except Exception as e:
                logger.debug(f"[长期记忆] 获取会话 ID 失败，回退 user 作用域: {e}")
            if cid:
                return f"conversation:{event.unified_msg_origin}:{cid}"
            # 取不到会话 ID（异常/极旧版本）时回退到 user 作用域，避免数据丢失
            return event.unified_msg_origin
        return event.unified_msg_origin

    # ========== Provider 获取 ==========
    # 注意：AstrBot 4.2x 起 ProviderManager 已移除 get_using_providers() 与
    # get_using_provider_ids()（旧版插件依赖它们，现已失效）。当前正确的来源为：
    #   - Context.get_all_providers() / get_all_embedding_providers() 等官方方法
    #   - ProviderManager.provider_insts / embedding_provider_insts / rerank_provider_insts
    #   - ProviderManager.default_chat_provider_id / curr_provider_inst
    #   - ProviderManager.inst_map（最后兜底）
    def _provider_manager(self):
        return getattr(self.context, "provider_manager", None)

    @staticmethod
    def _provider_id_of(p) -> str:
        """从 Provider 实例上取 ID（兼容 meta() 形式）。"""
        for attr in ("id", "provider_id"):
            v = getattr(p, attr, None)
            if isinstance(v, str) and v:
                return v
        meta = getattr(p, "meta", None)
        if callable(meta):
            try:
                m = meta()
                v = getattr(m, "id", None)
                if isinstance(v, str) and v:
                    return v
            except Exception:
                pass
        return ""

    def _iter_instances(self, extra_attrs: tuple = ()) -> list:
        """按官方优先顺序收集 Provider 实例（去重），供能力嗅探使用。"""
        out, seen = [], set()

        def add(obj):
            if obj is None:
                return
            oid = id(obj)
            if oid in seen:
                return
            seen.add(oid)
            out.append(obj)

        # 1) Context 官方方法
        for m in ("get_all_providers", "get_all_embedding_providers"):
            try:
                fn = getattr(self.context, m, None)
                if callable(fn):
                    for p in (fn() or []):
                        add(p)
            except Exception:
                pass
        # 2) ProviderManager 实例列表 / 映射（官方属性）
        pm = self._provider_manager()
        if pm is not None:
            for attr in tuple(extra_attrs) + ("provider_insts", "inst_map"):
                try:
                    v = getattr(pm, attr, None)
                    if isinstance(v, dict):
                        for p in v.values():
                            add(p)
                    elif isinstance(v, (list, tuple)):
                        for p in v:
                            add(p)
                except Exception:
                    pass
        return out

    def _llm_provider_ids(self) -> list:
        """按优先级收集可用的聊天 Provider ID（官方来源优先）。"""
        ids, seen = [], set()

        def add(pid):
            pid = str(pid) if pid else ""
            if pid and pid not in seen:
                seen.add(pid)
                ids.append(pid)

        pm = self._provider_manager()
        if pm is not None:
            add(getattr(pm, "default_chat_provider_id", "") or "")
            cur = getattr(pm, "curr_provider_inst", None)
            if cur is not None:
                add(self._provider_id_of(cur))
            for p in (getattr(pm, "provider_insts", None) or []):
                add(self._provider_id_of(p))
            inst_map = getattr(pm, "inst_map", None)
            if isinstance(inst_map, dict):
                for pid, p in inst_map.items():
                    # 仅接受具备文本生成能力的实例，避免把 embedding/TTS 当成聊天模型
                    if (callable(getattr(p, "text_generate", None))
                            or callable(getattr(p, "text_chat", None))):
                        add(pid)
        return ids

    def _get_embedding_provider(self):
        if self._embedding_provider_loaded:
            return self._embedding_provider
        self._embedding_provider_loaded = True
        try:
            emb_id = self._get_config("embedding_provider_id", "")
            if emb_id:
                p = self._try_get_provider_by_id(emb_id)
                if p and callable(getattr(p, "get_embeddings", None)):
                    self._embedding_provider = p
                    return p
            # 官方来源：Context.get_all_embedding_providers() / pm.embedding_provider_insts
            for p in self._iter_instances(("embedding_provider_insts",)):
                if callable(getattr(p, "get_embeddings", None)):
                    self._embedding_provider = p
                    return p
            # 兜底：inst_map 扫描
            p = self._try_get_embedding_from_inst_map(emb_id)
            if p:
                self._embedding_provider = p
                return p
            return None
        except Exception as e:
            logger.warning(f"[长期记忆] 获取 Embedding Provider 出错: {e}")
            return None

    def _get_rerank_provider(self):
        if not self._get_config("enable_rerank", False):
            return None
        if self._rerank_provider_loaded:
            return self._rerank_provider
        self._rerank_provider_loaded = True
        try:
            rerank_id = self._get_config("rerank_provider_id", "")
            if rerank_id:
                p = self._try_get_provider_by_id(rerank_id)
                if p and callable(getattr(p, "rerank", None)):
                    self._rerank_provider = p
                    return p
            for p in self._iter_instances(("rerank_provider_insts",)):
                if callable(getattr(p, "rerank", None)):
                    self._rerank_provider = p
                    return p
            return None
        except Exception as e:
            logger.warning(f"[长期记忆] 获取 Rerank Provider 出错: {e}")
            return None

    async def _get_llm_provider_id(self, umo: str) -> Optional[str]:
        try:
            pid = await self.context.get_current_chat_provider_id(umo=umo)
            if pid:
                return pid
        except Exception:
            pass  # 未配置/未找到时抛 ProviderNotFoundError，继续走回退
        fallback = self._get_config("summary_fallback_provider_id", "")
        if fallback and self._try_get_provider_by_id(fallback):
            return fallback
        ids = self._llm_provider_ids()
        return ids[0] if ids else None

    async def _check_llm_availability(self) -> Optional[str]:
        """启动时检查是否有可用的 LLM Provider（不依赖 event/umo）。
        返回 provider_id 或 None。
        """
        fallback = self._get_config("summary_fallback_provider_id", "")
        if fallback and self._try_get_provider_by_id(fallback) is not None:
            return fallback
        ids = self._llm_provider_ids()
        return ids[0] if ids else None

    def _try_get_provider_by_id(self, provider_id: str):
        try:
            if hasattr(self.context, "get_provider_by_id"):
                return self.context.get_provider_by_id(provider_id)
        except Exception: pass
        return None

    def _try_get_embedding_from_inst_map(self, preferred_id=""):
        """最后兜底：直接在 ProviderManager.inst_map 中嗅探具备 get_embeddings 的实例。"""
        try:
            pm = self._provider_manager()
            inst_map = getattr(pm, "inst_map", None) if pm is not None else None
            if isinstance(inst_map, dict):
                if preferred_id and preferred_id in inst_map:
                    p = inst_map[preferred_id]
                    if callable(getattr(p, "get_embeddings", None)):
                        return p
                for pid, p in inst_map.items():
                    if callable(getattr(p, "get_embeddings", None)):
                        return p
        except Exception: pass
        return None

    # ========== 锁管理 ==========
    def _get_buffer_lock(self, scope_id: str) -> asyncio.Lock:
        if scope_id not in self._buffer_locks:
            self._buffer_locks[scope_id] = asyncio.Lock()
        return self._buffer_locks[scope_id]

    def _get_memory_lock(self, scope_id: str) -> asyncio.Lock:
        if scope_id not in self._memory_locks:
            self._memory_locks[scope_id] = asyncio.Lock()
        return self._memory_locks[scope_id]

    def _get_profile_lock(self, scope_id: str) -> asyncio.Lock:
        if scope_id not in self._profile_locks:
            self._profile_locks[scope_id] = asyncio.Lock()
        return self._profile_locks[scope_id]

    # ========== 记忆存取 ==========
    async def _load_memories_internal(self, scope_id: str) -> list:
        if scope_id in self._memory_cache:
            return list(self._memory_cache[scope_id])
        key = self._KV_MEMORIES_PREFIX + scope_id
        data = await self.get_kv_data(key, None)
        if data is None:
            self._cache_put(self._memory_cache, scope_id, [])
            return []
        if isinstance(data, dict) and "memories" in data:
            self._cache_put(self._memory_cache, scope_id, list(data["memories"]))
            await self._register_scope(scope_id)
            return list(data["memories"])
        if isinstance(data, list):
            self._cache_put(self._memory_cache, scope_id, list(data))
            await self._register_scope(scope_id)
            return list(data)
        self._cache_put(self._memory_cache, scope_id, [])
        return []

    async def _save_memories_internal(self, scope_id: str, memories: list):
        key = self._KV_MEMORIES_PREFIX + scope_id
        await self.put_kv_data(key, {"memories": memories})
        # 落盘即登记：避免"有数据但从未被读过"的作用域不出现在管理页下拉框
        await self._register_scope(scope_id)
        self._cache_put(self._memory_cache, scope_id, list(memories))
        # 记忆变更后使检索结果缓存失效（新增/删除/合并/补嵌都会经过这里）
        rc = getattr(self, "_retrieval_cache", None)
        if rc:
            rc.clear()

    async def _backup_memories_snapshot(self, scope_id: str, reason: str = "") -> int:
        """清空/迁移前把记忆快照到备份键，供「↩ 回滚到备份」恢复。

        返回快照条数（0 表示本来就没有记忆，无需备份）。
        背景：此前清空作用域（管理页清空 / 聊天里的 /new /del）是**不可恢复**的，
        一次误操作就永久丢数据。
        """
        try:
            mems = await self._load_memories(scope_id)
            if not mems:
                return 0
            await self.put_kv_data(self._KV_MEMORIES_BACKUP_PREFIX + scope_id, {
                "timestamp": time.time(),
                "memories": mems,
                "reason": reason or "clear",
            })
            logger.info(f"[长期记忆] 已备份 {len(mems)} 条记忆（{scope_id}，原因：{reason or 'clear'}）")
            return len(mems)
        except Exception as e:
            logger.warning(f"[长期记忆] 备份记忆失败: {e}")
            return 0

    def _select_retained(self, memories: list, limit: int) -> list:
        """挑选要保留的记忆，保持原有时间顺序。

        留存价值 = (0.55 × 重要性归一 + 0.45 × 新鲜度) × 类型权重
          · 重要性：(importance-1)/4 —— 姓名/联系方式这类高价值信息优先保留
          · 新鲜度：exp(-λ × 天数)，λ 复用 memory_decay_lambda
          · 类型：summary 权重 1.0，raw 原始记录 0.6（总结比原始流水更有留存价值）
        """
        if len(memories) <= limit:
            return list(memories)
        try:
            decay_lambda = float(self._get_config("memory_decay_lambda", 0.02))
        except (TypeError, ValueError):
            decay_lambda = 0.02
        now = time.time()
        scored = []
        for idx, m in enumerate(memories):
            try:
                imp = max(1, min(5, int(m.get("importance", 3))))
            except (TypeError, ValueError):
                imp = 3
            imp_norm = (imp - 1) / 4.0
            try:
                ts = float(m.get("timestamp") or now)
            except (TypeError, ValueError):
                ts = now
            age_days = max(0.0, (now - ts) / 86400.0)
            recency = math.exp(-decay_lambda * age_days)
            type_w = 1.0 if m.get("type") == "summary" else 0.6
            value = (0.55 * imp_norm + 0.45 * recency) * type_w
            scored.append((value, idx))
        # 价值高的优先保留；价值并列时保留较新的（idx 大）
        scored.sort(key=lambda x: (-x[0], -x[1]))
        keep_idx = {idx for _v, idx in scored[:limit]}
        return [m for i, m in enumerate(memories) if i in keep_idx]

    async def _load_memories(self, scope_id: str) -> list:
        async with self._get_memory_lock(scope_id):
            return await self._load_memories_internal(scope_id)

    async def _save_memories(self, scope_id: str, memories: list):
        async with self._get_memory_lock(scope_id):
            await self._save_memories_internal(scope_id, memories)

    # ========== 缓冲区存取 ==========
    async def _load_buffer_internal(self, scope_id: str) -> list:
        if scope_id in self._buffer_cache:
            return list(self._buffer_cache[scope_id])
        key = self._KV_BUFFER_PREFIX + scope_id
        data = await self.get_kv_data(key, None)
        if data is None or not isinstance(data, list):
            self._cache_put(self._buffer_cache, scope_id, [])
            return []
        self._cache_put(self._buffer_cache, scope_id, list(data))
        await self._register_scope(scope_id)
        return list(data)

    async def _save_buffer_internal(self, scope_id: str, buffer: list):
        # 缓冲区硬上限：LLM 持续故障时防止无限累积（丢弃最旧）
        if len(buffer) > _MAX_BUFFER_ENTRIES:
            buffer = buffer[-_MAX_BUFFER_ENTRIES:]
        # 只有缓冲区数据的新会话也要能被登记，否则管理页看不到它
        if buffer:
            await self._register_scope(scope_id)
        key = self._KV_BUFFER_PREFIX + scope_id
        await self.put_kv_data(key, buffer)
        self._cache_put(self._buffer_cache, scope_id, list(buffer))

    async def _load_buffer(self, scope_id: str) -> list:
        async with self._get_buffer_lock(scope_id):
            return await self._load_buffer_internal(scope_id)

    async def _save_buffer(self, scope_id: str, buffer: list):
        async with self._get_buffer_lock(scope_id):
            await self._save_buffer_internal(scope_id, buffer)

    # ========== 画像存取 ==========
    async def _load_profile_internal(self, scope_id: str) -> str:
        if scope_id in self._profile_cache:
            return self._profile_cache[scope_id]
        key = self._KV_PROFILE_PREFIX + scope_id
        data = await self.get_kv_data(key, "")
        if isinstance(data, str):
            self._cache_put(self._profile_cache, scope_id, data)
            return data
        self._cache_put(self._profile_cache, scope_id, "")
        return ""

    async def _save_profile_internal(self, scope_id: str, text: str):
        key = self._KV_PROFILE_PREFIX + scope_id
        await self.put_kv_data(key, text)
        self._cache_put(self._profile_cache, scope_id, text)

    async def _load_profile(self, scope_id: str) -> str:
        async with self._get_profile_lock(scope_id):
            return await self._load_profile_internal(scope_id)

    async def _save_profile(self, scope_id: str, text: str):
        async with self._get_profile_lock(scope_id):
            await self._save_profile_internal(scope_id, text)

    async def _load_last_summary_time(self, scope_id: str) -> float:
        key = self._KV_LAST_SUMMARY_PREFIX + scope_id
        val = await self.get_kv_data(key, 0.0)
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0

    async def _save_last_summary_time(self, scope_id: str, t: float):
        key = self._KV_LAST_SUMMARY_PREFIX + scope_id
        await self.put_kv_data(key, t)

    # ========== 画像间隔更新 ==========
    async def _load_profile_last_update(self, scope_id: str) -> float:
        key = self._KV_PROFILE_LAST_UPDATE_PREFIX + scope_id
        val = await self.get_kv_data(key, 0.0)
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0

    async def _save_profile_last_update(self, scope_id: str, t: float):
        key = self._KV_PROFILE_LAST_UPDATE_PREFIX + scope_id
        await self.put_kv_data(key, t)

    async def _load_profile_anchor(self, scope_id: str) -> float:
        key = self._KV_PROFILE_ANCHOR_PREFIX + scope_id
        val = await self.get_kv_data(key, 0.0)
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0

    async def _save_profile_anchor(self, scope_id: str, t: float):
        key = self._KV_PROFILE_ANCHOR_PREFIX + scope_id
        await self.put_kv_data(key, t)

    # ========== 待确认人设变更存取 ==========
    async def _load_pending_persona(self, scope_id: str) -> dict:
        key = self._KV_PENDING_PERSONA + scope_id
        data = await self.get_kv_data(key, None)
        if isinstance(data, dict):
            return data
        return {}

    async def _save_pending_persona(self, scope_id: str, data: dict):
        key = self._KV_PENDING_PERSONA + scope_id
        await self.put_kv_data(key, data)

    async def _clear_pending_persona(self, scope_id: str):
        await self.delete_kv_data(self._KV_PENDING_PERSONA + scope_id)


    # ========== 向量计算与检索 ==========
    @staticmethod
    def _normalize_vector(vec):
        """L2 归一化向量，使模长为 1。返回 list，零向量原样返回。"""
        try:
            arr = np.array(vec, dtype=np.float32)
            n = np.linalg.norm(arr)
            if n > 0:
                arr = arr / n
            return arr.tolist()
        except Exception:
            return vec

    @staticmethod
    def _batch_dot_product(query_vec, memory_vecs):
        """预归一化向量的批量点积（等价于余弦相似度）。所有向量必须已 L2 归一化。"""
        if not memory_vecs:
            return []
        try:
            q = np.array(query_vec, dtype=np.float32).reshape(1, -1)
            m = np.array(memory_vecs, dtype=np.float32)
            return (m @ q.T).flatten().tolist()
        except Exception:
            return [float(np.dot(np.array(query_vec, dtype=np.float32),
                                 np.array(mv, dtype=np.float32))) for mv in memory_vecs]

    @staticmethod
    def _cosine_similarity(a, b):
        """兜底：单对向量余弦相似度（无需预归一化）。"""
        try:
            a = np.array(a, dtype=np.float32)
            b = np.array(b, dtype=np.float32)
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0
        except Exception:
            return 0.0

    async def _embed_text(self, text: str, max_retries=2):
        """单条嵌入（内部走批量实现，命中缓存时不发请求）。"""
        res = await self._embed_texts([text], max_retries=max_retries)
        return res[0] if res else None

    async def _embed_texts(self, texts: list, max_retries: int = 2) -> list:
        """批量嵌入：命中缓存的不重复请求，未命中的合并为**一次** API 调用。

        返回与输入等长的列表，失败项为 None。
        同时记录最近一次失败原因（self._embed_last_error），供管理页诊断展示。
        """
        if not texts:
            return []
        cache = getattr(self, "_embed_cache", None)
        if cache is None:
            self._embed_cache = {}
            cache = self._embed_cache

        now = time.time()
        out = [None] * len(texts)
        miss_idx, miss_keys = [], []
        for i, t in enumerate(texts):
            key = (t or "").strip()
            if not key:
                continue
            hit = cache.get(key)
            if hit and now - hit[1] < 300:  # 5 分钟 TTL（命中即滑动续期）
                cache[key] = (hit[0], now)
                out[i] = hit[0]
            else:
                miss_idx.append(i)
                miss_keys.append(key)

        if not miss_idx:
            return out

        provider = self._get_embedding_provider()
        if not provider:
            self._embed_last_error = "未找到可用的 Embedding Provider（请检查 embedding_provider_id 或嵌入模型配置）"
            return out

        payload = [texts[i] for i in miss_idx]
        # 分片请求：多数 Embedding 服务对单次批量有上限（常见 32/64/128），
        # 一次性提交上百条很容易超限或被服务端截断，导致"数量不匹配"。
        failed_chunks = 0
        for start in range(0, len(payload), _EMBED_CHUNK):
            chunk = payload[start:start + _EMBED_CHUNK]
            got = None
            for attempt in range(max_retries + 1):
                try:
                    res = await provider.get_embeddings(list(chunk))
                    if res and len(res) == len(chunk):
                        got = res
                        break
                    self._embed_last_error = (
                        f"Provider 返回数量不匹配（收到 {len(res) if res else 0} / 请求 {len(chunk)}）"
                    )
                except Exception as e:
                    self._embed_last_error = f"{type(e).__name__}: {e}"
                    logger.warning(f"[长期记忆] Embedding 批次失败 (attempt {attempt+1}): {e}")
                    if attempt < max_retries:
                        await asyncio.sleep(1)
            if not got:
                failed_chunks += 1
                continue
            for j, vec in enumerate(got):
                idx = start + j
                v = CoreMixin._normalize_vector(vec)
                out[miss_idx[idx]] = v
                cache[miss_keys[idx]] = (v, time.time())

        if failed_chunks == 0:
            self._embed_last_error = ""
        if len(cache) > 256:  # 缓存淘汰：丢掉最旧的一半
            stale = sorted(cache.items(), key=lambda x: x[1][1])[:128]
            for k, _ in stale:
                cache.pop(k, None)
        return out

    async def _retrieve_memories(self, scope_id: str, query_text: str):
        """带短期缓存的记忆检索：近义查询直接命中缓存，避免重复 embedding API 调用与排序。
        记忆写入（_save_memories_internal）会使缓存失效。
        """
        top_k = self._get_config("retrieve_top_k", 3)
        threshold = self._get_config("similarity_threshold", 0.7)

        cache = getattr(self, "_retrieval_cache", None)
        if cache is None:
            self._retrieval_cache = {}
            cache = self._retrieval_cache
        cache_key = f"{scope_id}|{hashlib.sha1(query_text.strip().encode('utf-8')).hexdigest()}"
        hit = cache.get(cache_key)
        if hit and time.time() - hit[1] < _RETRIEVE_CACHE_TTL:
            return list(hit[0])

        result = await self._retrieve_memories_uncached(scope_id, query_text, top_k, threshold)

        # 缓存结果（含空结果，避免高频冷查询反复打 embedding API）
        cache[cache_key] = (result, time.time())
        while len(cache) > _RETRIEVE_CACHE_MAX:
            try:
                cache.pop(next(iter(cache)), None)
            except StopIteration:
                break
        return result

    async def _retrieve_memories_uncached(self, scope_id: str, query_text: str,
                                           top_k: int, threshold: float) -> list:
        # ---- ChromaDB 路径（优先） ----
        if self._get_config("chromadb_enabled", False):
            query_vec = await self._embed_text(query_text)
            if query_vec:
                # 多取候选（top_k 的数倍）以便后续去重/多样性挑选有选择空间
                fetch_k = max(top_k * 4, 12)
                chroma_results = await self._chroma_search(scope_id, query_vec, fetch_k, threshold)
                if chroma_results:
                    scored = []
                    for item in chroma_results:
                        scored.append(({
                            "id": item["id"], "text": item["text"], "type": item["type"],
                            "timestamp": item["timestamp"], "importance": item.get("importance", 3),
                            "tags": item.get("tags", []),
                            # 透传向量，使近重复剔除与 MMR 在 ChromaDB 路径同样生效
                            "embedding": item.get("embedding"), "dialog_ids": [],
                        }, item["similarity"]))
                    return self._rank_and_truncate(scored, top_k, threshold)
            logger.debug("[长期记忆] ChromaDB 未返回结果，回退 NumPy 检索")

        # ---- NumPy 路径（回退 / 默认） ----
        memories = await self._load_memories(scope_id)
        searchable = [m for m in memories if m.get("type") == "summary" and not m.get("_embed_failed")]
        if not searchable:
            return []

        query_vec = await self._embed_text(query_text)
        if not query_vec:
            return []
        query_vec = self._normalize_vector(query_vec)  # 确保查询向量已归一化（兼容旧数据）

        valid, valid_vecs = [], []
        for m in searchable:
            emb = m.get("embedding")
            if emb and isinstance(emb, (list, tuple)):
                valid.append(m)
                valid_vecs.append(emb)
        if not valid_vecs:
            return []

        dim = len(query_vec)
        filtered, filtered_vecs = [], []
        for m, v in zip(valid, valid_vecs):
            if len(v) == dim:
                filtered.append(m)
                filtered_vecs.append(v)
            else:
                if scope_id not in self._dim_check_triggered:
                    self._dim_check_triggered.add(scope_id)
                    asyncio.create_task(self._fix_dim_mismatch(scope_id, m, v, dim))

        if not filtered_vecs:
            return []

        sims = self._batch_dot_product(query_vec, filtered_vecs)
        scored = [(filtered[i], s) for i, s in enumerate(sims) if s >= threshold]

        # 重排（异步）
        rerank = self._get_rerank_provider()
        if rerank and len(scored) > 1:
            docs = [it[0]["text"] for it in scored[:top_k * 2]]
            try:
                rr = await rerank.rerank(query=query_text, documents=docs)
                if rr and hasattr(rr, "results"):
                    reranked = []
                    for r in rr.results:
                        idx = r.index if hasattr(r, "index") else r.get("index", 0)
                        if idx < len(scored):
                            reranked.append((scored[idx][0],
                                r.relevance_score if hasattr(r, "relevance_score")
                                else r.get("relevance_score", 0)))
                    reranked.sort(key=lambda x: x[1], reverse=True)
                    scored = reranked
            except Exception:
                pass

        return self._rank_and_truncate(scored, top_k, threshold)

    def _rank_and_truncate(self, scored: list, top_k: int, threshold: float) -> list:
        """排序管线：衰减 → 重要性加权 → 排序 → 截断。返回记忆对象列表。"""
        if not scored:
            return []

        # ---- 记忆衰减 ----
        if self._get_config("memory_decay_enabled", True):
            decay_lambda = self._get_config("memory_decay_lambda", 0.02)
            if decay_lambda > 0:
                now = time.time()
                decayed = []
                for mem, score in scored:
                    age_days = (now - mem.get("timestamp", now)) / 86400.0
                    decay = math.exp(-decay_lambda * max(age_days, 0))
                    adjusted = score * decay
                    decayed.append((mem, adjusted))
                scored = [(m, s) for m, s in decayed if s >= threshold]

        # ---- 重要性加权 ----
        if self._get_config("memory_importance_enabled", True):
            imp_weight = self._get_config("memory_importance_weight", 0.3)
            if imp_weight > 0:
                imp_scored = []
                for mem, score in scored:
                    imp = max(1, min(5, mem.get("importance", 3)))
                    imp_norm = (imp - 1) / 4.0
                    adjusted = score * (1 - imp_weight) + imp_norm * imp_weight
                    imp_scored.append((mem, adjusted))
                scored = imp_scored

        scored.sort(key=lambda x: x[1], reverse=True)

        # ---- 近重复过滤：同一件事被存成多条时，只保留最相关的一条 ----
        scored = self._drop_near_duplicates(scored)

        # ---- MMR 多样性选择：避免 top_k 全是同一话题的不同说法 ----
        lam = self._get_config("retrieve_diversity_lambda", 0.7)
        try:
            lam = float(lam)
        except (TypeError, ValueError):
            lam = 0.7
        lam = max(0.0, min(1.0, lam))
        if lam >= 1.0 or len(scored) <= 1:
            return [it[0] for it in scored[:top_k]]

        return self._mmr_select(scored, top_k, lam)

    def _drop_near_duplicates(self, scored: list) -> list:
        """丢掉与已保留项高度相似的候选（向量余弦 >=0.95 或文本 2-gram Jaccard >=0.9）。

        这类候选几乎总是「同一件事被重复总结」，占满 top_k 只会挤掉别的话题。
        """
        kept = []
        for mem, score in scored:
            dup = False
            for km, _ks in kept:
                if self._pair_similarity(mem, km) >= 0.95:
                    dup = True
                    break
            if not dup:
                kept.append((mem, score))
        return kept

    def _mmr_select(self, scored: list, top_k: int, lam: float) -> list:
        """最大边际相关（MMR）贪心选择。

        每轮挑选「lam × 相关性 - (1-lam) × 与已选项的最大相似度」最大的候选，
        在「相关」和「不重复」之间折中。lam=1 退化为纯相关性排序。
        """
        pool = list(scored)
        selected = []
        while pool and len(selected) < top_k:
            best_i, best_val = 0, None
            for i, (mem, score) in enumerate(pool):
                max_sim = 0.0
                for sel_mem in selected:
                    s = self._pair_similarity(mem, sel_mem)
                    if s > max_sim:
                        max_sim = s
                val = lam * score - (1.0 - lam) * max_sim
                if best_val is None or val > best_val:
                    best_val, best_i = val, i
            selected.append(pool.pop(best_i)[0])
        return selected

    @classmethod
    def _pair_similarity(cls, m1: dict, m2: dict) -> float:
        """两条记忆的相似度：优先用已 L2 归一化向量的点积，缺失时退回文本 2-gram Jaccard。"""
        v1, v2 = m1.get("embedding"), m2.get("embedding")
        if (isinstance(v1, (list, tuple)) and isinstance(v2, (list, tuple))
                and v1 and len(v1) == len(v2)):
            try:
                a = np.array(v1, dtype=np.float32)
                b = np.array(v2, dtype=np.float32)
                return max(0.0, min(1.0, float(a @ b)))
            except Exception:
                pass
        return cls._text_similarity(m1.get("text", ""), m2.get("text", ""))

    @staticmethod
    def _text_similarity(t1: str, t2: str) -> float:
        """字符 2-gram Jaccard 相似度（无需向量，ChromaDB 路径同样可用）。"""
        a = (t1 or "").strip()
        b = (t2 or "").strip()
        if not a or not b:
            return 0.0
        if a == b:
            return 1.0

        def grams(s: str) -> set:
            g = {s[i:i + 2] for i in range(len(s) - 1)}
            return g or {s}

        ga, gb = grams(a), grams(b)
        if not ga or not gb:
            return 0.0
        return len(ga & gb) / float(len(ga | gb))

    async def _fix_dim_mismatch(self, scope_id, mem, vec, expected_dim):
        try:
            async with self._get_memory_lock(scope_id):
                mems = await self._load_memories_internal(scope_id)
                for m in mems:
                    if m.get("id") == mem["id"] and not m.get("_embed_failed"):
                        m["_embed_failed"] = True
                        asyncio.create_task(self._retry_embed(scope_id, m["id"]))
                await self._save_memories_internal(scope_id, mems)
        except Exception:
            pass

    async def _add_memory(self, scope_id, text, mem_type, dialog_ids=None,
                          importance: int = 3, tags: list = None, narrative_time: str = ""):
        need_embed = mem_type != "raw"
        embedding = None
        failed = False
        if need_embed:
            embedding = await self._embed_text(text)
            if embedding is None:
                failed = True
        mem_id = str(uuid.uuid4())
        evicted_ids = []
        async with self._get_memory_lock(scope_id):
            mem = {
                "id": mem_id,
                "text": text,
                "type": mem_type,
                "timestamp": time.time(),
                "embedding": embedding,
                "dialog_ids": dialog_ids or [],
                "importance": importance,
                "tags": tags or [],
                "narrative_time": narrative_time or "",
            }
            if failed:
                mem["_embed_failed"] = True
                asyncio.create_task(self._retry_embed(scope_id, mem["id"]))
            memories = await self._load_memories_internal(scope_id)
            memories.append(mem)
            # 边界保护：0/负数会让切片语义失效（memories[:-0] 为空、memories[-0:] 为全量），
            # 导致淘汰逻辑静默失效，故强制至少保留 1 条
            try:
                limit = int(self._get_config("memory_limit", 500))
            except (TypeError, ValueError):
                limit = 500
            limit = max(1, limit)
            if len(memories) > limit:
                strategy = self._get_config("memory_eviction_strategy", "importance")
                if strategy == "fifo":
                    keep = memories[-limit:]
                else:
                    # 按"留存价值"（重要性 + 新鲜度 + 类型）淘汰：
                    # 纯 FIFO 会让一条姓名/生日这类关键记忆被无关的新记忆挤掉。
                    keep = self._select_retained(memories, limit)
                keep_ids = {id(m) for m in keep}
                evicted = [m for m in memories if id(m) not in keep_ids]
                memories = keep
                evicted_ids = [m.get("id") for m in evicted if m.get("id")]
                if evicted and strategy != "fifo":
                    dropped_important = sum(
                        1 for m in evicted if max(1, min(5, m.get("importance", 3))) >= 4
                    )
                    logger.info(
                        f"[长期记忆] 超出上限 {limit}，按留存价值淘汰 {len(evicted)} 条"
                        f"（其中高重要性 {dropped_important} 条）"
                    )
            await self._save_memories_internal(scope_id, memories)

        # 同步写入 ChromaDB（锁外执行）
        if embedding and not failed and self._get_config("chromadb_enabled", False):
            asyncio.create_task(
                self._chroma_add_memory(scope_id, mem_id, text, embedding,
                                        mem_type, importance, tags, time.time(),
                                        narrative_time)
            )
        # FIFO 淘汰的记忆同步从 ChromaDB 删除，避免幽灵记忆
        if evicted_ids and self._get_config("chromadb_enabled", False):
            asyncio.create_task(self._chroma_delete_memories(scope_id, evicted_ids))

    async def _retry_embed(self, scope_id, mem_id):
        async with self._embed_retry_semaphore:
            max_retries = 3
            for attempt in range(max_retries):
                await asyncio.sleep(5 * (attempt + 1))  # 5s, 10s, 15s 递增延迟
                meta = None
                async with self._get_memory_lock(scope_id):
                    mems = await self._load_memories_internal(scope_id)
                    target = next((m for m in mems if m["id"] == mem_id and m.get("_embed_failed")), None)
                    if not target:
                        return  # 已被其他任务修复或删除
                    text = target["text"]
                    meta = {
                        "type": target.get("type", "summary"),
                        "importance": target.get("importance", 3),
                        "tags": target.get("tags", []) or [],
                        "timestamp": target.get("timestamp", time.time()),
                        "narrative_time": target.get("narrative_time", ""),
                    }
                emb = await self._embed_text(text)
                if emb:
                    found = False
                    async with self._get_memory_lock(scope_id):
                        mems = await self._load_memories_internal(scope_id)
                        for m in mems:
                            if m["id"] == mem_id:
                                m["embedding"] = emb
                                m.pop("_embed_failed", None)
                                await self._save_memories_internal(scope_id, mems)
                                logger.info(f"[长期记忆] 后台补嵌成功 (id={mem_id[:8]}, attempt={attempt+1})")
                                found = True
                                break
                    # 同步更新 ChromaDB 中的 embedding
                    if found and self._get_config("chromadb_enabled", False) and meta:
                        await self._chroma_upsert_memory(
                            scope_id, mem_id, text, emb,
                            meta["type"], meta["importance"], meta["tags"],
                            meta["timestamp"], meta["narrative_time"],
                        )
                    return
                logger.debug(f"[长期记忆] 补嵌失败 (id={mem_id[:8]}, attempt={attempt+1}/{max_retries})")
            logger.warning(f"[长期记忆] 补嵌最终失败 (id={mem_id[:8]}, 已重试 {max_retries} 次)")