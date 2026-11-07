"""AstrBot 长期记忆插件主模块，整合所有功能。"""
import asyncio
import re
import time
import traceback
import uuid

from astrbot.api import logger, AstrBotConfig
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.provider import ProviderRequest, LLMResponse
from astrbot.api.star import Context, Star, register

from .core import CoreMixin, _looks_like_error, _RETRIEVE_TIMEOUT
from .engine import EngineMixin
from .web import WebMixin
from .affection import AffectionMixin
from .chroma_db import ChromaDBMixin


def _parse_plugin_meta(text: str) -> dict:
    """解析 metadata.yaml 文本里的 name/author/version（纯函数，便于测试）。"""
    meta = {"version": "0.0.0", "author": "", "name": "astrbot_plugin_longterm_memory"}
    for key in ("version", "author", "name"):
        m = re.search(rf"(?m)^{key}:\s*(.+?)\s*$", text or "")
        if not m:
            continue
        # 支持带引号的值（version: "1.0.0"），并忽略空值
        val = m.group(1).strip().strip('"').strip("'").strip()
        if val:
            meta[key] = val
    return meta


def _read_plugin_meta() -> dict:
    """从 metadata.yaml 读取 name/author/version（单一来源）。

    这三个值都是"元数据"，而且 author 还参与决定 AstrBot 的 KV 命名空间
    （plugin_id = author/name）。双轨发布（私有 2.9.x / 开源 1.0.0）时，
    若在代码里硬编码就会出现版本或作者对不上——所以统一从 metadata.yaml 读。

    读取失败或 author 为空时**不是静默降级**：会记录原因，并在启动日志里告警，
    因为 author 缺失会让 plugin_id 变成 "unknown/..."，数据会"看不见"。
    """
    try:
        from pathlib import Path
        text = (Path(__file__).resolve().parent / "metadata.yaml").read_text(encoding="utf-8")
    except Exception as e:
        meta = _parse_plugin_meta("")
        meta["_error"] = f"{type(e).__name__}: {e}"
        return meta
    meta = _parse_plugin_meta(text)
    if not meta["author"]:
        meta["_error"] = "metadata.yaml 缺少 author 字段"
    elif meta["version"] == "0.0.0":
        meta["_error"] = "metadata.yaml 缺少 version 字段"
    return meta


_PLUGIN_META = _read_plugin_meta()
_VERSION = _PLUGIN_META["version"]
_AUTHOR = _PLUGIN_META["author"] or "unknown"
_PLUGIN_NAME = _PLUGIN_META["name"] or "astrbot_plugin_longterm_memory"

if _PLUGIN_META.get("_error"):
    logger.warning(
        f"[长期记忆] metadata.yaml 读取不完整：{_PLUGIN_META['_error']}。"
        f"当前 author={_AUTHOR!r} version={_VERSION!r}——"
        "author 参与决定 KV 命名空间，缺失会导致已有记忆读不出来，请检查该文件。"
    )


@register(_PLUGIN_NAME, _AUTHOR, f"长期记忆插件 v{_VERSION}", _VERSION)
class LongTermMemory(CoreMixin, EngineMixin, WebMixin, AffectionMixin, ChromaDBMixin, Star):
    """长期记忆插件主类。"""

    # 本插件注册的全部命令（on_llm_request 中仅对本插件命令跳过记忆注入，
    # 其他 / 开头消息（第三方命令/普通聊天）继续正常流程）
    _LTM_COMMANDS = {
        "/new", "/del",
        "/ltm_view", "/ltm_delete", "/ltm_buffer", "/ltm_stats",
        "/ltm_summary", "/ltm_clear", "/ltm_profile", "/ltm_profile_update",
        "/ltm_profile_clear", "/ltm_affection",
        "/ltm_persona_pending", "/ltm_persona_confirm", "/ltm_persona_reject",
    }

    @staticmethod
    def _load_conf_schema() -> dict:
        """读取插件自带的 _conf_schema.json（用于取新默认值）。"""
        import json as _json
        from pathlib import Path
        path = Path(__file__).resolve().parent / "_conf_schema.json"
        return _json.loads(path.read_text(encoding="utf-8"))

    def _log_effective_params(self):
        """打印关键参数的**实际生效值**。

        存在的意义：配置里"没有这一项"和"有这一项且为某值"在行为上不同，
        而升级默认值是否生效光看启动日志是看不出来的——把这行打出来就能一眼核对。
        """
        try:
            items = [
                ("记忆上限", self._get_config("memory_limit", 500)),
                ("淘汰策略", self._get_config("memory_eviction_strategy", "importance")),
                ("隔离模式", self._get_config("memory_scope", "conversation")),
                ("检索多样性", self._get_config("retrieve_diversity_lambda", 0.7)),
                ("抽取条数", self._get_config("retrieve_top_k", 3)),
                ("画像开关", self._get_config("profile_enabled", False)),
                ("画像注入上限", self._get_config("profile_max_chars", 1200)),
                ("亲密度开关", self._get_config("affection_enabled", False)),
                ("每日上限", self._get_config("affection_daily_cap", 30.0)),
                ("增益系数", self._get_config("affection_ema_alpha", 0.4)),
                ("难度除数", self._get_config("affection_difficulty_divisor", 200.0)),
                ("评估冷却", f'{self._get_config("affection_eval_cooldown_seconds", 180)}s'),
            ]
            logger.info(
                "[长期记忆] 生效参数：" + "  ".join(f"{k}={v}" for k, v in items)
            )
        except Exception as e:
            logger.debug(f"[长期记忆] 打印生效参数失败: {e}")

    def _migrate_stale_defaults(self) -> list:
        """把"从未改过、仍是旧默认值"的配置项升级到新默认值。

        返回被迁移的 [(key, old, new), ...]；无变化时返回空列表。
        只改"当前值 == 旧默认值"的项，用户自定义过的值不会被动。
        """
        cfg = getattr(self, "config", None)
        if cfg is None:
            return []
        changed = []

        for key, old, new in self._STALE_DEFAULT_MIGRATIONS:
            cur = cfg.get(key, None)
            if cur is None:
                continue
            try:
                if abs(float(cur) - float(old)) < 1e-9:
                    cfg[key] = new
                    changed.append((key, cur, new))
            except (TypeError, ValueError):
                continue

        schema = {}
        try:
            schema = self._load_conf_schema()
        except Exception as e:
            logger.debug(f"[长期记忆] 读取 schema 失败，跳过提示词迁移: {e}")

        for key, marker in self._STALE_PROMPT_MARKERS:
            cur = cfg.get(key, None)
            if not isinstance(cur, str) or marker not in cur:
                continue
            new_val = (schema.get(key) or {}).get("default")
            if isinstance(new_val, str) and new_val and new_val != cur:
                cfg[key] = new_val
                changed.append((key, "旧版提示词", "新版提示词"))

        if not changed:
            logger.info("[长期记忆] 配置迁移检查：无需升级（配置中未发现旧默认值）")
            return []

        saver = getattr(cfg, "save_config", None)
        if callable(saver):
            try:
                saver()
            except Exception as e:
                logger.warning(f"[长期记忆] 迁移后的配置保存失败: {e}")
        else:
            logger.warning("[长期记忆] 配置对象没有 save_config，迁移仅在本次运行生效")

        for key, old, new in changed:
            logger.info(f"[长期记忆] 配置项已升级：{key}  {old} → {new}")
        logger.info(f"[长期记忆] 共升级 {len(changed)} 项旧默认配置")
        return changed

    @staticmethod
    def _is_own_command(text: str) -> bool:
        """判断消息是否为本插件命令（首词匹配，忽略大小写）。"""
        if not text or not text.strip().startswith("/"):
            return False
        return text.strip().split()[0].lower() in LongTermMemory._LTM_COMMANDS

    SIMPLE_LOGIN_HTML = r"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><title>Login</title>
<style>
    body { font-family: sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; background: #f8fafc; }
    .box { background: #fff; padding: 32px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); text-align: center; }
    input[type=password] { width: 260px; padding: 10px; border: 1px solid #ccc; border-radius: 6px; font-size: 14px; }
    button { padding: 10px 24px; border: none; background: #3b82f6; color: #fff; border-radius: 6px; cursor: pointer; font-size: 14px; margin-top: 12px; }
    .error { color: red; font-size: 12px; margin-top: 8px; }
</style>
</head>
<body>
<div class="box">
    <h2>长期记忆管理</h2>
    <p>请输入访问令牌</p>
    <input type="password" id="tokenInput" placeholder="Token">
    <br><button onclick="doLogin()">登录</button>
    <div class="error" id="errorMsg"></div>
</div>
<script>
async function doLogin() {
    const token = document.getElementById('tokenInput').value.trim();
    const err = document.getElementById('errorMsg');
    if (!token) { err.textContent = '令牌不能为空'; return; }
    try {
        const resp = await fetch('/api/login', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({token})
        });
        if (resp.status === 401) { err.textContent = '令牌错误'; return; }
        if (!resp.ok) { err.textContent = '登录失败'; return; }
        window.location.reload();
    } catch(e) {
        err.textContent = '网络错误';
    }
}
document.getElementById('tokenInput').addEventListener('keydown', e => { if(e.key==='Enter') doLogin(); });
</script>
</body>
</html>"""

    # ===== 旧 KV 命名空间迁移 =====
    # AstrBot 的插件 KV 按 plugin_id 分命名空间，而 plugin_id = f"{author}/{name}"（均小写）。
    # 一旦 metadata.yaml 的 author 变了，插件就会去读一个全新的空命名空间——
    # 表现为"记忆全没了"，但旧数据其实完好地留在旧命名空间里。
    # 这里在启动时把旧命名空间的数据**补搬**过来（只补当前缺失的键，绝不覆盖）。
    _LEGACY_PLUGIN_IDS = (
        "super z   deepseek/astrbot_plugin_longterm_memory",
        "super z deepseek/astrbot_plugin_longterm_memory",
        "super z/astrbot_plugin_longterm_memory",
    )

    # ===== 旧默认值自动迁移 =====
    # AstrBot 首次加载插件时会把 schema 默认值写进用户配置；此后**升级插件不会**
    # 用新默认值覆盖它们，于是旧默认值会一直生效（例如每日上限卡在 15）。
    # 下面这张表只迁移"当前值恰好等于旧默认值"的项——那说明用户从未改过它；
    # 用户手动改过的值（哪怕是别的数字）一律不动。
    _STALE_DEFAULT_MIGRATIONS = [
        ("affection_daily_cap", 15.0, 30.0),
        ("affection_ema_alpha", 0.15, 0.4),
        ("affection_difficulty_divisor", 120.0, 200.0),
        ("affection_eval_cooldown_seconds", 300, 180),
        ("profile_max_chars", 500, 1200),
    ]
    # 提示词类：只有内容里仍带"旧版特征串"时才替换（用户改写过就不会命中）
    _STALE_PROMPT_MARKERS = [
        ("affection_analysis_prompt_template", "用户略显不耐烦或敷衍"),
        ("summary_prompt_template", "输出一句话或简短的段落"),
        # 旧画像提示词会要求"不超过 10 条"且没要求保留旧内容，导致画像越更新越短
        ("profile_prompt_template", "总共不超过 10 条"),
        ("profile_prompt_template", "总数不超过 10 条"),
    ]

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config

        # 尽早执行，让后续读取配置的逻辑直接用上新默认值
        try:
            self._migrate_stale_defaults()
        except Exception as _e:
            logger.warning(f"[长期记忆] 配置迁移检查失败（不影响运行）: {_e}")

        self._KV_MEMORIES_PREFIX = "ltm_memories_"
        self._KV_BUFFER_PREFIX = "ltm_buffer_"
        self._KV_LAST_SUMMARY_PREFIX = "ltm_last_summary_"
        self._KV_CONFIRM_PREFIX = "ltm_confirm_"
        self._KV_ALL_SCOPES = "ltm_all_scopes"
        self._KV_PROFILE_PREFIX = "ltm_profile_"
        self._KV_PROFILE_LAST_UPDATE_PREFIX = "ltm_profile_last_update_"
        self._KV_PROFILE_ANCHOR_PREFIX = "ltm_profile_anchor_"
        self._KV_AFFECTION_PREFIX = "ltm_affection_"
        self._KV_AFFECTION_DAILY_PREFIX = "ltm_affection_daily_"
        # 命名空间迁移时要逐个带走的"按作用域"键前缀
        self._KV_MIGRATE_PREFIXES = (
            self._KV_MEMORIES_PREFIX, self._KV_BUFFER_PREFIX,
            self._KV_LAST_SUMMARY_PREFIX, self._KV_CONFIRM_PREFIX,
            self._KV_PROFILE_PREFIX, self._KV_PROFILE_LAST_UPDATE_PREFIX,
            self._KV_PROFILE_ANCHOR_PREFIX, self._KV_AFFECTION_PREFIX,
            self._KV_AFFECTION_DAILY_PREFIX, "ltm_memories_backup_",
        )
        self._KV_PENDING_PERSONA = "ltm_pending_persona_"
        # 记忆重整前的备份（用于回滚）
        self._KV_MEMORIES_BACKUP_PREFIX = "ltm_memories_backup_"
        # 记忆重整失败批次的原文（供「重试失败批次」精确续跑）
        self._KV_REORG_PENDING_PREFIX = "ltm_reorg_pending_"

        self._dim_check_triggered: set = set()
        self._memory_cache = {}
        self._buffer_cache = {}
        self._profile_cache = {}

        self._buffer_locks = {}
        self._memory_locks = {}
        self._profile_locks = {}

        self._embedding_provider = None
        self._embedding_provider_loaded = False
        self._rerank_provider = None
        self._rerank_provider_loaded = False

        self._embed_retry_semaphore = asyncio.Semaphore(5)
        self._retrieval_cache = {}
        # 最近一次嵌入失败原因（管理页「嵌入诊断」展示用）
        self._embed_last_error = ""
        # 批量补嵌任务状态（由 web.py 的 _job() 懒初始化，这里显式声明便于查阅）
        self._embed_job = None

        # 好感度系统
        self._affection_cache = {}
        self._affection_locks = {}

        # 注册 Dashboard 插件 Web API（管理界面入口，认证由 Dashboard 负责）
        try:
            self.register_web_apis()
        except Exception as e:
            logger.warning(f"[长期记忆] 注册插件 Web API 失败: {e}")

        logger.info(f"[长期记忆] 插件 v{_VERSION} 已加载")

        # 若当前已有运行中的事件循环（插件热重载场景），则直接调度健康检查；
        # 否则交给 on_astrbot_loaded 处理。
        self._schedule_startup_health_check()
        self._init_chromadb()

    def _schedule_startup_health_check(self):
        """安全调度启动健康检查任务。
        若当前无运行中的事件循环则静默跳过，由 on_astrbot_loaded 兜底。
        """
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._startup_health_check())
        except RuntimeError:
            pass

    async def _startup_health_check(self):
        """启动/重载时检查关键 Provider 连通性，输出清晰日志。"""
        # 给 Provider 管理器初始化时间
        await asyncio.sleep(1.5)

        # 1. Embedding Provider
        emb_provider = self._get_embedding_provider()
        if emb_provider:
            emb_name = getattr(emb_provider, "id", None) or getattr(emb_provider, "provider_id", None) or ""
            desc = f": {emb_name}" if emb_name else ""
            logger.info(f"[长期记忆] ✅ Embedding Provider 已就绪{desc}")
        else:
            logger.warning(
                "[长期记忆] ❌ Embedding Provider 未找到！向量检索将不可用，"
                "新记忆无法生成嵌入向量。请检查 embedding_provider_id 配置或确保已配置支持 embedding 的 Provider。"
            )

        # 2. LLM Provider（总结 / 关键词提取 / 画像 / 日记等共用）
        # 带重试：Provider 管理器可能尚未完成初始化
        llm_id = None
        for retry in range(5):
            llm_id = await self._check_llm_availability()
            if llm_id:
                break
            await asyncio.sleep(1.5)
        if llm_id:
            # 尝试获取更详细的信息
            p = self._try_get_provider_by_id(llm_id)
            model = ""
            if p:
                model = getattr(p, "model_name", "") or getattr(p, "model", "")
            desc = f"{llm_id}" + (f" (模型: {model})" if model else "")
            logger.info(f"[长期记忆] ✅ LLM Provider（总结）已就绪: {desc}")
        else:
            logger.warning(
                "[长期记忆] ❌ LLM Provider（总结）未找到！"
                "对话总结、画像更新、好感度 LLM 评估等功能将不可用。"
                "请检查 summary_fallback_provider_id 配置或确保已配置至少一个支持 text_generate 的 Provider。"
            )

        # 3. Rerank Provider（可选）
        if self._get_config("enable_rerank", False):
            rerank = self._get_rerank_provider()
            if rerank:
                rerank_name = getattr(rerank, "id", None) or getattr(rerank, "provider_id", None) or ""
                desc = f": {rerank_name}" if rerank_name else ""
                logger.info(f"[长期记忆] ✅ Rerank Provider 已就绪{desc}")
            else:
                logger.warning(
                    "[长期记忆] ⚠️ Rerank 已启用但未找到 Rerank Provider，"
                    "检索将跳过重排步骤。请检查 rerank_provider_id 配置。"
                )

        # 重置 Provider 缓存：健康检查可能过早运行导致缓存了 None，
        # 实际消息处理时需要重新查找（届时 Provider 管理器已完全就绪）
        self._embedding_provider_loaded = False
        self._embedding_provider = None
        self._rerank_provider_loaded = False
        self._rerank_provider = None

    @filter.on_astrbot_loaded()
    async def on_astrbot_loaded(self):
        # 首要任务：把旧 plugin_id 命名空间的数据补搬过来（author 变更会导致命名空间漂移）
        try:
            await self._migrate_legacy_kv_namespace()
        except Exception as _e:
            logger.warning(f"[长期记忆] 旧命名空间迁移失败（不影响运行）: {_e}")
        # 异步调度健康检查（避免阻塞 Provider 初始化）
        asyncio.create_task(self._startup_health_check())
        logger.info(
            f"[长期记忆] 插件 v{_VERSION} 已就绪（管理页：Dashboard → 插件 → 长期记忆）"
            f"｜KV 命名空间：{getattr(self, 'plugin_id', '?')}"
        )
        self._log_effective_params()

    async def _get_scope_registry_lock(self) -> asyncio.Lock:
        """作用域注册表读写锁（懒创建，避免在 __init__ 无事件循环时创建）。"""
        lock = getattr(self, "_scope_registry_lock", None)
        if lock is None:
            self._scope_registry_lock = asyncio.Lock()
            lock = self._scope_registry_lock
        return lock

    async def _migrate_legacy_kv_namespace(self) -> int:
        """把旧 plugin_id 命名空间下的数据补搬到当前命名空间（只补缺、不覆盖）。

        返回搬迁的条目数。取不到 AstrBot 内部 sp 模块时静默跳过（不影响运行）。
        """
        try:
            from astrbot.core import sp as _sp
        except Exception:
            return 0
        cur = str(getattr(self, "plugin_id", "") or "")
        if not cur:
            return 0
        moved = 0
        for legacy in self._LEGACY_PLUGIN_IDS:
            if not legacy or legacy == cur:
                continue
            try:
                old_scopes = await _sp.get_async("plugin", legacy, self._KV_ALL_SCOPES, None)
            except Exception:
                continue
            if not isinstance(old_scopes, list) or not old_scopes:
                continue
            try:
                new_scopes = await _sp.get_async("plugin", cur, self._KV_ALL_SCOPES, [])
            except Exception:
                new_scopes = []
            new_scopes = new_scopes if isinstance(new_scopes, list) else []
            for sid in old_scopes:
                if not isinstance(sid, str) or not sid:
                    continue
                for prefix in self._KV_MIGRATE_PREFIXES:
                    key = prefix + sid
                    try:
                        old_val = await _sp.get_async("plugin", legacy, key, None)
                        if old_val is None:
                            continue
                        cur_val = await _sp.get_async("plugin", cur, key, None)
                        if cur_val is not None:
                            continue  # 当前已有数据 → 不覆盖
                        await _sp.put_async("plugin", cur, key, old_val)
                        moved += 1
                    except Exception:
                        continue
                if sid not in new_scopes:
                    new_scopes.append(sid)
            try:
                await _sp.put_async("plugin", cur, self._KV_ALL_SCOPES, new_scopes)
            except Exception:
                pass
            logger.warning(
                f"[长期记忆] 检测到旧 KV 命名空间（{legacy}），已补搬 {moved} 项到 {cur}。"
                "原数据未被删除；数据已完整则无需处理。"
            )
        if moved:
            for cache in (getattr(self, "_memory_cache", None),
                          getattr(self, "_buffer_cache", None),
                          getattr(self, "_profile_cache", None)):
                if isinstance(cache, dict):
                    cache.clear()
            self._known_scopes = set()
        return moved

    async def _register_scope(self, scope_id: str):
        """把作用域登记到全局列表（管理页下拉框的数据来源）。

        进程内用 _known_scopes 去重：写缓冲区/记忆是高频路径，
        没有这层缓存时每次落盘都会触发一次 KV 读 + 写，代价过高。
        """
        if not scope_id:
            return
        known = getattr(self, "_known_scopes", None)
        if known is None:
            known = set()
            self._known_scopes = known
        if scope_id in known:
            return
        known.add(scope_id)
        lock = await self._get_scope_registry_lock()
        async with lock:
            try:
                scopes = await self.get_kv_data(self._KV_ALL_SCOPES, [])
                if isinstance(scopes, list) and scope_id not in scopes:
                    scopes.append(scope_id)
                    await self.put_kv_data(self._KV_ALL_SCOPES, scopes)
            except Exception as e:
                known.discard(scope_id)  # 失败时允许下次重试
                logger.debug(f"注册作用域出错: {e}")

    async def _unregister_scope(self, scope_id: str):
        lock = await self._get_scope_registry_lock()
        async with lock:
            try:
                scopes = await self.get_kv_data(self._KV_ALL_SCOPES, [])
                if isinstance(scopes, list) and scope_id in scopes:
                    scopes.remove(scope_id)
                    await self.put_kv_data(self._KV_ALL_SCOPES, scopes)
            except Exception as e:
                logger.debug(f"注销作用域出错: {e}")

    async def _get_all_scopes(self):
        try:
            return [s for s in await self.get_kv_data(self._KV_ALL_SCOPES, []) if isinstance(s, str)]
        except Exception:
            return []

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        try:
            scope_id = await self._get_scope_id(event)
            query = event.message_str
            if not query or not query.strip():
                return

            if query.strip().startswith("/"):
                # /new /del：清除当前作用域全部数据
                stripped = query.strip()
                first = stripped.split()[0].lower()
                if first == "/new":
                    await self._clear_scope_for_command(event, "new")
                    return
                if first == "/del":
                    await self._clear_scope_for_command(event, "del")
                    return
                if first in self._LTM_COMMANDS:
                    # 本插件命令：不做记忆注入，但保留 system_prompt 和已有 contexts
                    return
                # 其他 / 开头消息（第三方命令/普通聊天）继续正常流程

            # ---- 上下文窗口感知：对话较短时跳过记忆检索，减少延迟 ----
            ctx_window = await self._detect_context_window(event)
            skip_retrieval = False
            if ctx_window > 0:
                try:
                    conv_mgr = self.context.conversation_manager
                    curr_cid = await conv_mgr.get_curr_conversation_id(event.unified_msg_origin)
                    if curr_cid:
                        conv = await conv_mgr.get_conversation(event.unified_msg_origin, curr_cid)
                        if conv and hasattr(conv, "messages") and isinstance(conv.messages, list):
                            # 粗略估算 token 数：中文约 1.5 字符/token，英文约 4 字符/token。
                            # 统一按 2 字符/token 保守估算（略高估，提前触发检索，更安全）
                            total_chars = sum(
                                len(m.get("content", "") or "") + len(m.get("text", "") or "")
                                for m in conv.messages if isinstance(m, dict)
                            )
                            est_tokens = max(1, int(total_chars / 2))
                            if est_tokens < ctx_window * 0.75:
                                skip_retrieval = True
                                logger.debug(
                                    f"[长期记忆] 估算对话 token {est_tokens}/{ctx_window} (<75%)，跳过记忆检索"
                                )
                except Exception:
                    pass  # 获取会话失败时回退到正常流程

            # 旁路注入（始终执行，开销可忽略：纯 KV 读取 + 字符串拼接）
            # 顺便统计本轮注入体积：mark_as_temp 的内容不落历史，但每次请求仍要付这部分 token
            _inj = {"profile": 0, "affection": 0, "memory": 0}
            _c0 = self._request_injected_chars(req)
            await self._inject_profile(event, req, scope_id)
            _c1 = self._request_injected_chars(req)
            await self._inject_affection(event, req, scope_id)
            _c2 = self._request_injected_chars(req)
            _inj["profile"] = max(0, _c1 - _c0)
            _inj["affection"] = max(0, _c2 - _c1)

            if skip_retrieval:
                self._log_injection_cost(scope_id, _inj, skip_retrieval=True)
                return

            # ---- 记忆检索与注入（仅在上下文足够长时执行） ----
            # 带超时：检索（含 embedding API 调用）超时则跳过注入，绝不阻塞对话
            try:
                memories = await asyncio.wait_for(
                    self._retrieve_memories(scope_id, query), timeout=_RETRIEVE_TIMEOUT
                )
            except asyncio.TimeoutError:
                logger.debug(f"[长期记忆] 记忆检索超时（{_RETRIEVE_TIMEOUT}s），跳过本次注入")
                memories = []

            if not memories:
                logger.info(f"[长期记忆] 未检索到相关记忆 (scope={scope_id})")
                self._log_injection_cost(scope_id, _inj)
                return

            ctx = self._build_memory_context(memories)
            pos = self._get_config("memory_inject_position", "extra_user_content")
            previews = [f"{i+1}. {m['text'][:30]}..." for i, m in enumerate(memories)]
            logger.info(f"[长期记忆] 召回 {len(memories)} 条记忆，注入位置={pos}，预览: {'; '.join(previews)}")

            if pos == "system_prompt":
                req.system_prompt += "\n\n" + ctx
            else:
                # 官方推荐：extra_user_content_parts + mark_as_temp
                # （本轮有效、不持久化到会话历史，避免记忆内容被再次总结的"记忆套娃"）
                ok = self._inject_extra_content(req, "[系统注入的历史记忆上下文]\n" + ctx)
                if not ok:
                    logger.debug("[长期记忆] extra_user_content 不可用，记忆已降级为 system_prompt 注入")
            _inj["memory"] = len(ctx)
            self._log_injection_cost(scope_id, _inj)
        except Exception as e:
            logger.warning(f"注入记忆出错: {traceback.format_exc()}")

    # ========== 注入开销统计 ==========
    @staticmethod
    def _request_injected_chars(req) -> int:
        """统计「由本插件注入」的字符数：extra_user_content_parts + system_prompt 兜底。

        只用于**求差**（注入前后各测一次），因此 system_prompt 里原有的
        persona 等内容会被差值消掉，不会计入插件开销。
        """
        total = len(getattr(req, "system_prompt", "") or "")
        parts = getattr(req, "extra_user_content_parts", None) or []
        for p in parts:
            total += len(getattr(p, "text", "") or "")
        return total

    def _log_injection_cost(self, scope_id: str, inj: dict, skip_retrieval: bool = False):
        """打印本轮注入体积。

        只统计插件注入的部分（画像/好感度/记忆）。主对话历史由 AstrBot 决定，
        不在统计内——那通常才是 token 的大头。
        """
        total = sum(int(v or 0) for v in inj.values())
        if total <= 0:
            return
        logger.info(
            f"[长期记忆] 📊 本轮注入 {total} 字符"
            f"（画像 {inj.get('profile', 0)} / 好感度 {inj.get('affection', 0)}"
            f" / 记忆 {inj.get('memory', 0)}）≈{int(total * 0.7)} tokens"
            + (" | 对话较短，跳过记忆检索" if skip_retrieval else "")
        )

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse):
        try:
            scope_id = await self._get_scope_id(event)
            user = event.message_str
            # 移除 DeepSeek 等模型的 thinking 标签，防止思考内容泄露给用户
            resp.completion_text = strip_thinking(resp.completion_text)
            assistant = resp.completion_text

            # 防御: 用户消息以 / 开头（指令）或 LLM 返回空响应 → 跳过
            if not user or not assistant or user.strip().startswith("/"):
                logger.debug(f"[长期记忆] on_llm_response 跳过: user={'/cmd' if user and user.strip().startswith('/') else user}")
                return

            # 过滤 AstrBot 内部错误消息，防止污染对话缓冲/记忆总结
            if _looks_like_error(assistant):
                logger.info(f"[长期记忆] 检测到错误响应，跳过缓冲: {assistant[:100]}")
                return

            mid = ""
            if event.message_obj:
                mid = getattr(event.message_obj, "message_id", "") or ""

            # 缓存配置读取，避免重复 dict.get
            profile_enabled = self._get_config("profile_enabled", False)
            affection_enabled = self._get_config("affection_enabled", False)

            # ---- 缓冲区写入（锁内只做 IO，不做 LLM 调用） ----
            is_duplicate = False
            async with self._get_buffer_lock(scope_id):
                buf = await self._load_buffer_internal(scope_id)
                is_duplicate = mid and any(i.get("message_id") == mid for i in buf)

                if not is_duplicate:
                    buf.append({"role": "user", "content": user.strip(), "message_id": mid,
                                "timestamp": time.time(), "buf_id": str(uuid.uuid4())})
                    buf.append({"role": "assistant", "content": assistant.strip(), "message_id": "",
                                "timestamp": time.time(), "buf_id": str(uuid.uuid4())})
                    await self._save_buffer_internal(scope_id, buf)

            # ---- 异步触发周期性任务（不阻塞消息响应） ----
            if is_duplicate:
                logger.debug(f"[长期记忆] 跳过重复消息的定时任务触发")

            asyncio.create_task(self._check_and_trigger_summary(scope_id, event))
            if profile_enabled:
                asyncio.create_task(self._check_and_trigger_profile_update(scope_id, event))
            if affection_enabled:
                dialog_items = [
                    {"role": "user", "content": user.strip()},
                    {"role": "assistant", "content": assistant.strip()},
                ]
                asyncio.create_task(self._check_and_trigger_affection_eval(scope_id, dialog_items, event))

        except Exception as e:
            logger.warning(f"缓冲对话出错: {traceback.format_exc()}")

    async def _clear_scope_for_command(self, event, cmd_name: str):
        """响应 /new /del：清除记忆、画像、好感度、缓冲区并重置时间戳。"""
        try:
            sid = await self._get_scope_id(event)
            # 清空前自动快照：/new /del 是聊天里的命令，误触发代价很高
            backup_n = await self._backup_memories_snapshot(sid, f"命令 {cmd_name}")
            async with self._get_buffer_lock(sid):
                await self._save_buffer_internal(sid, [])
            async with self._get_memory_lock(sid):
                await self._save_memories_internal(sid, [])
            async with self._get_profile_lock(sid):
                await self._save_profile_internal(sid, "")
            if hasattr(self, "_affection_cache"):
                self._affection_cache.pop(sid, None)
            await self.delete_kv_data(self._KV_AFFECTION_PREFIX + sid)
            await self.delete_kv_data(self._KV_AFFECTION_DAILY_PREFIX + sid)
            await self.delete_kv_data(self._KV_CONFIRM_PREFIX + sid)
            await self.delete_kv_data(self._KV_PENDING_PERSONA + sid)
            await self._save_last_summary_time(sid, 0.0)
            await self._save_profile_last_update(sid, 0.0)
            await self._save_profile_anchor(sid, 0.0)
            self._memory_cache.pop(sid, None)
            self._buffer_cache.pop(sid, None)
            self._profile_cache.pop(sid, None)
            rc = getattr(self, "_retrieval_cache", None)
            if rc:
                rc.clear()
            # 清理按 scope 持有的锁，防止长期运行内存增长
            for pool in (self._buffer_locks, self._memory_locks,
                         self._profile_locks, getattr(self, "_affection_locks", {})):
                if isinstance(pool, dict):
                    pool.pop(sid, None)
            self._dim_check_triggered.discard(sid)
            if self._get_config("chromadb_enabled", False):
                await self._chroma_delete_scope(sid)
            await self._unregister_scope(sid)
            logger.info(f"[长期记忆] /{cmd_name}：已清除 {sid} 的记忆、画像、好感度和缓冲区")
        except Exception as e:
            logger.warning(f"[长期记忆] /{cmd_name} 清除失败: {e}")

    @filter.command("ltm_buffer")
    async def ltm_buffer(self, event):
        """查看当前对话缓冲区（待总结的原始消息）"""
        sid = await self._get_scope_id(event)
        buf = await self._load_buffer(sid)
        if not buf:
            yield event.plain_result("缓冲区为空，暂无待总结的对话。"); return
        lines = [f"缓冲区共 {len(buf)} 条消息："]
        for i, item in enumerate(buf[-20:], 1):
            role = item.get("role", "?")
            text = item.get("content", "")
            emoji = "👤" if role == "user" else "🤖"
            preview = text[:80] + ("..." if len(text) > 80 else "")
            lines.append(f"{i}. {emoji} [{role}] {preview}")
        if len(buf) > 20:
            lines.append(f"... 及更早 {len(buf) - 20} 条消息")
        yield event.plain_result("\n".join(lines))

    @filter.command("ltm_view")
    async def ltm_view(self, event):
        """查看最近记忆"""
        try:
            sid = await self._get_scope_id(event)
            mems = await self._load_memories(sid)
            if not mems:
                yield event.plain_result("当前没有任何记忆记录。")
                return
            recent = mems[-5:]
            lines = [f"最近 {len(recent)} 条记忆（共 {len(mems)} 条）："]
            for i, m in enumerate(reversed(recent), 1):
                ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(m.get("timestamp", 0)))
                preview = m["text"][:30] + ("..." if len(m["text"])>30 else "")
                t = "总结" if m["type"]=="summary" else "原始"
                f = " [不可检索]" if m.get("_embed_failed") else ""
                lines.append(f"{i}. [{t}{f}] ({ts}) {preview}")
            yield event.plain_result("\n".join(lines))
        except Exception as e:
            logger.error(f"查看记忆出错: {e}")
            yield event.plain_result(f"查看记忆时出错: {e}")

    @filter.command("ltm_delete")
    async def ltm_delete(self, event, index: int = -1):
        """删除单条记忆。用法：/ltm_delete <序号>（序号来自 /ltm_view）
        参数由 AstrBot 命令系统自动解析（filter.command 声明式参数）。
        """
        sid = await self._get_scope_id(event)
        n = index
        if n < 1:
            yield event.plain_result("请提供有效序号，如 /ltm_delete 3。序号来自 /ltm_view 的列表。"); return
        mems = await self._load_memories(sid)
        if not mems:
            yield event.plain_result("当前没有任何记忆。"); return
        # ltm_view 是按时间倒序展示的（最新的在前），序号 1 = mems[-1]
        idx = len(mems) - n
        if idx < 0 or idx >= len(mems):
            yield event.plain_result(f"序号 {n} 超出范围（共 {len(mems)} 条记忆，/ltm_view 查序号）。"); return
        target = mems[idx]
        preview = target["text"][:40] + ("..." if len(target["text"]) > 40 else "")
        async with self._get_memory_lock(sid):
            mems = await self._load_memories_internal(sid)
            idx2 = len(mems) - n
            if idx2 < 0 or idx2 >= len(mems):
                yield event.plain_result("记忆已变更，请重新 /ltm_view 查看。"); return
            del mems[idx2]
            await self._save_memories_internal(sid, mems)
        # ChromaDB 同步
        if self._get_config("chromadb_enabled", False):
            await self._chroma_delete_memory(sid, target["id"])
        yield event.plain_result(f"已删除记忆: [{n}] {preview}")

    @filter.command("ltm_clear")
    async def ltm_clear(self, event):
        """清除所有记忆与对话上下文（两步确认）"""
        try:
            sid = await self._get_scope_id(event)
            key = self._KV_CONFIRM_PREFIX + sid
            confirmed = await self.get_kv_data(key, False)
            if confirmed:
                async with self._get_buffer_lock(sid):
                    await self._save_buffer_internal(sid, [])
                async with self._get_memory_lock(sid):
                    await self._save_memories_internal(sid, [])
                await self.delete_kv_data(key)
                await self.delete_kv_data(self._KV_PENDING_PERSONA + sid)
                await self._save_last_summary_time(sid, 0.0)
                await self._save_profile_last_update(sid, 0.0)
                await self._save_profile_anchor(sid, 0.0)
                self._memory_cache.pop(sid, None)
                self._buffer_cache.pop(sid, None)
                self._profile_cache.pop(sid, None)
                if hasattr(self, "_affection_cache"):
                    self._affection_cache.pop(sid, None)
                rc = getattr(self, "_retrieval_cache", None)
                if rc:
                    rc.clear()
                await self.delete_kv_data(self._KV_AFFECTION_PREFIX + sid)
                await self.delete_kv_data(self._KV_AFFECTION_DAILY_PREFIX + sid)
                if self._get_config("chromadb_enabled", False):
                    await self._chroma_delete_scope(sid)
                await self._unregister_scope(sid)
                yield event.plain_result("已清除所有记忆记录和对话缓冲区。")
            else:
                await self.put_kv_data(key, True)
                yield event.plain_result("确认要清除所有记忆吗？请在 60 秒内再次发送 /ltm_clear 以确认。")
                async def cancel():
                    await asyncio.sleep(60)
                    if await self.get_kv_data(key, False):
                        await self.put_kv_data(key, False)
                asyncio.create_task(cancel())
        except Exception as e:
            logger.error(f"清除记忆出错: {e}")
            yield event.plain_result(f"清除记忆时出错: {e}")

    @filter.command("ltm_stats")
    async def ltm_stats(self, event):
        """记忆统计信息"""
        try:
            sid = await self._get_scope_id(event)
            mems = await self._load_memories(sid)
            buf = await self._load_buffer(sid)
            last = await self._load_last_summary_time(sid)
            sc = sum(1 for m in mems if m["type"]=="summary")
            rc = sum(1 for m in mems if m["type"]=="raw")
            ef = sum(1 for m in mems if m.get("_embed_failed"))
            bu = sum(1 for i in buf if i["role"]=="user")
            last_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last)) if last>0 else "从未"
            lines = [
                "【记忆统计】",
                f"总结记忆数: {sc}",
                f"原始记忆数: {rc}",
                f"嵌入失败: {ef}",
                f"总记忆数: {len(mems)} / {self._get_config('memory_limit', 500)}",
                f"缓冲区: {bu} 轮 ({len(buf)} 条)",
                f"上次总结: {last_str}",
                f"Embedding: {'已就绪' if self._get_embedding_provider() else '未配置'}",
                "管理页: AstrBot Dashboard → 插件 → 长期记忆",
            ]
            yield event.plain_result("\n".join(lines))
        except Exception as e:
            logger.error(f"统计出错: {e}")
            yield event.plain_result(f"统计记忆时出错: {e}")

    @filter.command("ltm_summary")
    async def ltm_summary(self, event):
        """手动触发对话总结"""
        try:
            sid = await self._get_scope_id(event)
            buf = await self._load_buffer(sid)
            if not buf:
                yield event.plain_result("当前缓冲区没有待总结的对话。")
                return
            await event.send(f"正在总结 {sum(1 for i in buf if i['role']=='user')} 轮对话...")
            success = await self._summarize_buffer(sid, event)
            yield event.plain_result("对话总结完成！记忆已存储。" if success else "总结失败，请检查 LLM/Embedding 配置。")
        except Exception as e:
            logger.error(f"手动总结出错: {e}")
            yield event.plain_result(f"手动总结时出错: {e}")

    @filter.command("ltm_profile")
    async def ltm_profile(self, event):
        """查看用户画像"""
        sid = await self._get_scope_id(event)
        p = await self._load_profile(sid)
        yield event.plain_result(p if p and p.strip() else "当前没有用户画像信息。")

    @filter.command("ltm_profile_update")
    async def ltm_profile_update(self, event):
        """手动更新用户画像"""
        if not self._get_config("profile_enabled", False):
            yield event.plain_result("用户画像功能未启用。"); return
        sid = await self._get_scope_id(event)
        mems = await self._load_memories(sid)
        if not any(m.get("type")=="summary" and m.get("text") for m in mems):
            yield event.plain_result("没有足够的记忆数据来生成画像。"); return
        await event.send("正在更新用户画像...")
        await self._update_profile(sid, event)
        p = await self._load_profile(sid)
        yield event.plain_result(f"用户画像已更新：\n{p}" if p and p.strip() else "画像更新失败。")

    @filter.command("ltm_profile_clear")
    async def ltm_profile_clear(self, event):
        """清除用户画像"""
        sid = await self._get_scope_id(event)
        await self._save_profile(sid, "")
        self._profile_cache.pop(sid, None)
        yield event.plain_result("用户画像已清除。")

    @filter.command("ltm_affection")
    async def ltm_affection(self, event):
        """查看好感度"""
        if not self._get_config("affection_enabled", False):
            yield event.plain_result("好感度系统未启用。"); return
        sid = await self._get_scope_id(event)
        data = await self._load_affection(sid)
        score = data.get("score", 0.0)
        level_cn = data.get("level_cn", "陌生人")
        level = data.get("level", "stranger")
        total = data.get("total_interactions", 0)
        pos = data.get("positive_count", 0)
        neg = data.get("negative_count", 0)
        bar_len = 20
        progress = (score + 100) / 200
        filled = round(progress * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)
        emoji_map = {
            "nemesis": "💀", "hostile": "😠", "cold": "😐",
            "stranger": "👤", "acquaintance": "🙂", "friend": "😊",
            "close_friend": "😄", "confidant": "💖",
        }
        emoji = emoji_map.get(level, "❓")
        lines = [
            f"{emoji} 【好感度】",
            f"分数: {score:.1f} / 100  [{level_cn}]",
            f"[{bar}]",
            f"累计互动: {total} 轮 | 正向: {pos} | 负向: {neg}",
        ]
        history = data.get("history", [])
        if history:
            recent = history[-3:]
            lines.append("近期变化:")
            for h in reversed(recent):
                ts = time.strftime("%m-%d %H:%M", time.localtime(h.get("timestamp", 0)))
                d = h.get("delta", 0)
                sign = "+" if d >= 0 else ""
                lines.append(f"  {ts} {sign}{d:.1f} — {h.get('reason', '')}")
        yield event.plain_result("\n".join(lines))

    @filter.command("ltm_persona_confirm")
    async def ltm_persona_confirm(self, event):
        """确认应用待处理的角色状态变化到人设"""
        sid = await self._get_scope_id(event)
        pending = await self._load_pending_persona(sid)
        if not pending.get("changes"):
            yield event.plain_result("当前没有待确认的角色状态变化。")
            return
        changes = pending["changes"]
        try:
            from astrbot.api.star import logger
            pm = self.context.persona_manager
            persona_list = pm.get_all_personas() if callable(getattr(pm, "get_all_personas", None)) else []
            if not persona_list:
                yield event.plain_result("未找到可更新的人设。请先在 AstrBot 中配置至少一个人设。")
                return
            # 优先使用当前会话绑定的人设（Conversation.persona_id），避免改错人设；
            # 取不到时回退到第一个人设
            persona = None
            try:
                conv_mgr = self.context.conversation_manager
                curr_cid = await conv_mgr.get_curr_conversation_id(event.unified_msg_origin)
                if curr_cid:
                    conv = await conv_mgr.get_conversation(event.unified_msg_origin, curr_cid)
                    conv_pid = getattr(conv, "persona_id", None) if conv else None
                    if conv_pid:
                        for p in persona_list:
                            if getattr(p, "id", None) == conv_pid:
                                persona = p
                                break
            except Exception:
                pass
            if persona is None:
                persona = persona_list[0]
            current_prompt = getattr(persona, "system_prompt", "")
            new_prompt = current_prompt
            applied = []
            for c in changes:
                old_text = c.get("old", "")
                new_text = c.get("new", "")
                if old_text and new_text and old_text in new_prompt:
                    new_prompt = new_prompt.replace(old_text, new_text, 1)
                    applied.append(f"「{old_text}」→「{new_text}」")
                elif old_text == "" and new_text:
                    new_prompt += f"\n{new_text}"
                    applied.append(f"新增「{new_text}」")
            if applied:
                await pm.update_persona(persona_id=persona.id, system_prompt=new_prompt)
                await self._clear_pending_persona(sid)
                yield event.plain_result(f"✅ 人设已更新（{len(applied)} 处）：\n" + "\n".join(applied))
            else:
                yield event.plain_result("无法应用变更：人设原文中未找到匹配文本，请手动修改。")
        except Exception as e:
            logger.warning(f"人设更新失败: {e}")
            yield event.plain_result(f"更新失败: {e}")

    @filter.command("ltm_persona_reject")
    async def ltm_persona_reject(self, event):
        """拒绝待处理的角色状态变化"""
        sid = await self._get_scope_id(event)
        pending = await self._load_pending_persona(sid)
        if not pending.get("changes"):
            yield event.plain_result("当前没有待确认的角色状态变化。")
            return
        await self._clear_pending_persona(sid)
        yield event.plain_result("已拒绝角色状态变化，下次总结后重新检测。")

    @filter.command("ltm_persona_pending")
    async def ltm_persona_pending(self, event):
        """查看待确认的角色状态变化"""
        sid = await self._get_scope_id(event)
        pending = await self._load_pending_persona(sid)
        if not pending.get("changes"):
            yield event.plain_result("当前没有待确认的角色状态变化。")
            return
        lines = [f"📋 待确认的角色状态变化（{pending.get('summary', '')}）："]
        for i, c in enumerate(pending["changes"][:5], 1):
            lines.append(f"{i}. 「{c.get('old', '（新增）')}」→「{c.get('new', '')}」")
            if c.get("reason"):
                lines.append(f"   原因: {c['reason']}")
        lines.append("\n发送 /ltm_persona_confirm 确认应用，/ltm_persona_reject 拒绝")
        yield event.plain_result("\n".join(lines))

    def _build_memory_context(self, memories: list) -> str:
        max_single = self._get_config("max_single_memory_chars", 200)
        max_total = self._get_config("max_memory_context_length", 2000)
        lines = ["以下是关于该用户的历史记忆总结（按相关性排序）："]

        # 叙事时间线：收集有叙事时间的记忆，按时间排序展示
        narrative_times = [m.get("narrative_time", "") for m in memories if m.get("narrative_time")]
        if narrative_times:
            lines.append(f"[叙事时间线] {', '.join(narrative_times[:5])}")

        total = sum(len(l) for l in lines)
        for i, m in enumerate(memories, 1):
            prefix = f"[{m['narrative_time']}] " if m.get("narrative_time") else ""
            text = (prefix + m["text"])[:max_single-3] + "..." if len(prefix + m["text"]) > max_single else prefix + m["text"]
            line = f"{i}. {text}"
            if total + len(line) + 1 > max_total:
                lines.append(f"{i}. ...(更多记忆因长度限制被省略)")
                break
            lines.append(line)
            total += len(line) + 1
        return "\n".join(lines)

    async def terminate(self):
        # AstrBot 官方生命周期钩子：插件被禁用/重载前调用，用于清理资源
        self._memory_cache.clear()
        self._buffer_cache.clear()
        self._profile_cache.clear()
        self._affection_cache.clear()
        rc = getattr(self, "_retrieval_cache", None)
        if rc:
            rc.clear()
        ec = getattr(self, "_embed_cache", None)
        if ec:
            ec.clear()

_THINKING_RE = re.compile(r'<thinking>.*?</thinking>', re.DOTALL)


def strip_thinking(text: str) -> str:
    """移除 DeepSeek 等模型的 thinking 标签及其内容。"""
    return _THINKING_RE.sub('', text).strip()