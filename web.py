"""界面模块：AstrBot 插件 Web API（register_web_api）+ 用户命令。

v2.9.0 起，本插件的管理界面已并入 AstrBot Dashboard：
  - 前端页面放在 pages/<page_name>/（由 Dashboard 以受限 iframe 加载）
  - 后端接口通过 context.register_web_api() 注册，路由以插件名为前缀
  - 认证由 Dashboard 统一负责（request.username 即当前登录用户），
    因此不再需要独立端口、静态 Token、TOTP、session/CSRF 等自建鉴权

参考：https://docs.astrbot.app/dev/star/guides/plugin-pages.html
"""
import asyncio
import re
import time
import uuid
import traceback

from astrbot.api import logger
from astrbot.api.web import error_response, json_response, request

from .affection import compute_level, estimate_score_from_stats


# 插件名（作为所有 Web 路由的前缀，必须是插件目录名）
PLUGIN_NAME = "astrbot_plugin_longterm_memory"

# 统一对外错误文案（不泄漏内部细节）
_SAFE_ERROR_MSG = "Internal server error"
# Scope ID 校验：只拒绝**控制字符**与**路径分隔符**，其余一律放行。
# scope_id 仅用作 KV 键后缀（不参与任何文件路径），过窄的白名单反而会误伤：
# AstrBot 各平台的 umo 可能包含 ! . @ 甚至中文，例如内置 WebChat 的
# "webchat:FriendMessage:webchat!user!session"——早期只允许 [A-Za-z0-9_:-]，
# 这类作用域在管理页会**全部返回 400**，表现为"记忆和数据都加载不出来"。
_SCOPE_ID_RE = re.compile(r"^[^\x00-\x1f\x7f/\\]{1,512}$")
# Prompt 模板最大长度（10KB），防止恶意注入超长文本
_MAX_PROMPT_TEMPLATE_LENGTH = 10 * 1024
# 重试嵌入超时（秒）
_EMBED_RETRY_TIMEOUT = 30
# 导出备份的规模上限：作用域数与记忆总条数都要封顶。
# 只限作用域数不够——每个作用域可能有上百条记忆（500×200=10 万条），
# 单次 JSON 响应会大到拖垮 bridge / HTTP。
_EXPORT_MAX_SCOPES = 500
_EXPORT_MAX_MEMORIES = 20000
# 导入备份时单条记忆文本的长度上限（正常记忆 ≤200 字，留足余量）
_MAX_IMPORT_TEXT_LEN = 4000
# 重建画像时喂给 LLM 的记忆条数上限（比日常增量的窗口更大）
_PROFILE_REGEN_ITEMS = 60


def _validate_scope_id(sid: str) -> bool:
    """校验 scope_id 仅包含安全字符，防止路径/键注入。"""
    return bool(sid) and bool(_SCOPE_ID_RE.match(sid))


def _route(path: str) -> str:
    """拼接带插件名前缀的注册路由。"""
    return f"/{PLUGIN_NAME}/{path.lstrip('/')}"


class WebMixin:
    """Dashboard 插件 Web API 与命令处理。"""

    # ======================== 路由注册 ========================
    def register_web_apis(self):
        """注册全部插件 Web API（在 __init__ 中调用一次）。

        Note: Dashboard 的 bridge 只提供 apiGet / apiPost，因此所有写操作
        一律走 POST，不注册 DELETE/PUT 路由。
        """
        try:
            reg = self.context.register_web_api
        except Exception as e:  # pragma: no cover - 旧版 AstrBot 无此 API
            logger.error(f"[长期记忆] 当前 AstrBot 不支持 register_web_api，管理页不可用: {e}")
            return

        routes = [
            ("stats", self._api_stats, ["GET"], "记忆统计概览"),
            ("scopes", self._api_scopes, ["GET"], "作用域列表"),
            ("memories", self._api_get_memories, ["GET"], "读取指定作用域的记忆"),
            ("buffer", self._api_get_buffer, ["GET"], "读取待总结的对话缓冲区"),
            ("memories/clear", self._api_clear_scope, ["POST"], "清空指定作用域全部数据"),
            ("memory/delete", self._api_delete_memory, ["POST"], "删除单条记忆"),
            ("memory/retry", self._api_retry_embed, ["POST"], "重新生成单条记忆的嵌入"),
            ("memory/retry-all", self._api_retry_all, ["POST"], "一键补嵌全部失败记忆（后台任务）"),
            ("memory/delete-failed", self._api_delete_failed, ["POST"], "删除嵌入失败的记忆"),
            ("embed/info", self._api_embed_info, ["GET"], "Embedding 诊断信息"),
            ("embed/test", self._api_embed_test, ["POST"], "测试 Embedding 连通性"),
            ("chroma/reindex", self._api_chroma_reindex, ["POST"], "把已有向量回填到 ChromaDB"),
            ("job/status", self._api_job_status, ["GET"], "后台任务进度"),
            ("memory/update", self._api_update_memory, ["POST"], "编辑单条记忆"),
            ("memories/reorganize", self._api_reorganize, ["POST"], "重整记忆（合并/拆分话题，可回滚）"),
            ("memories/restore-backup", self._api_restore_backup, ["POST"], "回滚到重整前的备份"),
            ("memories/migrate", self._api_migrate_memories, ["POST"], "把记忆迁移/复制到另一个作用域"),
            ("export", self._api_export_backup, ["GET"], "导出全部数据备份（JSON）"),
            ("import", self._api_import_backup, ["POST"], "从备份导入数据"),
            ("profile/regenerate", self._api_regenerate_profile, ["POST"], "从现有记忆重新生成画像"),
            ("profile", self._api_get_profile, ["GET"], "读取用户画像"),
            ("profile/save", self._api_set_profile, ["POST"], "保存用户画像"),
            ("affection", self._api_get_affection, ["GET"], "读取亲密度"),
            ("affection/set", self._api_set_affection, ["POST"], "手动设定亲密度分数"),
            ("affection/reevaluate", self._api_reevaluate_affection, ["POST"], "让 AI 按既有互动重新评估亲密度"),
            ("config", self._api_get_config, ["GET"], "读取插件配置"),
            ("config/defaults", self._api_get_config_defaults, ["GET"], "读取各配置项的默认值"),
            ("config/save", self._api_set_config, ["POST"], "保存插件配置"),
        ]
        for path, handler, methods, desc in routes:
            try:
                reg(_route(path), handler, methods, desc)
            except Exception as e:
                logger.warning(f"[长期记忆] 注册 Web API '{path}' 失败: {e}")
        logger.info(f"[长期记忆] 已注册 {len(routes)} 个 Dashboard 插件 API")

    # ======================== 内部工具 ========================
    def _resolve_scope(self):
        """从 query 取并校验 scope 参数，返回 (scope_id, 错误响应)。"""
        sid = (request.query.get("scope", "") or "").strip()
        if not sid:
            return "", error_response("缺少 scope 参数", status_code=400)
        if not _validate_scope_id(sid):
            return "", error_response("scope 参数非法", status_code=400)
        return sid, None

    async def _read_json(self) -> dict:
        payload = await request.json(default={})
        return payload if isinstance(payload, dict) else {}

    # ======================== 只读接口 ========================
    async def _api_stats(self):
        """总览统计。"""
        try:
            scopes = await self._get_all_scopes()
            # 防御：作用域极多时限制统计范围，避免逐个 KV 读取拖慢接口
            if len(scopes) > 500:
                logger.debug(f"[长期记忆] 作用域数量过多（{len(scopes)}），统计仅处理前 500 个")
                scopes = scopes[:500]
            total = sc = rc = fc = 0
            aff_enabled = self._get_config("affection_enabled", False)
            aff_count = 0
            for sid in scopes:
                # 注意用 is not None 判断缓存命中：空列表也是有效缓存
                mems = self._memory_cache.get(sid)
                if mems is None:
                    mems = await self._load_memories(sid)
                for m in mems:
                    total += 1
                    if m.get("type") == "summary":
                        sc += 1
                    elif m.get("type") == "raw":
                        rc += 1
                    if m.get("_embed_failed"):
                        fc += 1
                if aff_enabled:
                    try:
                        aff = await self._load_affection(sid)
                        if aff.get("total_interactions", 0) > 0:
                            aff_count += 1
                    except Exception:
                        pass
            result = {
                "total_memories": total,
                "summary_count": sc,
                "raw_count": rc,
                "embed_failed_count": fc,
                "scope_count": len(scopes),
                "memory_limit": self._get_config("memory_limit", 500),
                "keep_raw": self._get_config("keep_raw_messages", False),
                "embedding_ready": self._get_embedding_provider() is not None,
                "memory_scope": self._get_config("memory_scope", "user"),
                "username": request.username,
            }
            if aff_enabled:
                result["affection_enabled"] = True
                result["affection_tracked_scopes"] = aff_count
            return json_response(result)
        except Exception as e:
            logger.warning(f"[长期记忆] 统计异常: {e}")
            return error_response(_SAFE_ERROR_MSG, status_code=500)

    async def _api_scopes(self):
        """作用域列表（含记忆/缓冲计数）。"""
        try:
            scopes = await self._get_all_scopes()
            result = []
            for sid in scopes:
                mems = self._memory_cache.get(sid)
                if mems is None:
                    mems = await self._load_memories(sid)
                buf = self._buffer_cache.get(sid)
                if buf is None:
                    buf = await self._load_buffer(sid)
                result.append({
                    "scope_id": sid,
                    "memory_count": len(mems),
                    "summary_count": sum(1 for m in mems if m.get("type") == "summary"),
                    "buffer_count": len(buf),
                })
            return json_response({
                "scopes": result,
                # 一并返回当前隔离模式：memory_scope=user 时每个用户只有一个作用域，
                # 页面上"只有一个/没有会话作用域"是预期行为而非数据丢失
                "memory_scope": self._get_config("memory_scope", "user"),
                "scope_count": len(result),
            })
        except Exception as e:
            logger.warning(f"[长期记忆] 作用域列表异常: {e}")
            return error_response(_SAFE_ERROR_MSG, status_code=500)

    async def _api_get_memories(self):
        """读取指定作用域的记忆列表（按时间倒序）。"""
        try:
            sid, err = self._resolve_scope()
            if err:
                return err
            mems = await self._load_memories(sid)
            items = [{
                "id": m.get("id"),
                "text": m.get("text"),
                "type": m.get("type"),
                "timestamp": m.get("timestamp"),
                "_embed_failed": m.get("_embed_failed", False),
                "importance": m.get("importance", 3),
                "tags": m.get("tags", []),
            } for m in mems]
            items.sort(key=lambda x: x.get("timestamp") or 0, reverse=True)
            # 附带"是否存在重整备份"，供页面显示回滚入口
            backup = await self.get_kv_data(self._KV_MEMORIES_BACKUP_PREFIX + sid, None)
            has_backup = isinstance(backup, dict) and isinstance(backup.get("memories"), list)
            try:
                pending = await self.get_kv_data(self._KV_REORG_PENDING_PREFIX + sid, None)
                pending_texts = (pending or {}).get("texts") if isinstance(pending, dict) else None
                reorg_pending = len([t for t in (pending_texts or []) if isinstance(t, str) and t.strip()])
            except Exception:
                reorg_pending = 0
            return json_response({
                "memories": items,
                "scope_id": sid,
                "has_backup": has_backup,
                "backup_time": (backup or {}).get("timestamp", 0) if has_backup else 0,
                "reorg_pending": reorg_pending,
            })
        except Exception as e:
            logger.warning(f"[长期记忆] 获取记忆异常: {e}")
            return error_response(_SAFE_ERROR_MSG, status_code=500)

    async def _api_get_buffer(self):
        """只读展示待总结缓冲区，便于排查"为什么没有生成新记忆"。"""
        try:
            sid, err = self._resolve_scope()
            if err:
                return err
            buf = await self._load_buffer(sid)
            items = [{
                "role": i.get("role", "?"),
                "preview": (i.get("content", "") or "")[:200],
                "timestamp": i.get("timestamp", 0),
            } for i in buf[-60:]]
            return json_response({
                "scope_id": sid,
                "count": len(buf),
                "user_rounds": sum(1 for i in buf if i.get("role") == "user"),
                "items": items,
            })
        except Exception as e:
            logger.warning(f"[长期记忆] 获取缓冲区异常: {e}")
            return error_response(_SAFE_ERROR_MSG, status_code=500)

    async def _api_get_profile(self):
        try:
            sid, err = self._resolve_scope()
            if err:
                return err
            profile = await self._load_profile(sid)
            return json_response({"scope_id": sid, "profile": profile})
        except Exception as e:
            logger.warning(f"[长期记忆] 获取画像异常: {e}")
            return error_response(_SAFE_ERROR_MSG, status_code=500)

    async def _api_get_affection(self):
        try:
            sid, err = self._resolve_scope()
            if err:
                return err
            data = await self._load_affection(sid)
            # 按历史互动量估算"若一开始就用当前参数"应得的分数（仅供参考，不落库）
            try:
                alpha = float(self._get_config("affection_ema_alpha", 0.4))
                divisor = float(self._get_config("affection_difficulty_divisor", 200.0))
            except (TypeError, ValueError):
                alpha, divisor = 0.4, 200.0
            estimated = estimate_score_from_stats(data, alpha, divisor)
            return json_response({
                "scope_id": sid,
                "affection": data,
                "estimated_score": estimated,
                "estimate_level_cn": compute_level(
                    estimated, self._get_config("affection_level_thresholds", None)
                ).get("level_cn", ""),
                "estimate_basis": {
                    "total_interactions": data.get("total_interactions", 0),
                    "positive_count": data.get("positive_count", 0),
                    "negative_count": data.get("negative_count", 0),
                    "alpha": alpha,
                    "divisor": divisor,
                },
            })
        except Exception as e:
            logger.warning(f"[长期记忆] 获取好感度异常: {e}")
            return error_response(_SAFE_ERROR_MSG, status_code=500)

    async def _api_reevaluate_affection(self):
        """让 LLM 依据既有互动记录重新判定亲密度（后台任务）。"""
        try:
            payload = await self._read_json()
            sid = str(payload.get("scope", "") or "").strip()
            if not _validate_scope_id(sid):
                return error_response("scope 参数非法", status_code=400)
            job = self._job("affection")
            if job["running"]:
                return error_response("AI 评估正在进行中", status_code=409)
            job.update({
                "running": True, "scope": sid, "total": 1, "done": 0, "ok": 0,
                "failed": 0, "error": "", "message": "AI 正在评估既有互动…",
                "started_at": time.time(), "finished_at": 0.0,
            })
            job["task"] = asyncio.create_task(self._run_reevaluate_affection(sid, job))
            return json_response({"started": True})
        except Exception as e:
            logger.warning(f"[长期记忆] 启动 AI 亲密度评估异常: {e}")
            return error_response(_SAFE_ERROR_MSG, status_code=500)

    async def _run_reevaluate_affection(self, scope: str, job: dict):
        try:
            res = await self._reevaluate_affection(scope, job)
            if res.get("ok"):
                job["ok"] = 1
                job["message"] = (
                    f"AI 评估完成：{res['old_score']} → {res['score']}（{res['level_cn']}）"
                    + (f" · {res['reason']}" if res.get("reason") else "")
                )
                job["result"] = {
                    "old_score": res["old_score"], "score": res["score"],
                    "level_cn": res["level_cn"], "reason": res.get("reason", ""),
                }
            else:
                job["failed"] = 1
                job["message"] = job.get("error") or "AI 评估未生效"
        except Exception as e:
            job["failed"] = 1
            job["error"] = str(e)
            logger.warning(f"[长期记忆] AI 亲密度评估失败: {e}")
        finally:
            job["running"] = False
            job["done"] = 1
            job["finished_at"] = time.time()

    async def _api_set_affection(self):
        """手动设定亲密度分数。

        用途：分数是在旧规则（增益系数 0.15、每日上限 10）下累积的，普遍偏低；
        升级后可以一次性修正到符合实际关系的水平，之后继续自然增长。
        """
        try:
            payload = await self._read_json()
            sid = str(payload.get("scope", "") or "").strip()
            if not _validate_scope_id(sid):
                return error_response("scope 参数非法", status_code=400)
            try:
                score = float(payload.get("score"))
            except (TypeError, ValueError):
                return error_response("score 必须是数字", status_code=400)
            if not (-100.0 <= score <= 100.0):
                return error_response("score 必须在 -100 ~ 100 之间", status_code=400)

            lock = self._get_affection_lock(sid)
            async with lock:
                aff = await self._load_affection(sid)
                old_score = float(aff.get("score", 0.0) or 0.0)
                info = compute_level(score, self._get_config("affection_level_thresholds", None))
                aff["score"] = round(score, 1)
                aff["level"] = info.get("level", "stranger")
                aff["level_cn"] = info.get("level_cn", "陌生人")
                aff["last_update"] = time.time()
                history = aff.get("history", [])
                history.append({
                    "delta": round(score - old_score, 2),
                    "reason": "手动调整",
                    "timestamp": time.time(),
                    "score_after": round(score, 1),
                    "level_after": info.get("level_cn", ""),
                })
                aff["history"] = history[-30:]
                await self._save_affection(sid, aff)
            logger.info(
                f"[好感度] 手动调整 ({sid}): {old_score:.1f} → {score:.1f}（{info.get('level_cn','')}）"
            )
            return json_response({
                "success": True,
                "old_score": round(old_score, 1),
                "score": round(score, 1),
                "level_cn": info.get("level_cn", ""),
            })
        except Exception as e:
            logger.warning(f"[长期记忆] 设定亲密度异常: {e}")
            return error_response(_SAFE_ERROR_MSG, status_code=500)

    # ======================== 配置读写 ========================
    # 配置键白名单（移除自建 WebUI 后不再包含 webui_* / totp_enabled）
    _ALLOWED_KEYS = {
        "memory_limit", "retrieve_top_k", "similarity_threshold",
        "memory_eviction_strategy", "summary_topic_boundary_enabled",
        "retrieve_diversity_lambda",
        "context_window_tokens", "chromadb_enabled", "enable_rerank",
        "memory_scope", "summary_trigger_rounds", "summary_interval_seconds",
        "min_summary_rounds", "keep_raw_messages",
        "embedding_provider_id", "rerank_provider_id",
        "summary_prompt_template", "memory_inject_position",
        "max_single_memory_chars", "max_memory_context_length",
        "llm_timeout_seconds", "llm_max_retries", "summary_fallback_provider_id",
        "profile_enabled", "profile_prompt_template", "profile_inject_position",
        "profile_max_chars", "profile_update_interval_hours", "min_profile_summaries",
        "affection_enabled", "affection_eval_cooldown_seconds",
        "affection_ema_alpha", "affection_daily_cap",
        "affection_rule_eval_rounds", "affection_difficulty_divisor",
        "affection_inject_position", "affection_analysis_prompt_template",
        "affection_level_thresholds",
        "memory_decay_enabled", "memory_decay_lambda",
        "memory_importance_enabled", "memory_importance_weight",
        "memory_conflict_detection_enabled", "persona_drift_enabled",
    }
    _INT_KEYS = {
        "memory_limit", "retrieve_top_k", "context_window_tokens",
        "summary_trigger_rounds", "summary_interval_seconds", "min_summary_rounds",
        "max_single_memory_chars", "max_memory_context_length",
        "llm_timeout_seconds", "llm_max_retries",
        "profile_max_chars", "profile_update_interval_hours", "min_profile_summaries",
        "affection_eval_cooldown_seconds", "affection_rule_eval_rounds",
    }
    _FLOAT_KEYS = {
        "similarity_threshold", "retrieve_diversity_lambda",
        "affection_ema_alpha", "affection_daily_cap",
        "affection_difficulty_divisor", "memory_importance_weight",
        "memory_decay_lambda",
    }
    _BOOL_KEYS = {
        "enable_rerank", "keep_raw_messages", "profile_enabled", "affection_enabled",
        "chromadb_enabled", "memory_decay_enabled", "memory_importance_enabled",
        "memory_conflict_detection_enabled", "persona_drift_enabled",
        "summary_topic_boundary_enabled",
    }
    _PROMPT_TEMPLATE_KEYS = {
        "summary_prompt_template", "profile_prompt_template",
        "affection_analysis_prompt_template",
    }
    _INJECT_POSITIONS = ("system_prompt", "extra_user_content")

    _CONFIG_KEYS = [
        "memory_scope", "memory_limit", "retrieve_top_k", "similarity_threshold",
        "memory_eviction_strategy", "summary_topic_boundary_enabled",
        "retrieve_diversity_lambda",
        "context_window_tokens", "chromadb_enabled",
        "summary_trigger_rounds", "summary_interval_seconds", "min_summary_rounds",
        "keep_raw_messages", "summary_prompt_template", "memory_inject_position",
        "max_single_memory_chars", "max_memory_context_length",
        "llm_timeout_seconds", "llm_max_retries",
        "embedding_provider_id", "rerank_provider_id", "summary_fallback_provider_id",
        "enable_rerank",
        "profile_enabled", "profile_prompt_template", "profile_inject_position",
        "profile_max_chars", "profile_update_interval_hours", "min_profile_summaries",
        "persona_drift_enabled",
        "affection_enabled", "affection_eval_cooldown_seconds", "affection_ema_alpha",
        "affection_daily_cap", "affection_rule_eval_rounds",
        "affection_difficulty_divisor", "affection_inject_position",
        "affection_analysis_prompt_template", "affection_level_thresholds",
        "memory_decay_enabled", "memory_decay_lambda",
        "memory_importance_enabled", "memory_importance_weight",
        "memory_conflict_detection_enabled",
    ]

    async def _api_get_config(self):
        try:
            return json_response({"config": {k: self._get_config(k) for k in self._CONFIG_KEYS}})
        except Exception as e:
            logger.warning(f"[长期记忆] 读取配置异常: {e}")
            return error_response(_SAFE_ERROR_MSG, status_code=500)

    async def _api_get_config_defaults(self):
        """返回 _conf_schema.json 中各项的默认值。

        供管理页「↺ 恢复默认」逐项使用——比 AstrBot 插件页的"整体恢复默认"安全得多：
        后者会顺带把 profile_enabled / affection_enabled / persona_drift_enabled
        等默认值为 false 的开关一起关掉。
        """
        try:
            import json as _json
            from pathlib import Path
            schema_path = Path(__file__).resolve().parent / "_conf_schema.json"
            schema = _json.loads(schema_path.read_text(encoding="utf-8"))
            defaults = {}
            for k in self._CONFIG_KEYS:
                item = schema.get(k)
                if isinstance(item, dict) and "default" in item:
                    defaults[k] = item["default"]
            return json_response({"defaults": defaults})
        except Exception as e:
            logger.warning(f"[长期记忆] 读取配置默认值失败: {e}")
            return error_response("读取默认值失败", status_code=500)

    async def _api_set_config(self):
        try:
            data = await self._read_json()
            raw_cfg = data.get("config")
            if not isinstance(raw_cfg, dict):
                return error_response("请求体需为 {config: {...}}", status_code=400)

            updated = []
            for k, v in raw_cfg.items():
                if k not in self._ALLOWED_KEYS:
                    logger.warning(f"[长期记忆] 拒绝未授权的配置键: {k}")
                    continue
                if k in self._INT_KEYS:
                    try:
                        v = int(v)
                    except (TypeError, ValueError):
                        continue
                elif k in self._FLOAT_KEYS:
                    try:
                        v = float(v)
                    except (TypeError, ValueError):
                        continue
                elif k in self._BOOL_KEYS:
                    if not isinstance(v, bool):
                        continue
                if k == "memory_eviction_strategy" and v not in ("importance", "fifo"):
                    logger.warning(f"[长期记忆] 拒绝非法淘汰策略: {v}")
                    continue
                if k == "memory_scope" and v not in ("user", "group_user", "user_bot", "conversation"):
                    logger.warning(f"[长期记忆] 拒绝非法 memory_scope: {v}")
                    continue
                if k in ("memory_inject_position", "profile_inject_position",
                         "affection_inject_position") and v not in self._INJECT_POSITIONS:
                    logger.warning(f"[长期记忆] 拒绝非法注入位置: {v}")
                    continue
                if k in self._PROMPT_TEMPLATE_KEYS and isinstance(v, str) and len(v) > _MAX_PROMPT_TEMPLATE_LENGTH:
                    logger.warning(f"[长期记忆] 拒绝过长的 Prompt 模板 '{k}'")
                    continue
                try:
                    self.config[k] = v
                    updated.append(k)
                except Exception:
                    continue

            if updated:
                if {"embedding_provider_id", "rerank_provider_id", "enable_rerank"} & set(updated):
                    self._embedding_provider = None
                    self._embedding_provider_loaded = False
                    self._rerank_provider = None
                    self._rerank_provider_loaded = False
                if "chromadb_enabled" in updated:
                    self._init_chromadb()
                try:
                    if hasattr(self.config, "save_config"):
                        self.config.save_config()
                except Exception as e:
                    logger.warning(f"[长期记忆] 配置持久化失败: {e}")
            return json_response({"success": True, "updated_keys": updated})
        except Exception as e:
            logger.warning(f"[长期记忆] 保存配置异常: {traceback.format_exc()}")
            return error_response(_SAFE_ERROR_MSG, status_code=500)

    # ======================== 写操作 ========================
    async def _api_set_profile(self):
        try:
            data = await self._read_json()
            sid = str(data.get("scope", "") or "").strip()
            if not _validate_scope_id(sid):
                return error_response("scope 参数非法", status_code=400)
            text = data.get("profile", "")
            if not isinstance(text, str):
                return error_response("profile 必须是字符串", status_code=400)
            if len(text) > 10000:
                return error_response("profile 长度不能超过 10000 字符", status_code=400)
            await self._save_profile(sid, text)
            return json_response({"success": True, "scope_id": sid})
        except Exception as e:
            logger.warning(f"[长期记忆] 保存画像异常: {e}")
            return error_response(_SAFE_ERROR_MSG, status_code=500)

    async def _api_delete_memory(self):
        try:
            data = await self._read_json()
            sid = str(data.get("scope", "") or "").strip()
            mid = str(data.get("id", "") or "").strip()
            if not _validate_scope_id(sid) or not mid:
                return error_response("scope/id 参数非法", status_code=400)
            async with self._get_memory_lock(sid):
                mems = await self._load_memories_internal(sid)
                new = [m for m in mems if m.get("id") != mid]
                deleted = len(mems) - len(new)
                if deleted:
                    await self._save_memories_internal(sid, new)
            if deleted and not new:
                await self._unregister_scope(sid)
            if deleted and self._get_config("chromadb_enabled", False):
                await self._chroma_delete_memory(sid, mid)
            return json_response({"deleted": deleted})
        except Exception as e:
            logger.warning(f"[长期记忆] 删除记忆异常: {e}")
            return error_response(_SAFE_ERROR_MSG, status_code=500)

    async def _api_retry_embed(self):
        """重新生成单条记忆的嵌入（带超时，避免 embedding API 卡死导致请求挂起）。"""
        try:
            import asyncio
            data = await self._read_json()
            sid = str(data.get("scope", "") or "").strip()
            mid = str(data.get("id", "") or "").strip()
            if not _validate_scope_id(sid) or not mid:
                return error_response("scope/id 参数非法", status_code=400)
            async with self._get_memory_lock(sid):
                mems = await self._load_memories_internal(sid)
                target = next((m for m in mems if m.get("id") == mid and m.get("_embed_failed")), None)
                if not target:
                    return error_response("该记忆未标记为嵌入失败或不存在", status_code=404)
                text = target.get("text", "")
            if not text:
                return error_response("记忆内容为空，无法嵌入", status_code=400)
            try:
                emb = await asyncio.wait_for(self._embed_text(text), timeout=_EMBED_RETRY_TIMEOUT)
            except asyncio.TimeoutError:
                return error_response(f"Embedding 生成超时（{_EMBED_RETRY_TIMEOUT}s），请检查 Provider 网络", status_code=504)
            if not emb:
                return error_response(
                    "Embedding 生成失败：请检查 Embedding Provider 是否可用"
                    "（配置 embedding_provider_id 或确认已配置嵌入模型）",
                    status_code=500,
                )
            async with self._get_memory_lock(sid):
                mems = await self._load_memories_internal(sid)
                for m in mems:
                    if m.get("id") == mid:
                        m["embedding"] = emb
                        m.pop("_embed_failed", None)
                        await self._save_memories_internal(sid, mems)
                        if self._get_config("chromadb_enabled", False):
                            try:
                                await self._chroma_upsert_memory(
                                    sid, mid, text, emb,
                                    m.get("type", "summary"), m.get("importance", 3),
                                    m.get("tags", []) or [],
                                    m.get("timestamp", time.time()),
                                    m.get("narrative_time", ""),
                                )
                            except Exception:
                                pass
                        logger.info(f"[长期记忆] WebUI 手动补嵌成功 (id={mid[:8]})")
                        return json_response({"success": True, "id": mid})
            return error_response("更新失败：记忆可能已被删除", status_code=500)
        except Exception as e:
            logger.warning(f"[长期记忆] 手动补嵌异常: {e}")
            return error_response(_SAFE_ERROR_MSG, status_code=500)

    async def _api_clear_scope(self):
        """清空指定作用域的全部数据（记忆/缓冲/画像/好感度/时间戳）。"""
        try:
            data = await self._read_json()
            sid = str(data.get("scope", "") or "").strip()
            if not _validate_scope_id(sid):
                return error_response("scope 参数非法", status_code=400)
            async with self._get_buffer_lock(sid):
                await self._save_buffer_internal(sid, [])
            # 清空前自动快照，避免误操作导致永久丢失
            backed_up = await self._backup_memories_snapshot(sid, "管理页清空作用域")
            async with self._get_memory_lock(sid):
                mems = await self._load_memories_internal(sid)
                count = len(mems)
                await self._save_memories_internal(sid, [])
            async with self._get_profile_lock(sid):
                await self._save_profile_internal(sid, "")
            aff_cache = getattr(self, "_affection_cache", None)
            if aff_cache is not None:
                aff_cache.pop(sid, None)
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
            for pool in (self._buffer_locks, self._memory_locks,
                         self._profile_locks, getattr(self, "_affection_locks", {})):
                if isinstance(pool, dict):
                    pool.pop(sid, None)
            self._dim_check_triggered.discard(sid)
            if self._get_config("chromadb_enabled", False):
                await self._chroma_delete_scope(sid)
            await self._unregister_scope(sid)
            logger.info(f"[长期记忆] Dashboard 已清除 {sid} 的全部数据")
            return json_response({"deleted": count})
        except Exception as e:
            logger.warning(f"[长期记忆] 清除作用域异常: {e}")
            return error_response(_SAFE_ERROR_MSG, status_code=500)

    # ======================== 嵌入诊断与批量补嵌 ========================
    # 批量补嵌的批大小：一次 API 调用处理多条，兼顾速度与单次响应体大小
    _EMBED_BATCH_SIZE = 8
    # 批间轻微节流，降低触发上游限流的概率
    _EMBED_BATCH_DELAY = 0.15

    def _job(self, name: str = "embed") -> dict:
        """后台任务状态槽（懒初始化）。name: embed / profile / reorg。"""
        jobs = getattr(self, "_jobs", None)
        if jobs is None:
            jobs = {}
            self._jobs = jobs
        job = jobs.get(name)
        if job is None:
            job = {
                "running": False, "scope": "", "total": 0, "done": 0,
                "ok": 0, "failed": 0, "error": "", "message": "",
                "started_at": 0.0, "finished_at": 0.0, "task": None,
            }
            jobs[name] = job
        return job

    @staticmethod
    def _job_public(job: dict) -> dict:
        return {k: v for k, v in job.items() if k != "task"}

    async def _collect_failed(self, scope: str = "") -> list:
        """收集待补嵌项：[(scope_id, mem_id, text), ...]。scope 为空表示全部作用域。"""
        scopes = [scope] if scope else await self._get_all_scopes()
        pending = []
        for sid in scopes:
            async with self._get_memory_lock(sid):
                mems = await self._load_memories_internal(sid)
                for m in mems:
                    if m.get("_embed_failed") and str(m.get("text") or "").strip():
                        pending.append((sid, m.get("id"), m.get("text")))
        return pending

    async def _run_retry_all(self, scope: str):
        """后台批量补嵌：分批调用 embedding，逐批写回并更新进度。"""
        job = self._job()
        try:
            pending = await self._collect_failed(scope)
            job["total"] = len(pending)
            if not pending:
                job["error"] = ""
                return
            for i in range(0, len(pending), self._EMBED_BATCH_SIZE):
                chunk = pending[i:i + self._EMBED_BATCH_SIZE]
                vecs = await self._embed_texts([c[2] for c in chunk], max_retries=1)
                # 按作用域聚合写回，减少加锁次数
                by_scope = {}
                for (sid, mid, text), vec in zip(chunk, vecs):
                    if vec is not None:
                        by_scope.setdefault(sid, []).append((mid, text, vec))
                for sid, items in by_scope.items():
                    async with self._get_memory_lock(sid):
                        mems = await self._load_memories_internal(sid)
                        index = {m.get("id"): m for m in mems}
                        for mid, _text, vec in items:
                            m = index.get(mid)
                            if m is not None:
                                m["embedding"] = vec
                                m.pop("_embed_failed", None)
                        await self._save_memories_internal(sid, mems)
                if self._get_config("chromadb_enabled", False):
                    for sid, items in by_scope.items():
                        for mid, text, vec in items:
                            try:
                                await self._chroma_upsert_memory(
                                    sid, mid, text, vec, "summary", 3, [], time.time(), "",
                                )
                            except Exception:
                                pass
                ok = sum(1 for v in vecs if v is not None)
                job["ok"] += ok
                job["failed"] += (len(chunk) - ok)
                job["done"] = i + len(chunk)
                if i + self._EMBED_BATCH_SIZE < len(pending):
                    await asyncio.sleep(self._EMBED_BATCH_DELAY)
        except Exception:
            job["error"] = "批量补嵌过程中出现异常，详见 AstrBot 日志"
            logger.warning(f"[长期记忆] 批量补嵌异常: {traceback.format_exc()}")
        finally:
            job["running"] = False
            job["finished_at"] = time.time()
            logger.info(
                f"[长期记忆] 批量补嵌结束：成功 {job['ok']} / 失败 {job['failed']} / 共 {job['total']}"
            )

    async def _api_embed_info(self):
        """Embedding 诊断信息：Provider 状态、维度、最近失败原因。"""
        try:
            p = self._get_embedding_provider()
            info = {
                "ready": p is not None,
                "provider_id": self._provider_id_of(p) if p is not None else "",
                "class_name": type(p).__name__ if p is not None else "",
                "dim": None,
                "last_error": getattr(self, "_embed_last_error", "") or "",
                "cache_size": len(getattr(self, "_embed_cache", None) or {}),
                "chromadb": self._chroma_status(),
                "chromadb_enabled": bool(self._get_config("chromadb_enabled", False)),
            }
            if p is not None:
                try:
                    dim_fn = getattr(p, "get_dim", None)
                    if callable(dim_fn):
                        info["dim"] = int(dim_fn())
                except Exception:
                    pass
            return json_response(info)
        except Exception as e:
            logger.warning(f"[长期记忆] Embedding 诊断异常: {e}")
            return error_response(_SAFE_ERROR_MSG, status_code=500)

    async def _api_chroma_reindex(self):
        """把 KV 里已有的向量回填到 ChromaDB 集合。

        背景：ChromaDB 只在**新增记忆**时写入向量。若先前用 NumPy 后端、后来才开启
        chromadb_enabled，历史记忆的向量不在集合里；而检索时 Chroma 一旦返回结果
        就直接使用（不会回退 NumPy），表现为"老记忆检索不到"。此接口做一次性回填。
        """
        try:
            if not self._get_config("chromadb_enabled", False):
                return error_response("未启用 ChromaDB（chromadb_enabled=false）", status_code=400)
            if not getattr(self, "_chroma_client", None):
                return error_response("ChromaDB 未初始化成功，请查看启动日志", status_code=400)
            payload = await self._read_json()
            sid = str(payload.get("scope", "") or "").strip()
            if sid and not _validate_scope_id(sid):
                return error_response("scope 参数非法", status_code=400)
            scopes = [sid] if sid else await self._get_all_scopes()
            done = 0
            total = 0
            skipped = 0
            for s in scopes:
                if not s or not _validate_scope_id(s):
                    skipped += 1
                    continue
                try:
                    mems = await self._load_memories(s)
                except Exception:
                    skipped += 1
                    continue
                n = 0
                for m in mems:
                    emb = m.get("embedding")
                    if not emb:
                        skipped += 1
                        continue
                    try:
                        await self._chroma_upsert_memory(
                            s, m.get("id"), m.get("text", ""), emb,
                            m.get("type", "summary"), m.get("importance", 3),
                            m.get("tags", []) or [], m.get("timestamp", time.time()),
                            m.get("narrative_time", ""),
                        )
                        n += 1
                    except Exception as e:
                        skipped += 1
                        logger.warning(f"[长期记忆] 回填向量失败 ({s}): {e}")
                total += n
                if n:
                    done += 1
            logger.info(f"[长期记忆] ChromaDB 回填完成：{done} 个作用域 / {total} 条向量（跳过 {skipped}）")
            return json_response({"scopes": done, "memories": total, "skipped": skipped})
        except Exception as e:
            logger.warning(f"[长期记忆] 回填向量异常: {e}")
            return error_response(_SAFE_ERROR_MSG, status_code=500)

    async def _api_embed_test(self):
        """对 Embedding Provider 发一次真实探针请求，返回维度与耗时。"""
        try:
            p = self._get_embedding_provider()
            if p is None:
                return error_response(
                    "未找到可用的 Embedding Provider：请在插件配置中指定 embedding_provider_id，"
                    "或确认 AstrBot 已配置嵌入模型",
                    status_code=503,
                )
            started = time.time()
            try:
                res = await asyncio.wait_for(p.get_embeddings(["嵌入连通性测试"]), timeout=30)
            except asyncio.TimeoutError:
                return error_response("Embedding 调用超时（30s），请检查 Provider 网络或额度", status_code=504)
            except Exception as e:
                return error_response(f"Embedding 调用失败：{e}", status_code=502)
            if not res or not res[0]:
                return error_response("Embedding 返回空向量", status_code=502)
            return json_response({
                "success": True,
                "dim": len(res[0]),
                "elapsed_ms": int((time.time() - started) * 1000),
            })
        except Exception as e:
            logger.warning(f"[长期记忆] Embedding 连通性测试异常: {e}")
            return error_response(_SAFE_ERROR_MSG, status_code=500)

    async def _api_retry_all(self):
        """一键补嵌：启动后台任务，立即返回；进度用 embed/status 查询。"""
        try:
            job = self._job()
            if job.get("running"):
                return error_response("已有补嵌任务正在运行，请等待其完成", status_code=409)
            data = await self._read_json()
            scope = str(data.get("scope", "") or "").strip()
            if scope and not _validate_scope_id(scope):
                return error_response("scope 参数非法", status_code=400)
            pending = await self._collect_failed(scope)
            job.update({
                "running": True, "scope": scope, "total": len(pending), "done": 0,
                "ok": 0, "failed": 0, "error": "", "started_at": time.time(),
                "finished_at": 0.0,
            })
            if not pending:
                job["running"] = False
                job["finished_at"] = time.time()
                return json_response({"started": False, "total": 0, "msg": "没有需要补嵌的记忆"})
            job["task"] = asyncio.create_task(self._run_retry_all(scope))
            return json_response({"started": True, "total": len(pending), "scope": scope})
        except Exception as e:
            logger.warning(f"[长期记忆] 启动批量补嵌异常: {e}")
            return error_response(_SAFE_ERROR_MSG, status_code=500)

    async def _api_job_status(self):
        """后台任务进度（embed / profile / reorg）。"""
        try:
            jobs = getattr(self, "_jobs", None) or {}
            out = {name: self._job_public(job) for name, job in jobs.items()}
            out.setdefault("embed", self._job_public(self._job("embed")))
            return json_response({"jobs": out})
        except Exception as e:
            logger.warning(f"[长期记忆] 查询任务进度异常: {e}")
            return error_response(_SAFE_ERROR_MSG, status_code=500)

    async def _api_delete_failed(self):
        """删除（指定作用域或全部）嵌入失败的记忆。"""
        try:
            data = await self._read_json()
            scope = str(data.get("scope", "") or "").strip()
            if scope and not _validate_scope_id(scope):
                return error_response("scope 参数非法", status_code=400)
            if self._job().get("running"):
                return error_response("补嵌任务运行中，请先等待完成再删除", status_code=409)
            scopes = [scope] if scope else await self._get_all_scopes()
            removed = 0
            for sid in scopes:
                gone_ids = []
                async with self._get_memory_lock(sid):
                    mems = await self._load_memories_internal(sid)
                    keep = [m for m in mems if not m.get("_embed_failed")]
                    gone = len(mems) - len(keep)
                    if gone:
                        gone_ids = [m.get("id") for m in mems if m.get("_embed_failed") and m.get("id")]
                        await self._save_memories_internal(sid, keep)
                        removed += gone
                if gone_ids and self._get_config("chromadb_enabled", False):
                    await self._chroma_delete_memories(sid, gone_ids)
            logger.info(f"[长期记忆] 已删除 {removed} 条嵌入失败的记忆")
            return json_response({"removed": removed})
        except Exception as e:
            logger.warning(f"[长期记忆] 删除失败记忆异常: {e}")
            return error_response(_SAFE_ERROR_MSG, status_code=500)

    # ======================== 旧数据调整（编辑 / 重建画像 / 重整记忆） ========================
    async def _api_update_memory(self):
        """编辑单条记忆（文本 / 重要性）；文本变化时重新生成向量。"""
        try:
            data = await self._read_json()
            sid = str(data.get("scope", "") or "").strip()
            mid = str(data.get("id", "") or "").strip()
            if not _validate_scope_id(sid) or not mid:
                return error_response("scope/id 参数非法", status_code=400)

            text = data.get("text")
            if text is not None:
                if not isinstance(text, str):
                    return error_response("text 必须是字符串", status_code=400)
                text = text.strip()
                if not text:
                    return error_response("记忆内容不能为空", status_code=400)
                if len(text) > 2000:
                    return error_response("单条记忆不能超过 2000 字符", status_code=400)
            importance = data.get("importance")
            if importance is not None:
                try:
                    importance = max(1, min(5, int(importance)))
                except (TypeError, ValueError):
                    return error_response("importance 必须是 1-5 的整数", status_code=400)

            async with self._get_memory_lock(sid):
                mems = await self._load_memories_internal(sid)
                target = next((m for m in mems if m.get("id") == mid), None)
                if target is None:
                    return error_response("该记忆不存在（可能已被删除）", status_code=404)
                old_text = target.get("text", "") or ""
                changed_text = text is not None and text != old_text
                if text is not None:
                    target["text"] = text
                if importance is not None:
                    target["importance"] = importance
                if changed_text:
                    target.pop("embedding", None)
                    target["_embed_failed"] = True  # 先标记，下面尝试立即补嵌
                    target["last_edited"] = time.time()
                await self._save_memories_internal(sid, mems)
                new_text = target.get("text", "")
                cur_importance = target.get("importance", 3)

            embedded = False
            if changed_text:
                try:
                    emb = await asyncio.wait_for(
                        self._embed_text(new_text), timeout=_EMBED_RETRY_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    emb = None
                if emb:
                    async with self._get_memory_lock(sid):
                        mems = await self._load_memories_internal(sid)
                        for m in mems:
                            if m.get("id") == mid:
                                m["embedding"] = emb
                                m.pop("_embed_failed", None)
                                await self._save_memories_internal(sid, mems)
                                embedded = True
                                break
                    if embedded and self._get_config("chromadb_enabled", False):
                        try:
                            await self._chroma_upsert_memory(
                                sid, mid, new_text, emb, "summary", cur_importance,
                                [], time.time(), "",
                            )
                        except Exception:
                            pass
                if not embedded:
                    logger.warning(f"[长期记忆] 编辑后补嵌失败，已标记待补嵌 (id={mid[:8]})")
            return json_response({
                "success": True, "id": mid,
                "text_changed": changed_text, "reembedded": embedded,
            })
        except Exception as e:
            logger.warning(f"[长期记忆] 编辑记忆异常: {e}")
            return error_response(_SAFE_ERROR_MSG, status_code=500)

    async def _api_regenerate_profile(self):
        """从现有记忆重建画像（后台任务）。

        用途：修复旧版"保存时就截断"导致的残缺画像——画像由记忆推导而来，
        所以只要记忆还在，就能重新生成一份完整的。
        """
        try:
            data = await self._read_json()
            sid = str(data.get("scope", "") or "").strip()
            if not _validate_scope_id(sid):
                return error_response("scope 参数非法", status_code=400)
            job = self._job("profile")
            if job["running"]:
                return error_response("画像重建正在进行中", status_code=409)
            mems = await self._load_memories(sid)
            if not any(m.get("type") == "summary" and (m.get("text") or "").strip() for m in mems):
                return error_response("该作用域没有可用的总结记忆，无法重建画像", status_code=400)
            job.update({
                "running": True, "scope": sid, "total": 1, "done": 0, "ok": 0,
                "failed": 0, "error": "", "message": "正在从记忆重建画像…",
                "started_at": time.time(), "finished_at": 0.0,
            })
            job["task"] = asyncio.create_task(self._run_regenerate_profile(sid))
            return json_response({"started": True})
        except Exception as e:
            logger.warning(f"[长期记忆] 启动画像重建异常: {e}")
            return error_response(_SAFE_ERROR_MSG, status_code=500)

    async def _run_regenerate_profile(self, scope: str):
        job = self._job("profile")
        try:
            before = await self._load_profile(scope)
            await self._update_profile(scope, None, max_items=_PROFILE_REGEN_ITEMS)
            after = await self._load_profile(scope)
            if after and after != before:
                job["ok"] = 1
                job["message"] = f"画像已重建：{len(before)} → {len(after)} 字符"
            elif after:
                job["ok"] = 1
                job["message"] = f"画像已重建（内容与之前相同，{len(after)} 字符）"
            else:
                job["failed"] = 1
                job["message"] = "未能生成画像，请检查 LLM Provider"
        except Exception as e:
            job["failed"] = 1
            job["error"] = str(e)
            logger.warning(f"[长期记忆] 画像重建失败: {e}")
        finally:
            job["running"] = False
            job["done"] = 1
            job["finished_at"] = time.time()

    async def _api_reorganize(self):
        """重整记忆（合并同话题碎片 / 拆分混合话题），后台执行，可回滚。"""
        try:
            data = await self._read_json()
            sid = str(data.get("scope", "") or "").strip()
            mode = str(data.get("mode", "full") or "full").lower()
            if not _validate_scope_id(sid):
                return error_response("scope 参数非法", status_code=400)
            if mode not in ("full", "resume"):
                return error_response("mode 只能是 full 或 resume", status_code=400)
            for name, j in (getattr(self, "_jobs", None) or {}).items():
                if j.get("running"):
                    return error_response(f"后台任务「{name}」正在运行，请等待完成", status_code=409)

            if mode == "resume":
                pending = await self.get_kv_data(self._KV_REORG_PENDING_PREFIX + sid, None)
                texts = (pending or {}).get("texts") if isinstance(pending, dict) else None
                count = len([t for t in (texts or []) if isinstance(t, str) and t.strip()])
                if count == 0:
                    return error_response("没有待重试的批次", status_code=400)
                start_msg = f"准备重试 {count} 条失败记忆…"
            else:
                mems = await self._load_memories(sid)
                count = sum(1 for m in mems if m.get("type") == "summary" and (m.get("text") or "").strip())
                if count < 2:
                    return error_response("可重整的总结记忆不足 2 条", status_code=400)
                start_msg = f"准备重整 {count} 条记忆…"

            job = self._job("reorg")
            job.update({
                "running": True, "scope": sid, "total": 0, "done": 0, "ok": 0,
                "failed": 0, "error": "", "message": start_msg,
                "started_at": time.time(), "finished_at": 0.0,
            })
            job["task"] = asyncio.create_task(self._run_reorganize(sid, job, mode == "resume"))
            return json_response({"started": True, "count": count, "mode": mode})
        except Exception as e:
            logger.warning(f"[长期记忆] 启动记忆重整异常: {e}")
            return error_response(_SAFE_ERROR_MSG, status_code=500)

    async def _run_reorganize(self, scope: str, job: dict, resume: bool = False):
        try:
            res = await self._reorganize_memories(scope, job, resume=resume)
            if res.get("ok"):
                job["ok"] = 1
                failed = res.get("failed_batches") or 0
                job["message"] = (
                    f"重整完成：{res['before']} 条 → {res['after']} 条"
                    + (f"；{failed} 批失败（可点「重试失败批次」）" if failed else "")
                )
                job["failed_batches"] = failed
            else:
                job["failed"] = 1
                job["message"] = job.get("message") or "重整未生效，原数据已保留"
        except Exception as e:
            job["failed"] = 1
            job["error"] = str(e)
            logger.warning(f"[长期记忆] 记忆重整失败: {e}")
        finally:
            job["running"] = False
            job["finished_at"] = time.time()

    async def _api_export_backup(self):
        """导出全部作用域的记忆/画像/亲密度为 JSON。

        重要性：AstrBot 卸载插件时会调用 sp.clear_async("plugin", plugin_id)
        **清空该插件的全部 KV**（记忆就存在 KV 里）。导出的文件在用户自己的磁盘上，
        卸载/重装插件都不会丢，是唯一可靠的自保手段。
        """
        try:
            scopes = await self._get_all_scopes()
            data = {}
            total = 0
            truncated = False
            for sid in scopes[:_EXPORT_MAX_SCOPES]:
                if not _validate_scope_id(sid):
                    continue
                try:
                    mems = await self._load_memories(sid)
                except Exception:
                    continue
                if not mems:
                    continue
                if total + len(mems) > _EXPORT_MAX_MEMORIES:
                    truncated = True
                    logger.warning(
                        f"[长期记忆] 导出超过 {_EXPORT_MAX_MEMORIES} 条上限，已截断；请分作用域多次导出"
                    )
                    break
                prof = ""
                aff = {}
                try:
                    prof = await self._load_profile(sid) or ""
                except Exception:
                    pass
                try:
                    a = await self._load_affection(sid)
                    aff = {k: a.get(k) for k in (
                        "score", "level", "level_cn", "total_interactions",
                        "positive_count", "negative_count", "history",
                    )}
                except Exception:
                    pass
                data[sid] = {"memories": mems, "profile": prof, "affection": aff}
                total += len(mems)
            return json_response({
                "format": "astrbot_plugin_longterm_memory.backup",
                "format_version": 1,
                "exported_at": time.time(),
                "scope_count": len(data),
                "memory_count": total,
                "truncated": truncated,
                "scopes": data,
            })
        except Exception as e:
            logger.warning(f"[长期记忆] 导出备份异常: {e}")
            return error_response(_SAFE_ERROR_MSG, status_code=500)

    @staticmethod
    def _sanitize_memory_entry(m) -> dict:
        """校验并规范化导入的单条记忆；无法修复时返回空 dict（调用方丢弃）。

        导入的是外部 JSON，不能直接信任：字段类型可能错、文本可能超长、
        时间戳可能是字符串、向量可能是残缺数组。这里做白名单收敛，
        避免把脏数据写进 KV 后污染检索与统计。
        """
        if not isinstance(m, dict):
            return {}
        text = m.get("text")
        if not isinstance(text, str):
            return {}
        text = text.strip()
        if not text or len(text) > _MAX_IMPORT_TEXT_LEN:
            return {}

        typ = m.get("type")
        typ = typ if typ in ("summary", "raw") else "summary"
        try:
            imp = max(1, min(5, int(m.get("importance", 3))))
        except (TypeError, ValueError):
            imp = 3
        try:
            ts = float(m.get("timestamp"))
        except (TypeError, ValueError):
            ts = time.time()

        tags = m.get("tags")
        tags = [str(x) for x in tags if isinstance(x, str)] if isinstance(tags, list) else []
        dialog_ids = m.get("dialog_ids")
        dialog_ids = ([str(x) for x in dialog_ids if isinstance(x, str)]
                      if isinstance(dialog_ids, list) else [])
        nt = m.get("narrative_time")
        mid = m.get("id")

        out = {
            "id": mid if isinstance(mid, str) and mid else str(uuid.uuid4()),
            "text": text,
            "type": typ,
            "timestamp": ts,
            "importance": imp,
            "tags": tags,
            "dialog_ids": dialog_ids,
            "narrative_time": nt if isinstance(nt, str) else "",
        }
        emb = m.get("embedding")
        if (isinstance(emb, list) and emb
                and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in emb)):
            out["embedding"] = [float(x) for x in emb]
        else:
            # 没有可用向量 → 标记待补嵌，之后可用「一键补嵌」修复
            out["_embed_failed"] = True
        return out

    async def _api_import_backup(self):
        """从导出的 JSON 恢复数据。导入前对每个受影响作用域自动快照。"""
        try:
            payload = await self._read_json()
            blob = payload.get("data") if isinstance(payload.get("data"), dict) else payload
            mode = str(payload.get("mode", "merge") or "merge").lower()
            if mode not in ("merge", "replace"):
                return error_response("mode 只能是 merge 或 replace", status_code=400)
            if not isinstance(blob, dict) or blob.get("format") != "astrbot_plugin_longterm_memory.backup":
                return error_response("不是本插件导出的备份文件", status_code=400)
            scopes = blob.get("scopes")
            if not isinstance(scopes, dict) or not scopes:
                return error_response("备份里没有任何作用域数据", status_code=400)

            done = 0
            mem_total = 0
            skipped_total = 0
            for sid, item in scopes.items():
                if not _validate_scope_id(sid) or not isinstance(item, dict):
                    skipped_total += 1
                    continue
                mems = item.get("memories")
                if not isinstance(mems, list):
                    skipped_total += 1
                    continue
                # 白名单收敛：非法条目丢弃并计数，不写进 KV
                cleaned = []
                for m in mems:
                    c = self._sanitize_memory_entry(m)
                    if c:
                        cleaned.append(c)
                    else:
                        skipped_total += 1
                if not cleaned:
                    continue
                await self._backup_memories_snapshot(sid, "导入前自动备份")
                if mode == "replace":
                    await self._save_memories(sid, cleaned)
                else:
                    cur = await self._load_memories(sid)
                    existing = {(m.get("text") or "").strip() for m in cur}
                    add = []
                    for m in cleaned:
                        t = m["text"]
                        if t in existing:
                            continue
                        existing.add(t)
                        add.append(m)
                    await self._save_memories(sid, cur + add)
                mem_total += len(cleaned)
                prof = item.get("profile")
                if isinstance(prof, str) and prof.strip():
                    if mode == "replace":
                        await self._save_profile(sid, prof)
                    elif not (await self._load_profile(sid)):
                        await self._save_profile(sid, prof)
                done += 1

            logger.info(
                f"[长期记忆] 已导入备份：{done} 个作用域 / {mem_total} 条记忆（mode={mode}）"
                + (f"，跳过非法条目 {skipped_total} 条" if skipped_total else "")
            )
            return json_response({
                "scopes": done, "memories": mem_total,
                "skipped": skipped_total, "mode": mode,
            })
        except Exception as e:
            logger.warning(f"[长期记忆] 导入备份异常: {e}")
            return error_response(_SAFE_ERROR_MSG, status_code=500)

    async def _api_migrate_memories(self):
        """把记忆从一个作用域迁移/复制到另一个作用域。

        典型用途：从 user 作用域切到 conversation 作用域后，把历史记忆搬到某个对话。
        注意：记忆条目本身**不记录来源对话**（只有 dialog_ids，不含 cid），
        因此只能整体迁移到指定作用域，无法自动还原到"原来那个对话"。
        """
        try:
            payload = await self._read_json()
            src = str(payload.get("from", "") or "").strip()
            dst = str(payload.get("to", "") or "").strip()
            mode = str(payload.get("mode", "move") or "move").strip().lower()
            if not _validate_scope_id(src) or not _validate_scope_id(dst):
                return error_response("作用域参数非法", status_code=400)
            if src == dst:
                return error_response("源与目标作用域相同", status_code=400)
            if mode not in ("move", "copy"):
                return error_response("mode 只能是 move 或 copy", status_code=400)

            src_mems = await self._load_memories(src)
            if not src_mems:
                return error_response("源作用域没有记忆", status_code=400)
            dst_mems = await self._load_memories(dst)
            dst_before = [dict(m) for m in dst_mems]  # 备份用：迁移前的目标状态

            existing = {(m.get("text") or "").strip() for m in dst_mems}
            added = 0
            for m in src_mems:
                text = (m.get("text") or "").strip()
                if not text or text in existing:
                    continue
                item = dict(m)
                item["id"] = str(uuid.uuid4())  # 重新分配，避免与目标已有 id 冲突
                dst_mems.append(item)
                existing.add(text)
                added += 1

            if added:
                # 迁移前备份目标作用域，便于回滚
                try:
                    await self.put_kv_data(self._KV_MEMORIES_BACKUP_PREFIX + dst, {
                        "timestamp": time.time(),
                        "memories": dst_before,
                    })
                except Exception as e:
                    logger.warning(f"[长期记忆] 迁移前备份失败，已中止: {e}")
                    return error_response("备份目标作用域失败，已中止迁移", status_code=500)
                await self._save_memories(dst, dst_mems)

                # 可选向量后端：把有向量的记忆同步进目标集合
                if self._get_config("chromadb_enabled", False):
                    for m in dst_mems[-added:]:
                        emb = m.get("embedding")
                        if not emb:
                            continue
                        try:
                            await self._chroma_upsert_memory(
                                dst, m.get("id"), m.get("text", ""), emb,
                                m.get("type", "summary"), m.get("importance", 3),
                                m.get("tags", []) or [], m.get("timestamp", time.time()),
                                m.get("narrative_time", ""),
                            )
                        except Exception:
                            pass

            if mode == "move" and added:
                # 移动会清空源作用域——必须先给源也留一份快照，
                # 否则"迁移到错误目标"就没有回头路。
                await self._backup_memories_snapshot(src, f"迁移到 {dst}")
                await self._save_memories(src, [])

            logger.info(
                f"[长期记忆] 记忆{mode == 'move' and '迁移' or '复制'} ({src} → {dst}): "
                f"新增 {added} 条，跳过重复 {len(src_mems) - added} 条"
            )
            return json_response({
                "added": added,
                "skipped": len(src_mems) - added,
                "mode": mode,
                "from": src,
                "to": dst,
            })
        except Exception as e:
            logger.warning(f"[长期记忆] 记忆迁移异常: {e}")
            return error_response(_SAFE_ERROR_MSG, status_code=500)

    async def _api_restore_backup(self):
        """回滚到最近一次记忆重整前的备份。"""
        try:
            data = await self._read_json()
            sid = str(data.get("scope", "") or "").strip()
            if not _validate_scope_id(sid):
                return error_response("scope 参数非法", status_code=400)
            backup = await self.get_kv_data(self._KV_MEMORIES_BACKUP_PREFIX + sid, None)
            if not isinstance(backup, dict) or not isinstance(backup.get("memories"), list):
                return error_response("没有可用的重整备份", status_code=404)
            await self._save_memories(sid, backup["memories"])
            # 刻意**不删**备份：误操作后可能需要多次回滚，保留一份兜底更安全
            logger.info(f"[长期记忆] 已从备份恢复 ({sid})，备份保留")
            logger.info(f"[长期记忆] 已回滚记忆重整 ({sid}): 恢复 {len(backup['memories'])} 条")
            return json_response({
                "success": True,
                "restored": len(backup["memories"]),
                "backup_time": backup.get("timestamp", 0),
            })
        except Exception as e:
            logger.warning(f"[长期记忆] 回滚失败: {e}")
            return error_response(_SAFE_ERROR_MSG, status_code=500)
