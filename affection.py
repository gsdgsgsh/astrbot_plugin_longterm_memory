"""好感度模块：数据存取、等级计算、规则引擎、LLM 分析、上下文注入。"""
import asyncio
import json
import time
import re
from typing import Optional

from astrbot.api import logger


# ========== 等级定义 ==========
# 分数范围 [-100, 100]，8 个等级（硬编码默认值，可被 affection_level_thresholds 配置覆盖）
_AFFECTION_LEVEL_ORDER = [
    ("nemesis",      "仇敌", "极度谨慎、保持防御，避免激怒对方"),
    ("hostile",      "敌对", "礼貌疏离，不主动引发冲突"),
    ("cold",         "冷淡", "中立偏冷，简短回应、不展开话题"),
    ("stranger",     "陌生人", "友好中立，保持基本礼貌"),
    ("acquaintance", "熟人", "温暖友好，适度互动"),
    ("friend",       "朋友", "亲切热情，可适度幽默"),
    ("close_friend", "好友", "亲密信任，可用内部梗、昵称"),
    ("confidant",    "挚友", "高度信任、深度共情、无话不谈"),
]
_DEFAULT_THRESHOLD_KEYS = ["nemesis_max", "hostile_max", "cold_max", "stranger_max",
                           "acquaintance_max", "friend_max", "close_friend_max"]


def _build_levels(thresholds: dict = None) -> list:
    """根据阈值 dict 构建等级区间列表，每项 (eng, chn, lo, hi, tone)。"""
    if thresholds and isinstance(thresholds, dict):
        try:
            # 从配置中读取各等级上限
            limits = [int(thresholds.get(k, 0)) for k in _DEFAULT_THRESHOLD_KEYS]
            # 构建区间：(-100, nemesis_max], (nemesis_max+1, hostile_max], ...
            levels = []
            lo = -100
            for i, (eng, chn, tone) in enumerate(_AFFECTION_LEVEL_ORDER):
                if i < len(limits):
                    hi = limits[i]
                else:
                    hi = 100  # confidant
                levels.append((eng, chn, lo, hi, tone))
                lo = hi + 1
            return levels
        except (ValueError, TypeError):
            pass
    # 回退到硬编码默认值
    return [
        ("nemesis",      "仇敌", -100, -71, "极度谨慎、保持防御，避免激怒对方"),
        ("hostile",      "敌对",  -70, -41, "礼貌疏离，不主动引发冲突"),
        ("cold",         "冷淡",  -40, -11, "中立偏冷，简短回应、不展开话题"),
        ("stranger",     "陌生人", -10,  10, "友好中立，保持基本礼貌"),
        ("acquaintance", "熟人",   11,  30, "温暖友好，适度互动"),
        ("friend",       "朋友",   31,  55, "亲切热情，可适度幽默"),
        ("close_friend", "好友",   56,  75, "亲密信任，可用内部梗、昵称"),
        ("confidant",    "挚友",   76, 100, "高度信任、深度共情、无话不谈"),
    ]


def compute_level(score: float, thresholds: dict = None) -> dict:
    """根据分数计算等级信息。可传入自定义阈值 dict（来自 affection_level_thresholds 配置）。"""
    levels = _build_levels(thresholds)
    for eng, chn, lo, hi, tone in levels:
        if lo <= score <= hi:
            return {"level": eng, "level_cn": chn, "tone_hint": tone}
    # 兜底
    if score < -100:
        return {"level": "nemesis", "level_cn": "仇敌", "tone_hint": levels[0][4]}
    return {"level": "confidant", "level_cn": "挚友", "tone_hint": levels[-1][4]}


# ========== 规则引擎（关键词匹配，作为 LLM 不可用时的兜底） ==========
# 设计原则：
#   1. **只分析用户自己说的话**。好感度记录的是"用户对机器人的态度"，
#      机器人说了什么与该分数无关（旧版把双方消息拼在一起匹配关键词，
#      机器人说"不对/算了"会误扣用户的分，已修正）。
#   2. **回复长度不是态度信号**。只回「嗯」「哦」「好」不代表敷衍，
#      可能是没意见、在思考、或不想多打字（旧版对 <5 字扣分，已移除）。
#   3. 只保留**明确**的情感表达，含糊词（随便/无所谓/行吧）不计分。
# 评分取向（v2.9.4 重构）：
#   这个分数衡量的是「关系有多近（亲密度）」，而不是「用户有多捧我」。
#   因此把信号分成两类，并刻意让"讨好机器人"成为低收益路径：
#
#   A. 亲密度信号（权重高）—— 真正代表关系在变近的行为：
#      · 自我表露：分享自己的经历/感受/计划/困扰（信任的表现）
#      · 关系连续性：引用共同历史（"上次你说的" "还记得"）
#      · 深度投入：长篇、有实质内容的交流
#   B. 礼貌信号（权重低）—— 社交润滑，有暖意但不代表关系变近：
#      · 谢谢/辛苦了、夸奖、哈哈哈、喜欢你
#      这类表达**权重刻意压低**，且连续出现时收益递减（见 _apply_affection_delta），
#      所以"靠刷夸奖涨分"是走不通的。
#
#   负向只保留**明确的敌意**；对内容的批评（"答非所问" "这个不对"）不算负向——
#   指出错误是正常交流，不应该扣分，更不应该让用户觉得"必须顺着 AI 说话"。
_SHARING_PATTERNS = [
    # 自我表露：第一人称 + 时间/心理/计划类动词
    (re.compile(
        r"我(?:今天|昨天|明天|最近|现在|刚才|觉得|感觉|想|打算|准备|决定|希望|"
        r"担心|害怕|难过|开心|高兴|烦|累|遇到|碰到|正在|要去|在做|买了|做了|吃了)"
    ), 0.25),
    # 关系连续性：引用双方共同的历史
    (re.compile(r"上次|之前你(?:说|提)|还记得|我们说(?:过|的)|你上次|前面你(?:说|提)"), 0.4),
]
_COURTESY_PATTERNS = [
    (re.compile(r"爱你|喜欢你|爱了|好喜欢你"), 0.5),
    (re.compile(r"谢谢|感谢|多谢|辛苦了|麻烦你了|帮大忙"), 0.2),
    (re.compile(r"太棒了|厉害|靠谱|真棒|优秀|赞"), 0.15),
    (re.compile(r"哈哈|笑死|😂|🤣|hhh|哈哈哈"), 0.15),
]
_NEGATIVE_PATTERNS = [
    (re.compile(r"垃圾|废物|没用|差劲|恶心|弱智|傻逼|蠢|滚"), -1.0),
    (re.compile(r"讨厌|烦死|闭嘴|气死|受够了|受不了"), -0.5),
    (re.compile(r"算了|不用了"), -0.1),  # 取消请求，属很轻的负向
]

# 明确敌意措辞：只有命中它，才允许对该轮做出**较大幅度**的扣分。
# 用于抑制"用户话少 / 被助手的烂回复整无语 / 指出错误"被误判为负向。
# 注意：刻意**不包含**"答非所问/不对/错了/不好"这类对内容的批评。
_STRONG_NEGATIVE_RE = re.compile(
    r"垃圾|废物|没用|差劲|恶心|弱智|傻逼|蠢|滚|讨厌|烦死|闭嘴|气死|"
    r"失望|生气|受够了|受不了|别烦我|不想说"
)


# ===== 历史重算的估算参数 =====
# 旧记录只保存了"最终落地 delta"，没有保存原始信号强度，因此无法精确重放历史。
# 这里按互动次数与正负比例做估算——定位是"给一个合理的起点供用户确认"。
_EST_RAW_POSITIVE = 1.0    # 典型一轮正向：LLM 评估多给 +0.5~+2
_EST_RAW_NEGATIVE = -1.0   # 典型一轮负向


def estimate_score_from_stats(data: dict, alpha: float, divisor: float) -> float:
    """按历史互动统计，估算「若一开始就用当前参数」应得的分数。

    做法：按 total_interactions 步逐轮重放（正向用 _EST_RAW_POSITIVE、
    负向用 _EST_RAW_NEGATIVE、其余视为中性），每步套用当前增益系数与难度曲线。
    未计入每日上限，因此结果可能**偏高**，仅作参考。
    """
    try:
        total = max(0, int(data.get("total_interactions", 0) or 0))
        pos = max(0, int(data.get("positive_count", 0) or 0))
        neg = max(0, int(data.get("negative_count", 0) or 0))
    except (TypeError, ValueError):
        return 0.0
    if total <= 0:
        return 0.0
    pos = min(pos, total)
    neg = min(neg, max(0, total - pos))
    neutral = max(0, total - pos - neg)

    steps = ([_EST_RAW_POSITIVE] * pos + [0.0] * neutral + [_EST_RAW_NEGATIVE] * neg)
    score = 0.0
    div = divisor if divisor and divisor > 0 else 200.0
    for raw in steps:
        if raw == 0.0:
            continue
        difficulty = max(0.1, 1.0 - max(0.0, score if raw > 0 else -score) / div)
        score += alpha * raw * difficulty
        score = max(-100.0, min(100.0, score))
    return round(score, 1)


def has_explicit_negative(text: str) -> bool:
    """文本中是否存在明确的负面表达（用于负向保护）。"""
    return bool(text) and bool(_STRONG_NEGATIVE_RE.search(text))


def rule_based_delta(user_msg: str, assistant_msg: str) -> tuple[float, str]:
    """规则引擎：基于关键词匹配估算好感度变化。
    返回 (delta, reason)。

    注意：assistant_msg 参数保留仅为签名兼容，**不参与打分**。
    """
    text = (user_msg or "").strip()
    delta = 0.0
    reason_parts = []

    # 先判明确敌意：命中就不再叠加正向信号（"垃圾但谢谢你" 应视为负向）
    for pat, val in _NEGATIVE_PATTERNS:
        if pat.search(text):
            delta += val
            reason_parts.append(f"负面表达({val})")
            break

    if delta >= 0:
        # A. 亲密度信号（自我表露 / 关系连续性）—— 权重最高，是"涨分"的主要途径
        for pat, val in _SHARING_PATTERNS:
            if pat.search(text):
                delta += val
                reason_parts.append(f"自我表露/关系连续性(+{val})")
                break
        # B. 礼貌信号 —— 权重低；靠它刷分的效率天生很低（且有递减，见 _apply_affection_delta）
        for pat, val in _COURTESY_PATTERNS:
            if pat.search(text):
                delta += val
                reason_parts.append(f"礼貌/暖意(+{val})")
                break
        # C. 深度投入：有实质内容的长消息（不给短消息扣分，只是长消息略有加成）
        if len(text) >= 80:
            delta += 0.1
            reason_parts.append("深度投入(+0.1)")
        delta = min(delta, 0.8)  # 单轮正向封顶，避免任何单一话术刷分

    reason = "; ".join(reason_parts) if reason_parts else "无明显情感倾向"
    # 保留 2 位小数：0.15 / 0.25 这类细粒度权重若只保留 1 位会被舍入抹平
    return round(delta, 2), reason


# 默认好感度分析提示词。
# 关键点：明确"简短回复 / 被烂回复整无语"都不算负面，避免误扣分。
_DEFAULT_ANALYSIS_PROMPT = """请评估用户与助手之间的**关系亲密度**变化（注意：不是"用户有多捧助手"）。
输出 JSON：
{"delta": 数值(-3到3), "reason": "简短原因(15字内)"}

这个分数衡量的是「关系有多近」。请按下面的取向打分：

【加分：关系在变近的行为】
+0.5~+1.5  用户主动分享自己的经历、感受、计划、困扰（自我表露 = 信任）
+0.5~+1.0  用户引用双方共同的历史（"上次你说的""还记得"），说明关系有连续性
+0.3~+0.8  有实质内容的深入交流，或主动追问、延展话题
+0.1~+0.3  礼貌与暖意（谢谢、辛苦了、夸奖、哈哈哈）——**只是社交润滑，不要给高分**

【0 分：正常交流，一律不扣分】
- 用户回复简短（「嗯」「哦」「好的」「行」「?」「…」）——可能只是没意见或在忙
- 用户提出不同意见、指出助手答错、纠正助手——这是正常且健康的交流
- 用户因**助手的回复不佳**而无话可说——责任不在用户
- 普通问答、没有明显情感倾向

【扣分：仅限明确的敌意】
-0.5~-1.5  用户明确表达厌烦/失望/被冒犯（如"闭嘴""别烦我""受够了"）
-2~-3      用户辱骂、人身攻击

【重要】
- 判断依据是用户**实际表达的内容**，不是回复长度、字数或礼貌程度
- 不要因为用户没夸奖助手就给低分；也不要因为用户夸奖就大幅加分
- 没有把握时一律给 0，宁可漏判也不要误扣

对话内容：
{对话历史}

请输出 JSON："""


class AffectionMixin:
    """好感度系统：评估、存取、等级计算、上下文注入。"""

    # ========== 存储 ==========
    async def _load_affection(self, scope_id: str) -> dict:
        """加载好感度数据，不存在时返回默认值。"""
        cache = getattr(self, "_affection_cache", None)
        if cache is None:
            self._affection_cache = {}
            cache = self._affection_cache
        if scope_id in cache:
            return dict(cache[scope_id])
        key = self._KV_AFFECTION_PREFIX + scope_id
        data = await self.get_kv_data(key, None)
        if data is None or not isinstance(data, dict):
            default = self._default_affection()
            self._cache_put(cache, scope_id, default)
            return dict(default)
        self._cache_put(cache, scope_id, data)
        return dict(data)

    async def _save_affection(self, scope_id: str, data: dict):
        """保存好感度数据。"""
        key = self._KV_AFFECTION_PREFIX + scope_id
        await self.put_kv_data(key, data)
        cache = getattr(self, "_affection_cache", None)
        if cache is None:
            self._affection_cache = {}
            cache = self._affection_cache
        self._cache_put(cache, scope_id, dict(data))
        await self._register_scope(scope_id)

    def _default_affection(self) -> dict:
        return {
            "score": 0.0,
            "level": "stranger",
            "level_cn": "陌生人",
            "total_interactions": 0,
            "positive_count": 0,
            "negative_count": 0,
            "last_update": 0.0,
            "last_eval_time": 0.0,
            "_rule_msg_count": 0,
            "history": [],
            "traits": {},
        }

    # ========== 每日变化追踪 ==========
    async def _load_daily_delta(self, scope_id: str) -> float:
        key = self._KV_AFFECTION_DAILY_PREFIX + scope_id
        val = await self.get_kv_data(key, None)
        if isinstance(val, dict):
            day = val.get("day", "")
            today = time.strftime("%Y-%m-%d")
            if day == today:
                return float(val.get("delta", 0.0))
        return 0.0

    async def _save_daily_delta(self, scope_id: str, delta: float):
        key = self._KV_AFFECTION_DAILY_PREFIX + scope_id
        today = time.strftime("%Y-%m-%d")
        await self.put_kv_data(key, {"day": today, "delta": delta})

    # ========== LLM 重新评估（按既有互动修正分数） ==========
    _REEVAL_TEMPLATE = """你在评估「用户」与「助手」之间的关系亲密度（注意：不是用户有多捧助手）。

【当前数据】（在旧规则下累积，可能偏低，仅供参考，不要被它束缚）
当前分数：{当前分数}（{当前等级}）
累计互动 {互动次数} 次：正向 {正向}、负向 {负向}

【最近的变化记录】
{变化记录}

【用户相关的记忆摘要】
{记忆摘要}

【等级标准】
-100~-71 仇敌 ｜ -70~-41 敌对 ｜ -40~-11 冷淡 ｜ -10~10 陌生人
11~30 熟人 ｜ 31~55 朋友 ｜ 56~75 好友 ｜ 76~100 挚友

【判断要求】
1. 依据**实际互动内容**判断：交流是否长期、深入、稳定；用户是否分享自己的生活、感受与计划；是否引用过共同经历
2. **礼貌用语（谢谢、夸奖）不能作为高亲密度的依据**
3. 互动次数多但内容很浅 → 属于「熟人」，不是「朋友」
4. 互动次数很少 → 不要给高分
5. 负面互动明显（争吵、被冒犯）应体现为低分或负分
6. 分数必须与上面的等级标准一致

输出 JSON：
{"score": 数值(-100到100), "reason": "简短理由(不超过30字)"}"""

    async def _reevaluate_affection(self, scope_id: str, job: dict = None) -> dict:
        """让 LLM 依据既有互动记录，重新判定关系亲密度并落库。

        与手动设定不同：这里由模型给出判断，理由写入历史便于回溯。
        属于「修正」而非「增长」，因此**不受每日上限限制**。
        返回 {"ok": bool, "old_score": float, "score": float, "level_cn": str, "reason": str}。
        """
        def _set(**kw):
            if isinstance(job, dict):
                job.update(kw)

        data = await self._load_affection(scope_id)
        old_score = float(data.get("score", 0.0) or 0.0)
        thresholds = self._get_config("affection_level_thresholds", None)
        old_level_cn = compute_level(old_score, thresholds).get("level_cn", "")

        # 变化记录
        hist_lines = []
        for h in (data.get("history") or [])[-20:]:
            try:
                ts = time.strftime("%m-%d", time.localtime(float(h.get("timestamp", 0) or 0)))
            except (ValueError, OSError, TypeError):
                ts = "??-??"
            try:
                dv = float(h.get("delta", 0) or 0)
            except (TypeError, ValueError):
                dv = 0.0
            hist_lines.append(f"{ts}  {dv:+.1f}  {str(h.get('reason', ''))[:40]}")
        # 记忆摘要（反映互动内容）
        mem_lines = []
        try:
            mems = await self._load_memories(scope_id)
            texts = [m.get("text", "") for m in mems
                     if m.get("type") == "summary" and m.get("text")]
            mem_lines = [t[:120] for t in texts[-25:]]
        except Exception:
            pass

        prompt = (self._REEVAL_TEMPLATE
                  .replace("{当前分数}", f"{old_score:.1f}")
                  .replace("{当前等级}", old_level_cn or "未知")
                  .replace("{互动次数}", str(data.get("total_interactions", 0)))
                  .replace("{正向}", str(data.get("positive_count", 0)))
                  .replace("{负向}", str(data.get("negative_count", 0)))
                  .replace("{变化记录}", "\n".join(hist_lines) if hist_lines else "（暂无）")
                  .replace("{记忆摘要}", "\n".join(f"- {t}" for t in mem_lines) if mem_lines else "（暂无记忆）"))

        provider_id = await self._get_llm_provider_id(scope_id)
        if not provider_id:
            _set(error="无可用 LLM")
            return {"ok": False, "old_score": old_score}

        timeout = self._get_config("llm_timeout_seconds", 60)
        _set(message="AI 正在评估既有互动…")
        try:
            resp = await asyncio.wait_for(
                self.context.llm_generate(chat_provider_id=provider_id, prompt=prompt),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            _set(error="LLM 评估超时")
            return {"ok": False, "old_score": old_score}
        except Exception as e:
            # 保证返回值契约一致：任何 LLM 调用失败都返回 ok=False，由调用方决定如何提示
            _set(error=f"LLM 调用失败：{e}")
            logger.warning(f"[好感度] 重新评估调用 LLM 失败: {e}")
            return {"ok": False, "old_score": old_score}

        text = (resp.completion_text or "").strip()
        parsed = None
        m = re.search(r'\{[^{}]*"score"[^{}]*\}', text, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group())
            except (json.JSONDecodeError, ValueError, TypeError):
                parsed = None
        if parsed is None:
            try:
                parsed = json.loads(text)
            except (json.JSONDecodeError, ValueError, TypeError):
                parsed = None
        if not isinstance(parsed, dict) or "score" not in parsed:
            _set(error="LLM 未返回可解析的分数")
            logger.warning(f"[好感度] 重新评估解析失败: {text[:120]}")
            return {"ok": False, "old_score": old_score}

        try:
            score = float(parsed.get("score"))
        except (TypeError, ValueError):
            _set(error="LLM 返回的分数不是数字")
            return {"ok": False, "old_score": old_score}
        score = max(-100.0, min(100.0, score))
        reason = str(parsed.get("reason", "") or "").strip()[:60]

        lock = self._get_affection_lock(scope_id)
        async with lock:
            aff = await self._load_affection(scope_id)
            base = float(aff.get("score", 0.0) or 0.0)
            info = compute_level(score, thresholds)
            aff["score"] = round(score, 1)
            aff["level"] = info.get("level", "stranger")
            aff["level_cn"] = info.get("level_cn", "陌生人")
            aff["last_update"] = time.time()
            history = aff.get("history", [])
            history.append({
                "delta": round(score - base, 2),
                "reason": ("AI 重新评估：" + reason) if reason else "AI 重新评估",
                "timestamp": time.time(),
                "score_after": round(score, 1),
                "level_after": info.get("level_cn", ""),
            })
            aff["history"] = history[-30:]
            await self._save_affection(scope_id, aff)

        logger.info(
            f"[好感度] AI 重新评估 ({scope_id}): {base:.1f} → {score:.1f}"
            f"（{info.get('level_cn', '')}）| {reason}"
        )
        return {
            "ok": True, "old_score": round(base, 1), "score": round(score, 1),
            "level_cn": info.get("level_cn", ""), "reason": reason,
        }

    # ========== 核心：应用好感度变化 ==========
    async def _apply_affection_delta(
        self, scope_id: str, raw_delta: float, reason: str = ""
    ) -> Optional[dict]:
        """应用好感度变化（含平滑、日上限、裁剪），返回更新后的数据。"""
        if not self._get_config("affection_enabled", False):
            return None

        lock = self._get_affection_lock(scope_id)
        async with lock:
            data = await self._load_affection(scope_id)

            # 1. 日上限检查
            daily_cap = self._get_config("affection_daily_cap", 30.0)  # 与 schema 默认值保持一致
            daily_sofar = await self._load_daily_delta(scope_id)
            remaining = daily_cap - abs(daily_sofar)
            if remaining <= 0:
                logger.info(f"[好感度] 日变化已达上限 ({daily_cap})，跳过")
                return data

            capped_delta = max(-remaining, min(remaining, raw_delta))

            # 2. 递减收益（反"刷好感度"）
            #    同一个加分理由反复出现时收益对折衰减：第 1 次 ×1、第 2 次 ×0.5、
            #    第 3 次 ×0.25、其后维持 ×0.25。这样"连续说谢谢/夸奖"无法刷分，
            #    用户也就不需要为了涨分去讨好机器人。
            if capped_delta > 0:
                recent = list(data.get("recent_positive_reasons") or [])
                repeats = recent.count(reason) if reason else 0
                if repeats:
                    factor = max(0.25, 0.5 ** repeats)
                    capped_delta *= factor
                    logger.debug(
                        f"[好感度] 加分理由「{reason}」近期已出现 {repeats} 次，收益 ×{factor:.2f}"
                    )
                recent.append(reason)
                data["recent_positive_reasons"] = recent[-6:]

            # 3. 难度系数：边际递减，高分难涨、低分难跌
            #    公式: difficulty = max(0.1, 1 - |score| / divisor)
            #    score=0 → 1.0 全速; score=±60 → 0.5; score=±100 → 0.17
            divisor = self._get_config("affection_difficulty_divisor", 200.0)
            old_score = data.get("score", 0.0)
            if capped_delta > 0:
                difficulty = max(0.1, 1.0 - max(0.0, old_score) / divisor)
            elif capped_delta < 0:
                difficulty = max(0.1, 1.0 - max(0.0, -old_score) / divisor)
            else:
                difficulty = 1.0
            capped_delta *= difficulty

            # 4. 增益系数（历史命名 alpha；数学上等价于"每步实际落地的增量比例"）
            #    展开后 new_score = old_score + alpha × delta × difficulty
            alpha = self._get_config("affection_ema_alpha", 0.4)
            target = old_score + capped_delta
            new_score = alpha * target + (1 - alpha) * old_score

            # 5. 裁剪到 [-100, 100]
            new_score = max(-100.0, min(100.0, new_score))
            actual_delta = round(new_score - old_score, 2)

            # 5. 更新数据
            thresholds = self._get_config("affection_level_thresholds", None)
            level_info = compute_level(new_score, thresholds)
            data["score"] = round(new_score, 1)
            data["level"] = level_info["level"]
            data["level_cn"] = level_info["level_cn"]
            data["total_interactions"] = data.get("total_interactions", 0) + 1
            data["last_update"] = time.time()

            if actual_delta > 0:
                data["positive_count"] = data.get("positive_count", 0) + 1
            elif actual_delta < 0:
                data["negative_count"] = data.get("negative_count", 0) + 1

            # 6. 记录历史（保留最近 30 条）
            history = data.get("history", [])
            history.append({
                "delta": actual_delta,
                "reason": reason or "规则评估",
                "timestamp": time.time(),
                "score_after": round(new_score, 1),
                "level_after": level_info["level_cn"],
            })
            if len(history) > 30:
                history = history[-30:]
            data["history"] = history

            await self._save_affection(scope_id, data)
            new_daily = daily_sofar + abs(actual_delta)
            await self._save_daily_delta(scope_id, new_daily)

            logger.info(
                f"[好感度] {scope_id}: {old_score:.1f}→{new_score:.1f} "
                f"(Δ={actual_delta:+.2f}, 等级={level_info['level_cn']}, "
                f"原因={reason[:30]})"
            )
            return data

    def _get_affection_lock(self, scope_id: str) -> asyncio.Lock:
        locks = getattr(self, "_affection_locks", None)
        if locks is None:
            self._affection_locks = {}
            locks = self._affection_locks
        if scope_id not in locks:
            locks[scope_id] = asyncio.Lock()
        return locks[scope_id]

    # ========== LLM 驱动的评估 ==========
    async def _llm_evaluate_affection(
        self, scope_id: str, dialog_items: list, event=None
    ) -> Optional[tuple[float, str]]:
        """使用 LLM 分析对话情感倾向，返回 (delta, reason)。
        返回 None 表示 LLM 不可用或冷却中，调用方不应回退到规则引擎。
        """
        cooldown = self._get_config("affection_eval_cooldown_seconds", 180)
        data = await self._load_affection(scope_id)
        last_eval = data.get("last_eval_time", 0.0)
        if time.time() - last_eval < cooldown:
            logger.debug(f"[好感度] LLM 评估冷却中，距上次 {time.time()-last_eval:.0f}s")
            return None  # 冷却中，跳过本轮（不回退规则引擎）

        umo = scope_id
        if event:
            try:
                umo = event.unified_msg_origin
            except Exception:
                pass

        provider_id = await self._get_llm_provider_id(umo)
        if not provider_id:
            logger.warning("[好感度] 无可用 LLM，回退到规则引擎")
            return None

        dialog_text = "\n".join(
            f"{'用户' if i['role']=='user' else '助手'}: {i['content']}"
            for i in dialog_items[-6:]  # 只分析最近 6 条
        )

        template = self._get_config(
            "affection_analysis_prompt_template",
            _DEFAULT_ANALYSIS_PROMPT,
        )
        prompt = template.replace("{对话历史}", dialog_text)

        timeout = self._get_config("llm_timeout_seconds", 60)
        try:
            resp = await asyncio.wait_for(
                self.context.llm_generate(chat_provider_id=provider_id, prompt=prompt),
                timeout=timeout,
            )
            text = resp.completion_text.strip()

            # 尝试解析 JSON
            json_match = re.search(r'\{[^{}]*"delta"[^{}]*"reason"[^{}]*\}', text, re.DOTALL)
            if not json_match:
                json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                delta = float(parsed.get("delta", 0))
                reason = str(parsed.get("reason", "LLM 评估"))
                delta = max(-3.0, min(3.0, delta))

                # ---- 负向保护 ----
                # LLM 给负分但对话中并无明确负面措辞时，判定为误判并归零。
                # 典型误判：用户只回「嗯/哦/好的」、被助手的烂回复整无语、正在忙。
                # 长回复/积极表达不在此列——只有"扣分"才需要举证。
                if delta < 0:
                    user_text = " ".join(
                        i.get("content", "") for i in dialog_items
                        if i.get("role") == "user"
                    )
                    if not has_explicit_negative(user_text):
                        logger.info(
                            f"[好感度] 抑制疑似误判的扣分 Δ={delta:+.1f}"
                            f"（用户侧无明确负面措辞）: {reason}"
                        )
                        return 0.0, "无明确负面表达，不扣分"

                logger.info(f"[好感度] LLM 评估: Δ={delta:+.1f}, reason={reason}")
                return delta, reason
        except asyncio.TimeoutError:
            logger.warning("[好感度] LLM 评估超时")
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"[好感度] LLM 返回解析失败: {e}, 原文={text[:100]}")
        except Exception as e:
            logger.warning(f"[好感度] LLM 评估异常: {e}")

        return None

    # ========== 触发入口（由 engine.py / main.py 调用） ==========
    async def _check_and_trigger_affection_eval(
        self, scope_id: str, dialog_items: list, event=None
    ) -> bool:
        """检查并触发好感度评估。
        优先 LLM（有冷却），LLM 冷却中则静默跳过；
        仅当 LLM Provider 完全不可用时才回退到规则引擎（有间隔轮数限制）。
        """
        if not self._get_config("affection_enabled", False):
            return False
        if not dialog_items:
            return False

        delta = None
        reason = ""

        # 先判断 LLM 是否可用
        umo = scope_id
        if event:
            try:
                umo = event.unified_msg_origin
            except Exception:
                pass
        llm_available = (await self._get_llm_provider_id(umo)) is not None

        if llm_available:
            # LLM 可用 → 尝试 LLM 评估（内部有冷却控制）
            llm_result = await self._llm_evaluate_affection(scope_id, dialog_items, event)
            if llm_result is not None:
                delta, reason = llm_result
                async with self._get_affection_lock(scope_id):
                    data = await self._load_affection(scope_id)
                    data["last_eval_time"] = time.time()
                    data["_rule_msg_count"] = 0
                    await self._save_affection(scope_id, data)
            else:
                # LLM 冷却中 → 静默跳过（不回退规则引擎）
                return False
        else:
            # LLM 完全不可用 → 回退规则引擎（有间隔轮数限制）
            rule_rounds = self._get_config("affection_rule_eval_rounds", 5)
            async with self._get_affection_lock(scope_id):
                data = await self._load_affection(scope_id)
                msg_count = data.get("_rule_msg_count", 0) + 1
                if msg_count < rule_rounds:
                    data["_rule_msg_count"] = msg_count
                    await self._save_affection(scope_id, data)
                    logger.debug(
                        f"[好感度] 规则评估跳过（{msg_count}/{rule_rounds} 轮）"
                    )
                    return False
                data["_rule_msg_count"] = 0
                await self._save_affection(scope_id, data)

            user_text = " ".join(
                i.get("content", "") for i in dialog_items if i.get("role") == "user"
            )
            asst_text = " ".join(
                i.get("content", "") for i in dialog_items if i.get("role") == "assistant"
            )
            delta, reason = rule_based_delta(user_text, asst_text)

        if abs(delta) < 0.05:
            logger.debug(f"[好感度] 变化微小 ({delta})，跳过")
            return False

        result = await self._apply_affection_delta(scope_id, delta, reason)
        return result is not None

    # ========== 好感度注入（在 on_llm_request 中调用） ==========
    async def _inject_affection(self, event, req, scope_id: str = ""):
        """将好感度上下文注入到 LLM 请求中。"""
        if not self._get_config("affection_enabled", False):
            return

        if not scope_id:  # 调用方已算好时直接复用，避免会话模式下重复查询会话 ID
            scope_id = await self._get_scope_id(event)
        data = await self._load_affection(scope_id)
        score = data.get("score", 0.0)
        level_cn = data.get("level_cn", "陌生人")
        thresholds = self._get_config("affection_level_thresholds", None)
        tone = compute_level(score, thresholds).get("tone_hint", "友好中立")

        ctx = (
            f"[好感度系统] 当前用户对你的好感度: {score:.0f}/100 (等级: {level_cn})。\n"
            f"建议回复语气: {tone}。\n"
            f"(注意：此信息仅作为语气参考，不要在回复中直接提及好感度数值)"
        )

        pos = self._get_config("affection_inject_position", "extra_user_content")
        preview = ctx.split("\n")[0]

        if pos == "system_prompt":
            req.system_prompt += "\n\n" + ctx
            logger.info(f"[好感度] ❤️ 注入(system) | score={score:.0f} | {level_cn}")
        else:
            # 统一走 extra_user_content_parts + mark_as_temp（官方推荐，不持久化、不破坏 prompt 缓存）
            ok = self._inject_extra_content(
                req, f"<affection_context>\n{ctx}\n</affection_context>"
            )
            if not ok:
                logger.warning("[好感度] extra_user_content 不可用，已降级为 system_prompt 注入")

            logger.info(
                f"[好感度] ❤️ 注入(extra) | scope={scope_id} | "
                f"score={score:.0f} | {level_cn} | 语气={tone[:20]}..."
            )
