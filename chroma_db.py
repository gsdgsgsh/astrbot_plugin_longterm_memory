"""ChromaDB 向量存储后端：嵌入式运行，零外部依赖。"""
import os
import sys
import time
from pathlib import Path
from typing import Optional

from astrbot.api import logger

# 导入失败时保留**真实原因**——此前用裸 except ImportError 静默丢弃，
# 导致"缺传递依赖 / 版本冲突"也被报成"未安装"，用户 repeatedly 装了也没用。
_CHROMA_IMPORT_ERROR = ""
_CHROMA_MISSING = False          # True 表示确实没有 chromadb 这个包
_CHROMA_MODULE_PATH = ""

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    HAS_CHROMADB = True
    _CHROMA_MODULE_PATH = getattr(chromadb, "__file__", "") or ""
except ImportError as _e:
    HAS_CHROMADB = False
    _CHROMA_IMPORT_ERROR = f"{type(_e).__name__}: {_e}"
    # ModuleNotFoundError 且缺的就是 chromadb 本身 → 真的没装；
    # 否则是缺传递依赖或版本冲突，必须提示"已装但导入失败"
    _CHROMA_MISSING = isinstance(_e, ModuleNotFoundError) and "chromadb" in str(_e)


class ChromaDBMixin:
    """ChromaDB 向量存储 Mixin。替代 KV + NumPy 方案，自动管理向量索引。"""

    @staticmethod
    def _chroma_status() -> dict:
        """供管理页诊断使用：是否可用、模块路径、导入错误、当前解释器。"""
        return {
            "available": HAS_CHROMADB,
            "missing": _CHROMA_MISSING,
            "module_path": _CHROMA_MODULE_PATH,
            "import_error": _CHROMA_IMPORT_ERROR,
            "executable": sys.executable,
        }

    def _init_chromadb(self):
        """初始化 ChromaDB 客户端（嵌入式模式，数据持久化到 data 目录）。"""
        if not HAS_CHROMADB:
            if _CHROMA_MISSING:
                logger.warning(
                    "[ChromaDB] 未检测到 chromadb 包，回退到 NumPy 方案。"
                    f"若已安装，请确认装进了**当前解释器**：{sys.executable}"
                    "（Docker 部署需在容器内安装，且容器重建后会丢失）"
                )
            else:
                logger.warning(
                    "[ChromaDB] chromadb **已安装但有导入错误**（不是没装），回退到 NumPy 方案。"
                    f"原因：{_CHROMA_IMPORT_ERROR}；当前解释器：{sys.executable}"
                )
            self._chroma_client = None
            return

        try:
            from astrbot.core.utils.astrbot_path import get_astrbot_data_path
            persist_dir = str(Path(get_astrbot_data_path()) / "plugin_data" / "ltm_chromadb")
        except Exception:
            persist_dir = str(Path("data") / "ltm_chromadb")

        try:
            self._chroma_client = chromadb.PersistentClient(
                path=persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            logger.info(f"[ChromaDB] 已初始化，持久化路径: {persist_dir}")
        except Exception as e:
            logger.warning(f"[ChromaDB] 初始化失败: {e}，回退到 NumPy 方案")
            self._chroma_client = None

    def _get_chroma_collection(self, scope_id: str):
        """获取或创建指定 scope 的 ChromaDB 集合。"""
        if not getattr(self, "_chroma_client", None):
            return None
        # 集合名只允许字母数字下划线连字符，ChromaDB 限制 63 字符
        safe_name = "ltm_" + "".join(c if c.isalnum() or c in "_-" else "_" for c in scope_id)
        safe_name = safe_name[:63].rstrip("_-")
        try:
            return self._chroma_client.get_or_create_collection(
                name=safe_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            logger.warning(f"[ChromaDB] 获取集合失败 ({scope_id}): {e}")
            return None

    async def _chroma_add_memory(self, scope_id: str, mem_id: str, text: str,
                                  embedding, mem_type: str, importance: int,
                                  tags: list, timestamp: float,
                                  narrative_time: str = ""):
        """向 ChromaDB 添加一条记忆（向量 + 元数据）。"""
        col = self._get_chroma_collection(scope_id)
        if not col:
            return False
        try:
            col.add(
                ids=[mem_id],
                embeddings=[embedding],
                metadatas=[{
                    "text": text[:2000],  # ChromaDB 元数据有限制
                    "type": mem_type,
                    "importance": importance,
                    "tags": ",".join(tags) if tags else "",
                    "timestamp": timestamp,
                    "narrative_time": (narrative_time or "")[:200],
                }],
            )
            return True
        except Exception as e:
            logger.warning(f"[ChromaDB] 添加记忆失败 ({mem_id[:8]}): {e}")
            return False

    async def _chroma_upsert_memory(self, scope_id: str, mem_id: str, text: str,
                                     embedding, mem_type: str, importance: int,
                                     tags: list, timestamp: float,
                                     narrative_time: str = ""):
        """更新或插入一条记忆到 ChromaDB（冲突合并 / 补嵌成功后同步）。"""
        col = self._get_chroma_collection(scope_id)
        if not col:
            return False
        try:
            col.upsert(
                ids=[mem_id],
                embeddings=[embedding],
                metadatas=[{
                    "text": text[:2000],
                    "type": mem_type,
                    "importance": importance,
                    "tags": ",".join(tags) if tags else "",
                    "timestamp": timestamp,
                    "narrative_time": (narrative_time or "")[:200],
                }],
            )
            return True
        except Exception as e:
            logger.warning(f"[ChromaDB] 更新记忆失败 ({mem_id[:8]}): {e}")
            return False

    async def _chroma_delete_memories(self, scope_id: str, mem_ids: list):
        """批量从 ChromaDB 删除多条记忆（FIFO 淘汰时同步）。"""
        col = self._get_chroma_collection(scope_id)
        if not col:
            return
        try:
            col.delete(ids=list(mem_ids))
        except Exception as e:
            logger.debug(f"[ChromaDB] 批量删除记忆失败: {e}")

    async def _chroma_search(self, scope_id: str, query_vec, top_k: int,
                              threshold: float = 0.0):
        """ChromaDB 向量检索，返回 [{id, text, similarity, embedding, ...}, ...] 列表。

        同时返回 embedding，使上调用的"近重复剔除 / MMR 多样性"在 ChromaDB 路径同样生效。
        """
        col = self._get_chroma_collection(scope_id)
        if not col or col.count() == 0:
            return []
        try:
            results = col.query(
                query_embeddings=[query_vec],
                n_results=min(top_k, col.count()),
                include=["metadatas", "distances", "embeddings"],
            )
            if not results or not results.get("ids") or not results["ids"][0]:
                return []

            embeddings = results.get("embeddings")
            items = []
            for i, mem_id in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i] if results.get("metadatas") else {}
                distance = results["distances"][0][i] if results.get("distances") else 0.0
                # ChromaDB cosine distance → similarity: sim = 1 - distance
                similarity = 1.0 - distance
                if similarity < threshold:
                    continue
                emb = None
                try:
                    if embeddings is not None and len(embeddings) > 0 and embeddings[0] is not None:
                        vec = embeddings[0][i]
                        emb = list(vec) if vec is not None else None
                except Exception:
                    emb = None
                items.append({
                    "id": mem_id,
                    "text": meta.get("text", ""),
                    "type": meta.get("type", "summary"),
                    "importance": int(meta.get("importance", 3)),
                    "tags": meta.get("tags", "").split(",") if meta.get("tags") else [],
                    "timestamp": float(meta.get("timestamp", 0)),
                    "similarity": similarity,
                    "embedding": emb,
                })
            return items
        except Exception as e:
            logger.warning(f"[ChromaDB] 检索失败 ({scope_id}): {e}")
            return []

    async def _chroma_delete_memory(self, scope_id: str, mem_id: str):
        """从 ChromaDB 删除单条记忆。"""
        col = self._get_chroma_collection(scope_id)
        if not col:
            return
        try:
            col.delete(ids=[mem_id])
        except Exception as e:
            logger.debug(f"[ChromaDB] 删除记忆失败 ({mem_id[:8]}): {e}")

    async def _chroma_delete_scope(self, scope_id: str):
        """删除整个 scope 的 ChromaDB 集合。"""
        if not getattr(self, "_chroma_client", None):
            return
        safe_name = "ltm_" + "".join(c if c.isalnum() or c in "_-" else "_" for c in scope_id)
        try:
            self._chroma_client.delete_collection(name=safe_name)
        except Exception:
            pass
