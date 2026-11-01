/* 长期记忆 · Dashboard 插件页面
 * 通过 window.AstrBotPluginPage bridge 与插件后端通信。
 * 认证由 Dashboard 负责，本页无需登录逻辑。
 */
"use strict";

var bridge = window.AstrBotPluginPage;
var state = {
  tab: "overview", scope: "", scopes: [], cfg: {},
  mems: [],                                   // 当前作用域记忆缓存（供筛选使用）
  buffer: null,                               // 待总结缓冲区快照
  bufferOpen: false,
  cfgDefaults: {},                            // 各配置项的 schema 默认值（用于逐项恢复默认）
  hasBackup: false,                           // 是否存在记忆重整备份（可回滚）
  reorgPending: 0,                            // 上次重整失败的条目数（可单独重试）
  editingId: "",                              // 正在编辑的记忆 id
  filter: { q: "", type: "", failedOnly: false }
};

/* ---------- 基础工具 ---------- */
function esc(s) {
  if (s === null || s === undefined) return "";
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}
function fmtTime(ts) {
  if (!ts) return "-";
  var d = new Date(ts * 1000);
  var p = function (n) { return String(n).padStart(2, "0"); };
  return d.getFullYear() + "-" + p(d.getMonth() + 1) + "-" + p(d.getDate()) +
    " " + p(d.getHours()) + ":" + p(d.getMinutes());
}
function showToast(msg, type) {
  var box = document.getElementById("toasts");
  var el = document.createElement("div");
  el.className = "toast " + (type || "");
  el.textContent = msg;
  box.appendChild(el);
  setTimeout(function () { if (el.parentNode) box.removeChild(el); }, 4200);
}
function shortScope(id) {
  if (!id) return "";
  var s = String(id);
  if (s.indexOf("conversation:") === 0) {
    var parts = s.split(":");
    var cid = parts[parts.length - 1] || "";
    var origin = parts.slice(1, parts.length - 1).join(":");
    return "会话 " + cid.slice(0, 8) + " · " + origin;
  }
  return s.length > 34 ? s.slice(0, 32) + "…" : s;
}
function errText(e) {
  if (!e) return "未知错误";
  return e.message || String(e);
}
function setView(html) { document.getElementById("view").innerHTML = html; }

/* ---------- bridge 调用 ---------- */
async function apiGet(endpoint, params) {
  return await bridge.apiGet(endpoint, params || {});
}
async function apiPost(endpoint, body) {
  return await bridge.apiPost(endpoint, body || {});
}

/* ---------- 零依赖 SVG 图表 ---------- */
function chartLine(pts, opts) {
  opts = opts || {};
  var W = opts.width || 640, H = opts.height || 150;
  var color = opts.color || "#5b8cff";
  if (!pts || !pts.length) return '<div class="empty">暂无数据</div>';
  var vals = pts.map(function (p) { return Number(p.value) || 0; });
  var max = Math.max.apply(null, vals), min = Math.min.apply(null, vals);
  var span = (max - min) || 1;
  var padL = 10, padR = 10, padT = 12, padB = 20;
  var innerW = W - padL - padR, innerH = H - padT - padB;
  var X = function (i) { return pts.length <= 1 ? padL + innerW / 2 : padL + (i * innerW) / (pts.length - 1); };
  var Y = function (v) { return padT + (1 - (v - min) / span) * innerH; };
  var coords = pts.map(function (p, i) { return [X(i).toFixed(1), Y(Number(p.value) || 0).toFixed(1)]; });
  var line = '<polyline points="' + coords.map(function (c) { return c.join(","); }).join(" ") +
    '" fill="none" stroke="' + color + '" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round"/>';
  var area = '<polygon points="' + X(0).toFixed(1) + "," + (H - padB) + " " +
    coords.map(function (c) { return c.join(","); }).join(" ") + " " +
    X(pts.length - 1).toFixed(1) + "," + (H - padB) + '" fill="' + color + '" opacity="0.12"/>';
  var dots = coords.map(function (c) {
    return '<circle cx="' + c[0] + '" cy="' + c[1] + '" r="2.6" fill="' + color + '"/>';
  }).join("");
  var labels = "";
  if (pts.length > 1) {
    var step = Math.max(1, Math.ceil(pts.length / 7));
    var out = [];
    for (var i = 0; i < pts.length; i += step) {
      out.push('<text x="' + X(i) + '" y="' + (H - 5) + '" text-anchor="middle" font-size="9" fill="currentColor" opacity=".55">' + esc(pts[i].label) + "</text>");
    }
    labels = out.join("");
  }
  return '<svg width="100%" viewBox="0 0 ' + W + " " + H + '" style="display:block;color:inherit">' +
    area + line + dots + labels + "</svg>";
}
function chartBars(items) {
  var arr = items || [];
  if (!arr.length) return '<div class="empty">暂无数据</div>';
  var max = Math.max.apply(null, arr.map(function (i) { return Number(i.value) || 0; }).concat([1]));
  return '<div style="display:flex;flex-direction:column;gap:9px">' + arr.map(function (i) {
    var v = Number(i.value) || 0;
    var pct = (v / max * 100).toFixed(1);
    return '<div class="row">' +
      '<span style="width:150px;font-size:11px;opacity:.72;text-align:right;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex-shrink:0">' + esc(i.label) + "</span>" +
      '<div class="bar-track"><i style="width:' + pct + '%;background:' + (i.color || "#34d399") + '"></i></div>' +
      '<span style="width:40px;font-size:11px;font-weight:700;text-align:right;flex-shrink:0">' + v + "</span></div>";
  }).join("") + "</div>";
}

/* ---------- 作用域 ---------- */
async function loadScopes() {
  var d = await apiGet("scopes");
  state.scopes = (d && d.scopes) || [];
  state.memoryScope = (d && d.memory_scope) || "";
  var sel = document.getElementById("scopeSelect");
  sel.innerHTML = "";
  if (!state.scopes.length) {
    var o = document.createElement("option");
    o.value = "";
    o.textContent = state.memoryScope === "user"
      ? "暂无数据（user 模式下每个用户只有 1 个作用域）"
      : "暂无数据（还没有产生记忆）";
    sel.appendChild(o);
    state.scope = "";
    var hintEl = document.getElementById("scopeHint");
    if (hintEl) {
      hintEl.textContent = state.memoryScope === "user"
        ? " 隔离模式 user：所有对话共用一份记忆；产生记忆后这里会出现作用域"
        : " 还没有任何作用域：先在对话里聊几句，产生总结后就会出现";
    }
    return;
  }
  state.scopes.forEach(function (s) {
    var op = document.createElement("option");
    op.value = s.scope_id;
    op.textContent = shortScope(s.scope_id) + "（" + (s.memory_count || 0) + "）";
    op.title = s.scope_id;
    sel.appendChild(op);
  });
  var exists = state.scopes.some(function (s) { return s.scope_id === state.scope; });
  if (!exists) state.scope = state.scopes[0].scope_id;
  sel.value = state.scope;
  var modeTip = {
    user: "所有对话共用一份",
    group_user: "按群 + 用户隔离",
    user_bot: "按用户 + 机器人隔离",
    conversation: "每个对话独立"
  }[state.memoryScope] || "";
  document.getElementById("scopeHint").textContent =
    " 共 " + state.scopes.length + " 个作用域" +
    (state.memoryScope ? " · 隔离模式：" + state.memoryScope + (modeTip ? "（" + modeTip + "）" : "") : "");
}

/* ---------- 页面渲染 ---------- */
function render() {
  if (state.tab === "overview") return renderOverview();
  if (state.tab === "memories") return renderMemories();
  if (state.tab === "profile") return renderProfile();
  if (state.tab === "affection") return renderAffection();
  if (state.tab === "config") return renderConfig();
}

async function renderOverview() {
  setView('<div class="empty">加载中…</div>');
  var stats = {}, trend = [];
  try { stats = await apiGet("stats"); } catch (e) { showToast("统计加载失败：" + errText(e), "err"); }
  trend = await buildTrend(14);
  var cards = [
    ["记忆总数", stats.total_memories || 0, ""],
    ["总结记忆", stats.summary_count || 0, ""],
    ["原始记忆", stats.raw_count || 0, ""],
    ["嵌入失败", stats.embed_failed_count || 0, (stats.embed_failed_count ? "var(--danger)" : "")],
    ["作用域", stats.scope_count || 0, ""],
    ["Embedding", stats.embedding_ready ? "就绪" : "未配置", (stats.embedding_ready ? "" : "var(--warn)")]
  ].map(function (c) {
    return '<div class="card"><div class="num" style="' + (c[2] ? "color:" + c[2] : "") + '">' + esc(c[1]) +
      '</div><div class="lbl">' + esc(c[0]) + "</div></div>";
  }).join("");
  var alert = (stats.embed_failed_count > 0)
    ? '<div class="alert"><div style="flex:1">⚠️ 有 <b>' + stats.embed_failed_count +
      '</b> 条记忆嵌入失败，无法参与语义检索。若下方「嵌入诊断」显示 Provider 就绪，可直接一键补嵌。</div>' +
      '<button class="btn btn-sm btn-primary" id="retryAllBtn">⚡ 一键重新嵌入全部</button></div>'
    : "";
  var diag = await buildEmbedDiag();
  setView(alert + '<div class="cards">' + cards + "</div>" +
    diag +
    '<div class="panel"><h3>📈 近 14 天记忆增长</h3>' + chartLine(trend, { color: "#5b8cff" }) + "</div>" +
    '<div class="panel"><h3>📊 各作用域记忆量</h3>' + chartBars(state.scopes.slice(0, 10).map(function (s) {
      return { label: shortScope(s.scope_id), value: s.memory_count || 0, color: "#34d399" };
    })) + "</div>" +
    '<div class="panel"><h3>ℹ️ 当前隔离策略</h3><div>memory_scope = <code class="mono">' +
      esc(stats.memory_scope || "-") + "</code>　每作用域上限 " + esc(stats.memory_limit || "-") + " 条</div></div>" +
    '<div class="panel"><h3>💾 数据备份 <span class="cfg-hint">推荐定期导出</span></h3>' +
    '<div class="cfg-hint">⚠️ AstrBot <b>卸载插件时会清空本插件的全部 KV 数据</b>（记忆就存在 KV 里）。' +
    '导出的文件保存在你自己的电脑上，卸载/重装插件都不会丢。</div>' +
    '<div class="prof-actions"><button class="btn btn-primary btn-sm" id="expBtn">💾 导出备份</button>' +
    '<button class="btn btn-sm" id="impBtn">📥 导入备份</button>' +
    '<input type="file" id="impFile" accept=".json,application/json" style="display:none"></div></div>');

  var btn = document.getElementById("retryAllBtn");
  if (btn) btn.onclick = function () { startRetryAll(""); };
  var tb = document.getElementById("embedTestBtn");
  if (tb) tb.onclick = startEmbedTest;
  var crb = document.getElementById("chromaReindexBtn");
  if (crb) crb.onclick = reindexChroma;
  var eb = document.getElementById("expBtn");
  if (eb) eb.onclick = exportBackup;
  var ib = document.getElementById("impBtn");
  if (ib) ib.onclick = function () {
    var f = document.getElementById("impFile");
    if (!f) return;
    f.value = "";
    f.click();
  };
  var fi = document.getElementById("impFile");
  if (fi) fi.onchange = function () {
    if (fi.files && fi.files[0]) doImport(fi.files[0]);
  };
}

/* ---------- ChromaDB 向量回填 ---------- */
function reindexChroma() {
  askConfirm(
    "把当前全部已有记忆的向量写入 ChromaDB？\n\n" +
    "开启 ChromaDB 后**必须做一次**：插件只在新增记忆时写向量，历史记忆不在集合里；\n" +
    "而检索时 Chroma 一旦有结果就不会回退到 NumPy，表现为旧记忆查不到。",
    async function () {
      try {
        var r = await apiPost("chroma/reindex", { scope: "" });
        showToast("已回填 " + r.memories + " 条向量（" + r.scopes + " 个作用域）" +
          (r.skipped ? "，跳过 " + r.skipped : ""), "ok");
      } catch (e) { showToast("回填失败：" + errText(e), "err"); }
    }, "开始回填"
  );
}

/* ---------- 备份导出 / 导入 ---------- */
function downloadJson(obj, filename) {
  var blob = new Blob([JSON.stringify(obj, null, 2)], { type: "application/json" });
  var url = URL.createObjectURL(blob);
  var a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  setTimeout(function () {
    try { document.body.removeChild(a); } catch (e) {}
    URL.revokeObjectURL(url);
  }, 800);
}

async function exportBackup() {
  try {
    var d = await apiGet("export");
    if (!d || !d.memory_count) { showToast("当前没有任何记忆可导出", "err"); return; }
    var ts = new Date().toISOString().slice(0, 10).replace(/-/g, "");
    downloadJson(d, "ltm_backup_" + ts + ".json");
    showToast("已导出 " + d.scope_count + " 个作用域 / " + d.memory_count + " 条记忆", "ok");
  } catch (e) { showToast("导出失败：" + errText(e), "err"); }
}

async function doImport(file) {
  var text;
  try { text = await file.text(); } catch (e) { showToast("读取文件失败", "err"); return; }
  var data;
  try { data = JSON.parse(text); } catch (e) { showToast("不是合法的 JSON 文件", "err"); return; }
  if (!data || data.format !== "astrbot_plugin_longterm_memory.backup") {
    showToast("不是本插件导出的备份文件", "err");
    return;
  }
  askConfirm(
    "导入 " + (data.scope_count || 0) + " 个作用域 / " + (data.memory_count || 0) + " 条记忆？\n\n" +
    "采用合并模式：只补充当前缺失的记忆，不会覆盖已有内容。\n导入前会自动备份现有数据。",
    async function () {
      try {
        var r = await apiPost("import", { data: data, mode: "merge" });
        showToast("已导入 " + r.scopes + " 个作用域 / " + r.memories + " 条记忆", "ok");
        await loadScopes();
        render();
      } catch (e) { showToast("导入失败：" + errText(e), "err"); }
    }, "合并导入"
  );
}

/* ---------- 嵌入连通性测试 ---------- */
async function startEmbedTest() {
  var out = document.getElementById("embedTestResult");
  var tb = document.getElementById("embedTestBtn");
  if (tb) tb.disabled = true;
  if (out) out.textContent = "测试中…";
  try {
    var r = await apiPost("embed/test", {});
    if (out) out.textContent = "✅ 连通正常：维度 " + r.dim + "，耗时 " + r.elapsed_ms + " ms";
    showToast("嵌入连通正常（" + r.dim + " 维）", "ok");
  } catch (e) {
    if (out) out.textContent = "❌ " + errText(e);
    showToast("嵌入测试失败：" + errText(e), "err");
  } finally {
    if (tb) tb.disabled = false;
  }
}

/* ---------- 批量补嵌任务 ---------- */
var jobPoll = null;
var watchedJobs = {};

async function startRetryAll(scope) {
  try {
    var r = await apiPost("memory/retry-all", { scope: scope || "" });
    if (!r.started) { showToast(r.msg || "没有需要补嵌的记忆", "ok"); return; }
    showToast("已开始补嵌 " + r.total + " 条记忆…", "ok");
    watchJob("embed");
  } catch (e) {
    showToast("启动补嵌失败：" + errText(e), "err");
  }
}

/* ---------- 页内确认框 ----------
 * 插件页运行在 sandbox="allow-scripts allow-forms allow-downloads" 的 iframe 中，
 * **没有 allow-modals** → 原生 confirm()/alert()/prompt() 会被浏览器静默拦截
 * （confirm 直接返回 false 且不弹窗），表现为"点按钮没反应"。
 * 因此一律使用本页自绘的确认框，不用原生对话框。
 */
function askConfirm(message, onOk, okLabel, onCancel) {
  var box = document.getElementById("cfmBox");
  if (!box) {
    box = document.createElement("div");
    box.id = "cfmBox";
    box.className = "cfm-mask";
    box.innerHTML = '<div class="cfm-card"><div class="cfm-msg"></div>' +
      '<div class="row" style="gap:8px;justify-content:flex-end;margin-top:14px">' +
      '<button class="btn" id="cfmCancel">取消</button>' +
      '<button class="btn btn-primary" id="cfmOk">确定</button></div></div>';
    document.body.appendChild(box);
  }
  box.querySelector(".cfm-msg").textContent = message;
  box.querySelector("#cfmOk").textContent = okLabel || "确定";
  var done = function (ok) {
    box.style.display = "none";
    if (ok) { try { onOk && onOk(); } catch (e) { console.error(e); } }
    else { try { onCancel && onCancel(); } catch (e) { console.error(e); } }
  };
  box.querySelector("#cfmCancel").onclick = function () { done(false); };
  box.querySelector("#cfmOk").onclick = function () { done(true); };
  box.onclick = function (ev) { if (ev.target === box) done(false); };
  box.style.display = "flex";
}

/* ---------- 记忆迁移到其他作用域 ---------- */
function askMigrate(scope) {
  var others = (state.scopes || []).filter(function (s) { return s.scope_id !== scope; });
  if (!others.length) {
    showToast("没有其他作用域可作为目标。先把 memory_scope 切到 conversation，并在目标对话里发一句话，让它的作用域出现后再迁移", "err");
    return;
  }
  var box = document.getElementById("migBox");
  if (!box) {
    box = document.createElement("div");
    box.id = "migBox";
    box.className = "cfm-mask";
    document.body.appendChild(box);
  }
  var opts = others.map(function (s) {
    return '<option value="' + esc(s.scope_id) + '">' + esc(shortScope(s.scope_id)) +
      "（" + (s.memory_count || 0) + " 条）</option>";
  }).join("");
  box.innerHTML = '<div class="cfm-card"><div class="cfm-msg">把当前作用域的记忆迁移到哪个作用域？</div>' +
    '<div class="cfg-hint" style="margin:8px 0">当前：' + esc(shortScope(scope)) + "</div>" +
    '<select id="migSel" style="width:100%">' + opts + "</select>" +
    '<div class="cfg-hint" style="margin-top:8px">记忆条目<b>不记录来源对话</b>，所以只能整体迁移到指定作用域；重复内容会自动跳过；迁移前会自动备份目标作用域（可用「↩ 回滚重整」还原）。</div>' +
    '<div class="row" style="gap:8px;justify-content:flex-end;margin-top:14px">' +
    '<button class="btn" id="migCancel">取消</button>' +
    '<button class="btn" id="migCopy">只复制</button>' +
    '<button class="btn btn-primary" id="migMove">移动</button></div></div>';
  box.style.display = "flex";
  var close = function () { box.style.display = "none"; };
  box.querySelector("#migCancel").onclick = close;
  box.onclick = function (ev) { if (ev.target === box) close(); };
  var run = async function (mode) {
    var to = box.querySelector("#migSel").value;
    close();
    try {
      var r = await apiPost("memories/migrate", { from: scope, to: to, mode: mode });
      showToast(
        (mode === "move" ? "已迁移 " : "已复制 ") + r.added + " 条" +
        (r.skipped ? "，跳过重复 " + r.skipped + " 条" : ""),
        "ok"
      );
      await loadScopes();
      render();
    } catch (e) { showToast("迁移失败：" + errText(e), "err"); }
  };
  box.querySelector("#migMove").onclick = function () { run("move"); };
  box.querySelector("#migCopy").onclick = function () { run("copy"); };
}

function reorganize(scope) {
  askConfirm(
    "将用 LLM 重整本作用域的记忆：\n· 同一话题的零散条目合并为一条\n· 混了多个话题的条目拆开\n\n重整前会自动备份，可随时回滚。",
    async function () {
      try {
        var r = await apiPost("memories/reorganize", { scope: scope });
        showToast("已开始重整 " + r.count + " 条记忆…", "ok");
        watchJob("reorg");
      } catch (e) { showToast("启动重整失败：" + errText(e), "err"); }
    }, "开始重整");
}

function retryReorg(scope) {
  askConfirm(
    "只重跑上次失败的 " + state.reorgPending + " 条？\n\n" +
    "已成功合并的记忆不会被改动（只替换参与重试的那些条目）。",
    async function () {
      try {
        var r = await apiPost("memories/reorganize", { scope: scope, mode: "resume" });
        showToast("已开始重试 " + r.count + " 条…", "ok");
        watchJob("reorg");
      } catch (e) { showToast("启动失败：" + errText(e), "err"); }
    }, "开始重试"
  );
}

function restoreBackup(scope) {
  askConfirm("回滚到重整前的记忆？当前的重整结果会被丢弃。", async function () {
    try {
      var r = await apiPost("memories/restore-backup", { scope: scope });
      showToast("已回滚，恢复 " + r.restored + " 条记忆", "ok");
      await loadScopes();
      render();
    } catch (e) { showToast("回滚失败：" + errText(e), "err"); }
  }, "回滚");
}

function regenProfile(scope) {
  askConfirm(
    "从现有记忆重新生成画像？\n\n这会覆盖当前画像（旧版画像可能因保存时截断而残缺，重建可得到完整版本）。",
    async function () {
      try {
        await apiPost("profile/regenerate", { scope: scope });
        showToast("已开始重建画像…", "ok");
        watchJob("profile");
      } catch (e) { showToast("启动失败：" + errText(e), "err"); }
    }, "开始重建");
}

function reevaluateAffection(scope) {
  askConfirm(
    "让 AI 读取你与它的互动记录，重新判定亲密度并写入分数？\n\n这会覆盖当前分数（趋势图会保留记录）。",
    async function () {
      try {
        await apiPost("affection/reevaluate", { scope: scope });
        showToast("AI 正在评估既有互动…", "ok");
        watchJob("affection");
      } catch (e) { showToast("启动失败：" + errText(e), "err"); }
    }, "开始评估");
}

function watchJob(name) {
  watchedJobs[name] = true;
  pollJobs();
}

function deleteFailed(scope) {
  askConfirm("确定删除嵌入失败的记忆？这些记忆无法参与语义检索，删除后不可恢复。", async function () {
    try {
      var r = await apiPost("memory/delete-failed", { scope: scope || "" });
      showToast("已删除 " + r.removed + " 条失败记忆", "ok");
      await loadScopes();
      render();
    } catch (e) { showToast("删除失败：" + errText(e), "err"); }
  }, "删除");
}

function pollJobs() {
  if (jobPoll) return;
  jobPoll = setInterval(async function () {
    var st;
    try { st = await apiGet("job/status"); } catch (e) { return; }
    var jobs = (st && st.jobs) || {};
    var running = null;
    ["reorg", "profile", "affection", "embed"].forEach(function (n) {
      var j = jobs[n];
      if (j && j.running && !running) running = { name: n, job: j };
    });
    if (running) { showJobBar(running.name, running.job); return; }

    clearInterval(jobPoll); jobPoll = null;
    ["embed", "profile", "reorg", "affection"].forEach(function (n) {
      var j = jobs[n];
      if (!j || !watchedJobs[n]) return;
      delete watchedJobs[n];
      var msg = j.message ||
        ("任务结束：成功 " + (j.ok || 0) + " / 失败 " + (j.failed || 0));
      if (j.error) msg += "（" + j.error + "）";
      showToast(msg, (j.failed || j.error) ? "err" : "ok");
    });
    setTimeout(hideJobBar, 2500);
    try { await loadScopes(); } catch (e) {}
    render();
  }, 1200);
}

var JOB_LABEL = { embed: "⚡ 批量补嵌", profile: "👤 重建画像", reorg: "🧩 重整记忆", affection: "❤️ AI 评估亲密度" };

function showJobBar(name, st) {
  var bar = document.getElementById("jobBar");
  if (!bar) return;
  bar.hidden = false;
  var total = st.total || 0, done = st.done || 0;
  var pct = total ? Math.min(100, (done / total) * 100) : (st.running ? 5 : 100);
  var text = st.message || "";
  if (name === "embed") {
    text = (total ? done + " / " + total + " · " : "") +
      "成功 " + (st.ok || 0) + "，失败 " + (st.failed || 0);
  } else if (total > 0) {
    text = done + " / " + total + (text ? " · " + text : "");
  }
  document.getElementById("jobFill").style.width = pct.toFixed(1) + "%";
  document.getElementById("jobText").textContent = (JOB_LABEL[name] || "任务") + " · " + (text || "进行中…");
}
function hideJobBar() {
  var bar = document.getElementById("jobBar");
  if (bar) bar.hidden = true;
}

/* ---------- 嵌入诊断面板 ---------- */
async function buildEmbedDiag() {
  var info = {};
  try { info = await apiGet("embed/info"); } catch (e) { info = { ready: false, last_error: errText(e) }; }
  var ok = !!info.ready;
  var rows = [
    ["Provider", info.provider_id || info.class_name || "—"],
    ["向量维度", info.dim === null || info.dim === undefined ? "—" : info.dim],
    ["缓存条目", info.cache_size === 0 || info.cache_size ? info.cache_size : "—"]
  ].map(function (r) {
    return '<div class="kv"><span class="k">' + esc(r[0]) + '</span><span class="v">' + esc(r[1]) + "</span></div>";
  }).join("");
  var ch = info.chromadb || {};
  var chOn = !!info.chromadb_enabled;
  var chState, chBadge;
  if (ch.available) { chState = "可用"; chBadge = "ok"; }
  else if (ch.missing) { chState = "未安装"; chBadge = "fail"; }
  else { chState = "已安装但导入失败"; chBadge = "fail"; }
  var chRows = [
    ["向量后端", chOn ? "ChromaDB" : "NumPy（内置）"],
    ["ChromaDB", chState]
  ].map(function (r) {
    return '<div class="kv"><span class="k">' + esc(r[0]) + '</span><span class="v">' + esc(r[1]) + "</span></div>";
  }).join("");
  // 只在真的出问题时才展开细节，避免正常状态下干扰阅读
  var chDetail = "";
  if (chOn && !ch.available) {
    chDetail = '<div class="alert" style="margin:10px 0 0"><div style="flex:1">' +
      (ch.missing
        ? "已启用 ChromaDB 但检测不到 chromadb 包。请把它装进**当前解释器**：<br><code>" + esc(ch.executable || "") + "</code>" +
          "<br>Docker 部署需在容器内安装（容器重建后会丢失）。"
        : "chromadb 已安装但导入失败，原因：<br><code>" + esc(ch.import_error || "") + "</code>") +
      "</div></div>";
  }
  var errLine = info.last_error
    ? '<div class="alert" style="margin:10px 0 0"><div style="flex:1">最近一次嵌入失败原因：' + esc(info.last_error) + "</div></div>"
    : "";
  var hint = ok ? "" : '<div class="cfg-hint" style="margin-top:8px">未找到 Embedding Provider：请在「配置」页设置 embedding_provider_id，或确认 AstrBot 已配置嵌入模型。</div>';
  return '<div class="panel"><h3>🔌 嵌入诊断 ' +
    '<span class="badge ' + (ok ? "ok" : "fail") + '">' + (ok ? "已就绪" : "不可用") + "</span></h3>" +
    rows + chRows + chDetail + hint + errLine +
    '<div class="prof-actions"><button class="btn btn-sm" id="embedTestBtn">🧪 测试嵌入连通性</button>' +
    (ch.available && chOn ? '<button class="btn btn-sm" id="chromaReindexBtn" title="把已有记忆的向量写入 ChromaDB——切换向量后端后必做，否则旧记忆查不到">🔄 回填向量到 ChromaDB</button>' : "") +
    '</div>' +
    '<div id="embedTestResult" class="cfg-hint" style="margin-top:8px"></div></div>';
}

async function buildTrend(days) {
  var now = new Date();
  var order = [], buckets = {};
  for (var i = days - 1; i >= 0; i--) {
    var d = new Date(now); d.setDate(now.getDate() - i);
    var k = (d.getMonth() + 1) + "-" + String(d.getDate()).padStart(2, "0");
    order.push(k); buckets[k] = 0;
  }
  var today0 = new Date(now); today0.setHours(0, 0, 0, 0);
  var lists = await Promise.all(state.scopes.slice(0, 8).map(function (s) {
    return apiGet("memories", { scope: s.scope_id }).catch(function () { return { memories: [] }; });
  }));
  lists.forEach(function (d) {
    ((d && d.memories) || []).forEach(function (m) {
      var ts = Number(m.timestamp); if (!ts) return;
      var dd = new Date(ts * 1000), day0 = new Date(dd); day0.setHours(0, 0, 0, 0);
      var diff = Math.round((today0 - day0) / 86400000);
      if (diff < 0 || diff >= days) return;
      var k = (dd.getMonth() + 1) + "-" + String(dd.getDate()).padStart(2, "0");
      if (buckets[k] !== undefined) buckets[k]++;
    });
  });
  return order.map(function (k) { return { label: k, value: buckets[k] || 0 }; });
}

async function renderMemories() {
  if (!state.scope) { setView('<div class="empty">请先在上方选择一个作用域</div>'); return; }
  setView('<div class="empty">加载中…</div>');
  var d;
  try { d = await apiGet("memories", { scope: state.scope }); }
  catch (e) { setView('<div class="empty">加载失败：' + esc(errText(e)) + "</div>"); return; }
  state.mems = (d && d.memories) || [];
  state.hasBackup = !!(d && d.has_backup);
  state.reorgPending = (d && d.reorg_pending) || 0;
  state.editingId = "";
  try { state.buffer = await apiGet("buffer", { scope: state.scope }); }
  catch (e) { state.buffer = null; }
  paintMemList();
}

/* 缓冲区面板 HTML（折叠式） */
function bufferPanelHtml() {
  var b = state.buffer;
  if (!b) return "";
  var open = state.bufferOpen;
  var head = '<div class="row" style="cursor:pointer" id="bufToggle"><span style="flex:1">' +
    "📥 待总结缓冲区：<b>" + (b.user_rounds || 0) + "</b> 轮（" + (b.count || 0) + " 条消息）" +
    (b.user_rounds ? "" : '　<span class="cfg-hint">对话积累不足，尚未触发总结</span>') +
    '</span><span class="cfg-hint">' + (open ? "收起 ▲" : "展开 ▼") + "</span></div>";
  if (!open) return '<div class="panel" style="margin-bottom:11px">' + head + "</div>";
  var items = (b.items || []).map(function (i) {
    var who = i.role === "user" ? "👤 用户" : "🤖 助手";
    return '<div class="kv"><span class="k">' + who + '</span><span class="v">' +
      esc(i.preview) + '</span><span class="cfg-hint" style="white-space:nowrap">' +
      esc(fmtTime(i.timestamp).slice(5)) + '</span></div>';
  }).join("") || '<div class="empty">缓冲区为空</div>';
  return '<div class="panel" style="margin-bottom:11px">' + head +
    '<div style="max-height:260px;overflow:auto;margin-top:9px">' + items + "</div></div>";
}

/* 记忆列表：客户端搜索 / 类型筛选 / 仅看失败（即时响应，无需往返） */
function paintMemList() {
  var all = state.mems || [];
  if (!all.length) {
    // 记忆为空时也必须给出「回滚」入口——否则清空/迁走之后在页面上没有任何恢复途径
    var restoreBtn = state.hasBackup
      ? '<div class="prof-actions" style="margin-top:12px">' +
        '<button class="btn btn-primary" id="restoreBtn" title="从清空/迁移前的自动备份恢复">↩ 回滚到备份</button></div>'
      : "";
    var tip = state.hasBackup
      ? "检测到该作用域在清空/迁移前留下的备份，可点下方按钮恢复。"
      : "如果记忆是被「迁移」到了别的作用域，请先在上方下拉框切到那个作用域查看。";
    setView('<div class="panel"><div class="empty" style="margin:0">该作用域暂无记忆。<br>' +
      esc(tip) + "</div>" + restoreBtn + "</div>");
    var rb0 = document.getElementById("restoreBtn");
    if (rb0) rb0.onclick = function () { restoreBackup(state.scope); };
    return;
  }
  var f = state.filter;
  var q = f.q.trim().toLowerCase();
  var list = all.filter(function (m) {
    if (f.failedOnly && !m._embed_failed) return false;
    if (f.type && m.type !== f.type) return false;
    if (q && String(m.text || "").toLowerCase().indexOf(q) < 0) return false;
    return true;
  });
  var failedAll = all.filter(function (m) { return m._embed_failed; }).length;

  var toolbar = bufferPanelHtml() + '<div class="panel" style="margin-bottom:11px">' +
    '<div class="row" style="flex-wrap:wrap;gap:8px">' +
    '<input type="text" id="memSearch" placeholder="🔍 搜索记忆内容…" value="' + esc(f.q) + '" style="flex:1;min-width:180px">' +
    '<select id="memType"><option value="">全部类型</option>' +
    '<option value="summary"' + (f.type === "summary" ? " selected" : "") + ">仅总结</option>" +
    '<option value="raw"' + (f.type === "raw" ? " selected" : "") + ">仅原始</option></select>" +
    '<label class="row" style="gap:6px;cursor:pointer"><input type="checkbox" id="memFailedOnly"' +
    (f.failedOnly ? " checked" : "") + ' style="width:15px;height:15px;accent-color:var(--accent)">仅看嵌入失败' +
    (failedAll ? ' <span class="badge fail">' + failedAll + "</span>" : "") + "</label>" +
    "</div>" +
    '<div class="row" style="margin-top:9px;flex-wrap:wrap;gap:8px">' +
    '<span class="cfg-hint" style="flex:1">显示 ' + list.length + " / " + all.length + " 条</span>" +
    (failedAll ? '<button class="btn btn-sm btn-primary" id="retryScopeFailed">⚡ 补嵌本作用域失败项（' + failedAll + "）</button>" +
      '<button class="btn btn-sm btn-danger" id="delScopeFailed">🗑 删除失败项</button>' : "") +
    '<button class="btn btn-sm" id="reorgBtn" title="用 LLM 合并同话题碎片、拆分混合话题">🧩 重整记忆</button>' +
    (state.reorgPending ? '<button class="btn btn-sm btn-primary" id="reorgRetryBtn" title="只重跑上次失败的批次，不会影响已合并好的记忆">🔁 重试失败批次（' + state.reorgPending + '）</button>' : "") +
    '<button class="btn btn-sm" id="migBtn" title="把记忆整体迁移到另一个作用域（如从 user 迁到某个对话）">📤 迁移到…</button>' +
    (state.hasBackup ? '<button class="btn btn-sm" id="restoreBtn" title="回到重整/清空/迁移前的备份">↩ 回滚到备份</button>' : "") +
    "</div>" +
    '<div class="cfg-hint" style="margin-top:7px">作用域：<code class="mono">' + esc(state.scope) + "</code></div></div>";

  if (!list.length) {
    setView(toolbar + '<div class="empty">没有符合筛选条件的记忆。</div>');
    bindMemToolbar();
    return;
  }
  var html = list.map(function (m) {
    var badge = m.type === "summary" ? '<span class="badge summary">总结</span>' : '<span class="badge">原始</span>';
    var fail = m._embed_failed ? '<span class="badge fail">嵌入失败</span>' : "";
    var imp = '<span class="badge imp">重要 ' + (m.importance || 3) + "</span>";
    var tags = (m.tags || []).map(function (t) { return '<span class="tag">' + esc(t) + "</span>"; }).join("");
    var head = '<div class="mem-head">' + badge + fail + imp +
      '<span class="mem-time">' + esc(fmtTime(m.timestamp)) + "</span></div>";
    if (state.editingId === m.id) {
      var opts = [1, 2, 3, 4, 5].map(function (v) {
        return '<option value="' + v + '"' + (v === (m.importance || 3) ? " selected" : "") + ">" + v + "</option>";
      }).join("");
      return '<div class="mem">' + head +
        '<textarea class="prof-edit" id="memEditText" style="min-height:110px">' + esc(m.text) + "</textarea>" +
        '<div class="row" style="margin-top:8px;gap:8px">' +
        '<span class="cfg-hint">重要性</span><select id="memEditImp">' + opts + "</select>" +
        '<button class="btn btn-sm btn-primary" data-act="save-edit" data-id="' + esc(m.id) + '">保存</button>' +
        '<button class="btn btn-sm" data-act="cancel-edit">取消</button>' +
        '<span class="cfg-hint">保存后会自动重新生成向量</span></div></div>';
    }
    var acts = "";
    acts += '<button class="btn btn-sm" data-act="edit" data-id="' + esc(m.id) + '">✏️ 编辑</button>';
    if (m._embed_failed) acts += '<button class="btn btn-sm" data-act="retry" data-id="' + esc(m.id) + '">↻ 重新嵌入</button>';
    acts += '<button class="btn btn-sm btn-danger" data-act="del" data-id="' + esc(m.id) + '">删除</button>';
    return '<div class="mem">' + head +
      '<div class="mem-text">' + esc(m.text) + "</div>" +
      (tags ? '<div class="mem-tags">' + tags + "</div>" : "") +
      '<div class="mem-acts">' + acts + "</div></div>";
  }).join("");
  setView(toolbar + html);
  bindMemToolbar();
}

function bindMemToolbar() {
  var bt = document.getElementById("bufToggle");
  if (bt) bt.onclick = function () { state.bufferOpen = !state.bufferOpen; paintMemList(); };
  var s = document.getElementById("memSearch");
  if (s) s.oninput = function () { state.filter.q = this.value; paintMemList(); refocusSearch(); };
  var t = document.getElementById("memType");
  if (t) t.onchange = function () { state.filter.type = this.value; paintMemList(); };
  var fo = document.getElementById("memFailedOnly");
  if (fo) fo.onchange = function () { state.filter.failedOnly = this.checked; paintMemList(); };
  var r = document.getElementById("retryScopeFailed");
  if (r) r.onclick = function () { startRetryAll(state.scope); };
  var df = document.getElementById("delScopeFailed");
  if (df) df.onclick = function () { deleteFailed(state.scope); };
  var rg = document.getElementById("reorgBtn");
  if (rg) rg.onclick = function () { reorganize(state.scope); };
  var rb = document.getElementById("restoreBtn");
  if (rb) rb.onclick = function () { restoreBackup(state.scope); };
  var mg = document.getElementById("migBtn");
  if (mg) mg.onclick = function () { askMigrate(state.scope); };
  var rrb = document.getElementById("reorgRetryBtn");
  if (rrb) rrb.onclick = function () { retryReorg(state.scope); };
}

/* 重绘后把光标与选区还原到搜索框，避免输入被打断 */
function refocusSearch() {
  var s = document.getElementById("memSearch");
  if (!s) return;
  s.focus();
  s.setSelectionRange(s.value.length, s.value.length);
}

async function renderProfile() {
  if (!state.scope) { setView('<div class="empty">请先在上方选择一个作用域</div>'); return; }
  setView('<div class="empty">加载中…</div>');
  var d;
  try { d = await apiGet("profile", { scope: state.scope }); }
  catch (e) { setView('<div class="empty">加载失败：' + esc(errText(e)) + "</div>"); return; }
  var p = (d && d.profile) || "";
  var body = p.trim() ? '<div class="prof-text">' + esc(p) + "</div>"
    : '<div class="empty">该作用域暂无画像。积累足够的对话总结后会自动生成。</div>';
  setView('<div class="panel"><h3>👤 用户画像</h3>' + body +
    '<div class="prof-actions"><button class="btn" id="profEdit">✏️ 编辑</button>' +
    '<button class="btn" id="profRegen" title="从现有记忆重新生成一份完整画像">🔄 从记忆重建</button>' +
    '<button class="btn btn-danger" id="profClear">清除画像</button></div>' +
    '<div id="profEditWrap" style="display:none;margin-top:13px">' +
    '<textarea id="profText" class="prof-edit"></textarea>' +
    '<div class="prof-actions"><button class="btn btn-primary" id="profSave">保存</button>' +
    '<button class="btn" id="profCancel">取消</button></div></div></div>');
  document.getElementById("profEdit").onclick = function () {
    document.getElementById("profEditWrap").style.display = "block";
    document.getElementById("profText").value = p;
    this.style.display = "none";
  };
  document.getElementById("profCancel").onclick = function () { renderProfile(); };
  document.getElementById("profRegen").onclick = function () { regenProfile(state.scope); };
  document.getElementById("profSave").onclick = async function () {
    try {
      await apiPost("profile/save", { scope: state.scope, profile: document.getElementById("profText").value });
      showToast("画像已保存", "ok"); renderProfile();
    } catch (e) { showToast("保存失败：" + errText(e), "err"); }
  };
  document.getElementById("profClear").onclick = function () {
    askConfirm("确定清除该作用域的画像？该操作不可恢复。", async function () {
      try {
        await apiPost("profile/save", { scope: state.scope, profile: "" });
        showToast("画像已清除", "ok"); renderProfile();
      } catch (e) { showToast("清除失败：" + errText(e), "err"); }
    }, "清除");
  };
}

async function renderAffection() {
  if (!state.scope) { setView('<div class="empty">请先在上方选择一个作用域</div>'); return; }
  setView('<div class="empty">加载中…</div>');
  var d;
  try { d = await apiGet("affection", { scope: state.scope }); }
  catch (e) { setView('<div class="empty">加载失败：' + esc(errText(e)) + "</div>"); return; }
  var a = (d && d.affection) || {};
  var score = Number(a.score || 0);
  var pct = Math.max(0, Math.min(100, (score + 100) / 2));
  var hist = a.history || [];
  var pts = hist.map(function (h) {
    return { label: fmtTime(h.timestamp).slice(5, 10), value: h.score_after || 0 };
  });
  var recent = hist.slice(-5).reverse().map(function (h) {
    var dl = Number(h.delta || 0), sign = dl >= 0 ? "+" : "";
    return '<div class="row" style="font-size:13px;padding:3px 0">' +
      '<span style="width:52px;font-weight:600;color:' + (dl >= 0 ? "var(--ok)" : "var(--danger)") + '">' + sign + dl.toFixed(1) + "</span>" +
      '<span style="color:var(--dim)">' + esc(h.reason || "") + "</span>" +
      '<span style="margin-left:auto;color:var(--dim);font-size:12px">' + esc(fmtTime(h.timestamp)) + "</span></div>";
  }).join("");
  var cards = [
    ["累计互动", a.total_interactions || 0],
    ["正向", a.positive_count || 0],
    ["负向", a.negative_count || 0]
  ].map(function (c) {
    return '<div class="card"><div class="num">' + esc(c[1]) + '</div><div class="lbl">' + esc(c[0]) + "</div></div>";
  }).join("");

  var est = Number(d.estimated_score || 0);
  var basis = (d && d.estimate_basis) || {};
  var estPanel = '<div class="panel"><h3>🔧 分数修正</h3>' +
    '<div class="cfg-hint">当前分数是在<b>旧规则</b>（增益系数 0.15、每日上限 10）下累积的，普遍偏低。' +
    '推荐让 <b>AI 读一遍你与它的实际互动记录</b>，重新判定关系程度——这属于「修正」，不受每日上限限制。</div>' +
    '<div class="row" style="margin-top:10px;gap:8px;flex-wrap:wrap;align-items:center">' +
    '<button class="btn btn-primary" id="affReeval">🤖 让 AI 重新评估</button>' +
    '<span class="cfg-hint">AI 会综合：累计互动 ' + (basis.total_interactions || 0) +
    ' 次（正 ' + (basis.positive_count || 0) + ' / 负 ' + (basis.negative_count || 0) +
    '）· 最近变化记录 · 记忆摘要，并给出理由</span>' +
    '</div>' +
    '<div class="cfg-hint" style="margin-top:12px;border-top:1px solid var(--border);padding-top:10px">' +
    '统计参考（按互动量用当前参数粗算，未计入每日上限，可能偏高）：' +
    '<b>' + score.toFixed(1) + '</b> → <span style="color:var(--accent)"><b>' + est.toFixed(1) + '</b>' +
    '（' + esc(d.estimate_level_cn || "") + '）</span>' +
    '<button class="btn btn-sm" id="affUseEst" style="margin-left:8px">填入</button></div>' +
    '<div class="row" style="margin-top:10px;gap:8px;flex-wrap:wrap;align-items:center">' +
    '<span class="cfg-hint">或手动设定：</span>' +
    '<input type="number" id="affSetVal" step="0.1" min="-100" max="100" value="' + score.toFixed(1) + '" style="width:110px">' +
    '<button class="btn btn-sm" id="affSave">保存分数</button>' +
    '</div></div>';

  setView('<div class="panel"><h3>❤️ 亲密度</h3><div class="aff-row">' +
    '<div><div class="aff-score" style="color:' + (score >= 0 ? "var(--ok)" : "var(--danger)") + '">' + score.toFixed(0) +
    '</div><div style="color:var(--accent);font-weight:600">' + esc(a.level_cn || "陌生人") + "</div></div>" +
    '<div class="bar-track"><i style="width:' + pct.toFixed(1) + '%"></i></div>' +
    '<div style="color:var(--dim);font-size:12px">-100 ←→ +100</div></div></div>' +
    '<div class="cards">' + cards + "</div>" +
    estPanel +
    '<div class="panel"><h3>📉 亲密度趋势</h3>' + chartLine(pts, { color: "#34d399" }) + "</div>" +
    '<div class="panel"><h3>🕒 近期变化</h3>' + (recent || '<div class="empty">暂无</div>') + "</div>");

  var reevalBtn = document.getElementById("affReeval");
  if (reevalBtn) reevalBtn.onclick = function () { reevaluateAffection(state.scope); };
  var estBtn = document.getElementById("affUseEst");
  if (estBtn) estBtn.onclick = function () {
    var el = document.getElementById("affSetVal");
    if (el) { el.value = est.toFixed(1); el.classList.add("cfg-dirty"); }
  };
  var saveBtn = document.getElementById("affSave");
  if (saveBtn) saveBtn.onclick = async function () {
    var el = document.getElementById("affSetVal");
    var v = el ? parseFloat(el.value) : NaN;
    if (isNaN(v) || v < -100 || v > 100) { showToast("请输入 -100 ~ 100 之间的数字", "err"); return; }
    askConfirm(
      "把亲密度设为 " + v.toFixed(1) + "？这会覆盖当前分数（趋势图会保留记录）。",
      async function () {
    saveBtn.disabled = true;
    try {
      var r = await apiPost("affection/set", { scope: state.scope, score: v });
      showToast("已调整：" + r.old_score + " → " + r.score + "（" + r.level_cn + "）", "ok");
      await renderAffection();
    } catch (e) {
      showToast("调整失败：" + errText(e), "err");
      saveBtn.disabled = false;
    }
      }
    );
  };
}

/* ---------- 配置页 ---------- */
var CFG_FIELDS = [
  /* ---- 提示词模板（可在此直接查看 / 编辑；点 ↺ 恢复插件自带的新版默认） ---- */
  { k: "summary_prompt_template", t: "text", label: "总结提示词模板",
    hint: "必须包含 {对话历史}。「按话题拆分输出 JSON」等硬性要求由插件自动追加，无需写在这里" },
  { k: "affection_analysis_prompt_template", t: "text", label: "亲密度评估提示词模板",
    hint: "必须包含 {对话历史}。新版要点：简短回复不等于冷淡、指出助手答错不扣分、礼貌用语不作为高亲密度依据" },
  { k: "profile_prompt_template", t: "text", label: "画像生成提示词模板",
    hint: "必须同时包含 {现有画像} 与 {记忆列表} 两个占位符" },
  /* ---- Provider 指定（留空 = 自动探测，推荐留空） ---- */
  { k: "embedding_provider_id", t: "str", label: "Embedding Provider ID",
    hint: "留空即自动探测（推荐）。仅当自动探测选错时才手动指定" },
  { k: "summary_fallback_provider_id", t: "str", label: "总结备用 LLM Provider ID",
    hint: "主 LLM 不可用时使用；留空则回退到默认聊天 Provider" },
  { k: "rerank_provider_id", t: "str", label: "Rerank Provider ID",
    hint: "仅在 enable_rerank 开启时生效；留空即自动探测" },
  { k: "affection_level_thresholds", t: "json", label: "亲密度等级阈值",
    hint: "JSON 对象，各等级上限：nemesis_max / hostile_max / cold_max / stranger_max / acquaintance_max / friend_max / close_friend_max" },
  { k: "memory_scope", t: "select", label: "记忆隔离范围", opts: ["conversation", "user", "group_user", "user_bot"],
    hint: "conversation：每个对话独立记忆（推荐）；user：同一用户全部对话共享" },
  { k: "memory_limit", t: "int", label: "每作用域记忆上限", hint: "超出上限后按淘汰策略处理" },
  { k: "memory_eviction_strategy", t: "select", label: "记忆淘汰策略", opts: ["importance", "fifo"],
    hint: "importance=按留存价值（重要性+新鲜度）淘汰，推荐；fifo=淘汰最旧的（旧版行为）" },
  { k: "retrieve_top_k", t: "int", label: "每次检索返回条数" },
  { k: "similarity_threshold", t: "float", label: "相似度阈值", hint: "0~1，越高越严格" },
  { k: "retrieve_diversity_lambda", t: "float", label: "检索多样性系数", hint: "1=只按相关性（可能召回多条同话题记忆），越小越分散；近重复记忆总会被剔除" },
  { k: "context_window_tokens", t: "int", label: "模型上下文窗口 (tokens)", hint: "0=自动检测" },
  { k: "summary_trigger_rounds", t: "int", label: "总结触发轮数" },
  { k: "summary_topic_boundary_enabled", t: "bool", label: "话题边界感知",
    hint: "收尾语（就这样/晚安）时立即总结；话题仍在延续时最多延后 3 轮——减少同一话题被切两次" },
  { k: "summary_interval_seconds", t: "int", label: "总结最小间隔 (秒)" },
  { k: "min_summary_rounds", t: "int", label: "总结最少轮数" },
  { k: "keep_raw_messages", t: "bool", label: "保留原始对话" },
  { k: "memory_inject_position", t: "select", label: "记忆注入位置", opts: ["extra_user_content", "system_prompt"],
    hint: "extra_user_content 推荐：不破坏 prompt 缓存且不落会话历史" },
  { k: "max_single_memory_chars", t: "int", label: "单条记忆注入上限 (字)" },
  { k: "max_memory_context_length", t: "int", label: "记忆上下文总长上限 (字)" },
  { k: "llm_timeout_seconds", t: "int", label: "LLM 超时 (秒)" },
  { k: "llm_max_retries", t: "int", label: "LLM 重试次数" },
  { k: "memory_decay_enabled", t: "bool", label: "记忆衰减" },
  { k: "memory_decay_lambda", t: "float", label: "衰减系数 λ" },
  { k: "memory_importance_enabled", t: "bool", label: "重要性评分" },
  { k: "memory_importance_weight", t: "float", label: "重要性权重" },
  { k: "memory_conflict_detection_enabled", t: "bool", label: "冲突自动合并" },
  { k: "chromadb_enabled", t: "bool", label: "启用 ChromaDB 后端", hint: "需 pip install chromadb" },
  { k: "profile_enabled", t: "bool", label: "启用用户画像" },
  { k: "profile_inject_position", t: "select", label: "画像注入位置", opts: ["extra_user_content", "system_prompt"] },
  { k: "profile_max_chars", t: "int", label: "画像注入最大字符数", hint: "只影响注入长度，不会再截断保存内容" },
  { k: "profile_update_interval_hours", t: "int", label: "画像更新间隔 (小时)" },
  { k: "min_profile_summaries", t: "int", label: "画像最少总结数" },
  { k: "persona_drift_enabled", t: "bool", label: "人设漂移检测" },
  { k: "affection_enabled", t: "bool", label: "启用好感度" },
  { k: "affection_eval_cooldown_seconds", t: "int", label: "好感度评估冷却 (秒)", hint: "越小评估越频繁、涨得越快" },
  { k: "affection_ema_alpha", t: "float", label: "好感度增益系数", hint: "每次评估实际落地的增量比例（0.4 ≈ LLM 给 +2 得 +0.8 分）" },
  { k: "affection_daily_cap", t: "float", label: "亲密度每日上限", hint: "一天最多涨/跌多少分（手动设定与 AI 重新评估不受此限）" },
  { k: "affection_rule_eval_rounds", t: "int", label: "规则评估间隔轮数" },
  { k: "affection_difficulty_divisor", t: "float", label: "好感度难度除数", hint: "越大曲线越平缓（越容易涨）；200 时 score=100 仍有 50% 增速" },
  { k: "affection_inject_position", t: "select", label: "好感度注入位置", opts: ["extra_user_content", "system_prompt"] },
  { k: "enable_rerank", t: "bool", label: "启用 Rerank 重排" }
];

/* 配置项分组：按键名前缀自动归类，便于定位 */
var CFG_GROUP_ORDER = ["🧠 记忆", "🔍 检索与注入", "📝 总结", "👤 用户画像",
                       "❤️ 亲密度", "🔌 模型与 Provider", "⚙️ 其他"];

function cfgGroup(k) {
  if (k.indexOf("affection") === 0) return "❤️ 亲密度";
  if (k.indexOf("profile") === 0) return "👤 用户画像";
  if (k.indexOf("summary") === 0 || k.indexOf("min_summary") === 0) return "📝 总结";
  if (k.indexOf("memory") === 0) return "🧠 记忆";
  if (/^(retrieve|similarity|context_window|chromadb|enable_rerank|max_)/.test(k)) return "🔍 检索与注入";
  if (/provider|^llm_/.test(k)) return "🔌 模型与 Provider";
  return "⚙️ 其他";
}

/* 把当前表单里的值收回 state（搜索重绘时不丢失未保存的修改） */
function harvestCfg() {
  CFG_FIELDS.forEach(function (f) {
    var elx = document.querySelector('[data-k="' + f.k + '"]');
    if (!elx) return;
    if (f.t === "bool") state.cfg[f.k] = elx.checked;
    else if (f.t === "text" || f.t === "str") state.cfg[f.k] = elx.value;
    else if (f.t === "json") {
      try { state.cfg[f.k] = JSON.parse(elx.value); }
      catch (ej) { state.cfg[f.k] = elx.value; }  // 非法 JSON 先原样存，保存时报错
    }
    else if (f.t === "select") state.cfg[f.k] = elx.value;
    else if (f.t === "float") { var n = parseFloat(elx.value); if (!isNaN(n)) state.cfg[f.k] = n; }
    else { var m = parseInt(elx.value, 10); if (!isNaN(m)) state.cfg[f.k] = m; }
  });
}

function cfgItemHtml(f) {
  var val = state.cfg[f.k];
  var dv = state.cfgDefaults[f.k];
  var control = "";
  if (f.t === "text") {
    // 提示词模板：多行编辑，独占整行
    var dvLen = dv === undefined || dv === null ? 0 : String(dv).length;
    var curLen = val === undefined || val === null ? 0 : String(val).length;
    var changedT = dv !== undefined && dv !== null && String(val) !== String(dv);
    var badgeT = changedT ? ' <span class="badge warn" title="当前值与默认值不同">已改</span>' : "";
    var resetT = dv === undefined ? "" :
      '<button type="button" class="btn btn-sm cfg-reset" data-reset="' + f.k +
      '" title="填入插件自带的新版默认提示词">↺ 恢复默认</button>';
    var hintT = f.hint ? '<div class="cfg-hint">' + esc(f.hint) + "</div>" : "";
    return '<div class="cfg-item cfg-item-wide"><label>' + esc(f.label) + badgeT + "</label>" +
      '<div class="cfg-key">' + esc(f.k) + "</div>" +
      '<textarea data-k="' + f.k + '" spellcheck="false">' +
      esc(val === null || val === undefined ? "" : val) + "</textarea>" +
      '<div class="row" style="margin-top:6px;gap:8px;align-items:center;flex-wrap:wrap">' +
      resetT + '<span class="cfg-hint">当前 ' + curLen + " 字符 · 默认 " + dvLen +
      " 字符 · 上限 10240" + "</span></div>" + hintT + "</div>";
  }
  if (f.t === "json") {
    var jsonTxt = "";
    try { jsonTxt = JSON.stringify(val === undefined ? {} : val, null, 2); }
    catch (e2) { jsonTxt = String(val === undefined ? "" : val); }
    var changedJ = dv !== undefined && dv !== null && String(val) !== String(dv);
    var badgeJ = changedJ ? ' <span class="badge warn">已改</span>' : "";
    var resetJ = dv === undefined ? "" :
      '<button type="button" class="btn btn-sm cfg-reset" data-reset="' + f.k + '" title="填入默认值">↺ 恢复默认</button>';
    return '<div class="cfg-item cfg-item-wide"><label>' + esc(f.label) + badgeJ + "</label>" +
      '<div class="cfg-key">' + esc(f.k) + "</div>" +
      '<textarea data-k="' + f.k + '" spellcheck="false">' + esc(jsonTxt) + "</textarea>" +
      '<div class="row" style="margin-top:6px">' + resetJ +
      '<span class="cfg-hint">JSON 格式，保存时会校验</span></div>' +
      (f.hint ? '<div class="cfg-hint">' + esc(f.hint) + "</div>" : "") + "</div>";
  }
  if (f.t === "bool") {
    control = '<input type="checkbox" data-k="' + f.k + '"' + (val ? " checked" : "") + ">";
  } else if (f.t === "str") {
    control = '<input type="text" data-k="' + f.k + '" placeholder="留空 = 自动" value="' +
      esc(val === null || val === undefined ? "" : val) + '">';
  } else if (f.t === "select") {
    control = '<select data-k="' + f.k + '">' + (f.opts || []).map(function (o) {
      return '<option value="' + o + '"' + (String(val) === o ? " selected" : "") + ">" + o + "</option>";
    }).join("") + "</select>";
  } else {
    var step = f.t === "float" ? ' step="0.01"' : "";
    control = '<input type="number" data-k="' + f.k + '" value="' +
      esc(val === null || val === undefined ? "" : val) + '"' + step + ">";
  }
  var changed = dv !== undefined && dv !== null && String(val) !== String(dv);
  var badge = changed ? ' <span class="badge warn" title="当前值与默认值不同">已改</span>' : "";
  var dvStr = dv === undefined || dv === null ? "" : String(dv);
  var dvTip = dvStr.length > 40 ? dvStr.slice(0, 40) + "…" : dvStr;
  var reset = dv === undefined ?
    "" : '<button type="button" class="btn btn-sm cfg-reset" data-reset="' + f.k +
         '" title="填回默认值：' + esc(dvTip) + '">↺</button>';
  var hint = f.hint ? '<div class="cfg-hint">' + esc(f.hint) + "</div>" : "";
  // 中文名 + 英文键名同时显示：配置页原文只显示中文描述，与文档里的键名对不上
  return '<div class="cfg-item"><label>' + esc(f.label) + badge + "</label>" +
    '<div class="cfg-key">' + esc(f.k) + "</div>" +
    '<div class="cfg-ctl">' + control + reset + "</div>" + hint + "</div>";
}

function renderCfgFields() {
  var q = (state.cfgQuery || "").trim().toLowerCase();
  var onlyChanged = !!state.cfgOnlyChanged;
  var shown = CFG_FIELDS.filter(function (f) {
    if (onlyChanged) {
      var dv = state.cfgDefaults[f.k];
      if (dv === undefined || String(state.cfg[f.k]) === String(dv)) return false;
    }
    if (!q) return true;
    return (f.label || "").toLowerCase().indexOf(q) >= 0 || f.k.toLowerCase().indexOf(q) >= 0;
  });
  var byGroup = {};
  shown.forEach(function (f) {
    var g = cfgGroup(f.k);
    (byGroup[g] = byGroup[g] || []).push(f);
  });
  var html = "";
  CFG_GROUP_ORDER.forEach(function (g) {
    var list = byGroup[g];
    if (!list || !list.length) return;
    html += '<div class="cfg-group-title">' + g +
      ' <span class="cfg-hint">' + list.length + " 项</span></div>";
    html += '<div class="cfg-grid">' + list.map(cfgItemHtml).join("") + "</div>";
  });
  return html || '<div class="empty">没有匹配的配置项</div>';
}

/* 推荐参数：这些项的推荐值即插件默认值（升级后引入的新默认）。
 * 只把值填进表单，不自动保存——用户仍需点「保存配置」确认。 */
var RECOMMENDED_KEYS = [
  "affection_ema_alpha", "affection_difficulty_divisor", "affection_daily_cap",
  "affection_eval_cooldown_seconds", "profile_max_chars",
  "memory_eviction_strategy", "retrieve_diversity_lambda",
  "summary_topic_boundary_enabled", "llm_timeout_seconds",
];

function paintCfg() {
  var body = document.getElementById("cfgBody");
  if (body) body.innerHTML = renderCfgFields();
  bindCfgResets();
}

function bindCfgResets() {
  Array.prototype.forEach.call(document.querySelectorAll(".cfg-reset"), function (b) {
    b.onclick = function () {
      harvestCfg();
      var key = b.getAttribute("data-reset");
      var dv = state.cfgDefaults[key];
      if (dv === undefined || dv === null) { showToast("该项没有可用的默认值", "err"); return; }
      state.cfg[key] = dv;
      paintCfg();
      showToast("已填入默认值，记得点「保存配置」", "ok");
    };
  });
}

async function renderConfig() {
  setView('<div class="empty">加载中…</div>');
  var d, dd = null;
  try { d = await apiGet("config"); }
  catch (e) { setView('<div class="empty">加载失败：' + esc(errText(e)) + "</div>"); return; }
  try { dd = await apiGet("config/defaults"); } catch (e) { dd = null; }
  state.cfg = (d && d.config) || {};
  state.cfgDefaults = (dd && dd.defaults) || {};
  state.cfgQuery = "";
  state.cfgOnlyChanged = false;
  setView('<div class="panel"><h3>⚙️ 配置</h3>' +
    '<div class="row" style="gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:12px">' +
    '<input type="search" id="cfgSearch" style="flex:1;min-width:220px" ' +
    'placeholder="搜索：中文名 或 英文键名（如「每日上限」/ affection_daily_cap）">' +
    '<button class="btn btn-sm" id="cfgOnlyChanged">只看已改</button>' +
    '<button class="btn btn-sm" id="cfgRecommended" title="把亲密度增益系数/难度除数/每日上限等填为插件推荐值（填完仍需点保存）">套用推荐参数</button>' +
    '<span class="cfg-hint" id="cfgCount"></span></div>' +
    '<div id="cfgBody">' + renderCfgFields() + "</div>" +
    '<div class="prof-actions"><button class="btn btn-primary" id="cfgSave">保存配置</button></div>' +
    '<div class="cfg-hint" style="margin-top:10px">每项都标注了<b>中文名</b>与<b>英文键名</b>（灰色等宽字），' +
    '方便与文档 / 日志对照；标了「已改」的表示当前值与默认值不同。' +
    '↺ = 把该项填回默认值（改完点「保存配置」）。' +
    '这里刻意没有做"全部恢复默认"：那会把 画像 / 亲密度 / 人设漂移 等默认值为 false 的开关一起关掉。</div></div>');
  bindCfgResets();
  var search = document.getElementById("cfgSearch");
  if (search) search.oninput = function () { harvestCfg(); state.cfgQuery = search.value; paintCfg(); };
  var recBtn = document.getElementById("cfgRecommended");
  if (recBtn) recBtn.onclick = function () {
    harvestCfg();
    var changed = [];
    RECOMMENDED_KEYS.forEach(function (k) {
      var dv = state.cfgDefaults[k];
      if (dv === undefined || dv === null) return;
      if (String(state.cfg[k]) === String(dv)) return;
      changed.push(k + ": " + state.cfg[k] + " → " + dv);
      state.cfg[k] = dv;
    });
    paintCfg();
    if (!changed.length) { showToast("推荐参数已全部生效，无需改动", "ok"); return; }
    showToast("已填入 " + changed.length + " 项推荐值，请确认后点「保存配置」：" + changed.join("；"), "ok");
  };
  var onlyBtn = document.getElementById("cfgOnlyChanged");
  if (onlyBtn) onlyBtn.onclick = function () {
    harvestCfg();
    state.cfgOnlyChanged = !state.cfgOnlyChanged;
    onlyBtn.classList.toggle("btn-primary", state.cfgOnlyChanged);
    paintCfg();
  };
  document.getElementById("cfgSave").onclick = async function () {
    // 先把手填的值收回 state，再从 state 组装请求体——
    // 否则在搜索过滤状态下保存时，被过滤掉的字段会整批漏掉。
    harvestCfg();
    var body = {};
    var badJson = [];
    CFG_FIELDS.forEach(function (f) {
      if (state.cfg[f.k] === undefined) return;
      if (f.t === "json" && typeof state.cfg[f.k] === "string") { badJson.push(f.k); return; }
      body[f.k] = state.cfg[f.k];
    });
    if (badJson.length) {
      showToast("❌ " + badJson.join(" / ") + " 不是合法 JSON，已取消保存", "err");
      return;
    }
    try {
      var r = await apiPost("config/save", { config: body });
      showToast("✅ 配置已保存" + (r && r.updated_keys ? "（" + r.updated_keys.length + " 项）" : ""), "ok");
      await loadScopes();
    } catch (e) { showToast("保存失败：" + errText(e), "err"); }
  };
}

/* ---------- 事件绑定 ---------- */
function bindEvents() {
  document.getElementById("tabs").addEventListener("click", function (ev) {
    var t = ev.target.closest ? ev.target.closest(".tab") : null;
    if (!t) return;
    document.querySelectorAll(".tab").forEach(function (x) { x.classList.remove("active"); });
    t.classList.add("active");
    state.tab = t.getAttribute("data-tab");
    render();
  });

  document.getElementById("scopeSelect").addEventListener("change", function () {
    state.scope = this.value;
    render();
  });

  document.getElementById("refreshBtn").onclick = async function () {
    try { await loadScopes(); render(); showToast("已刷新", "ok"); }
    catch (e) { showToast("刷新失败：" + errText(e), "err"); }
  };

  /* 记忆操作：事件委托（避免 inline onclick 引号转义问题） */
  document.getElementById("view").addEventListener("click", async function (ev) {
    var btn = ev.target.closest ? ev.target.closest("[data-act]") : null;
    if (!btn) return;
    var act = btn.getAttribute("data-act");
    var mid = btn.getAttribute("data-id");

    // 编辑相关：不需要记忆 id 也能操作
    if (act === "cancel-edit") { state.editingId = ""; paintMemList(); return; }
    if (act === "edit") { state.editingId = mid; paintMemList(); return; }
    if (act === "save-edit") {
      var ta = document.getElementById("memEditText");
      var sel = document.getElementById("memEditImp");
      if (!ta) return;
      btn.disabled = true;
      try {
        var resp = await apiPost("memory/update", {
          scope: state.scope, id: mid,
          text: ta.value,
          importance: sel ? parseInt(sel.value, 10) : undefined
        });
        state.editingId = "";
        showToast(
          resp.text_changed
            ? (resp.reembedded ? "已保存并重新生成向量" : "已保存，但补嵌失败（可用一键补嵌修复）")
            : "已保存",
          resp.text_changed && !resp.reembedded ? "err" : "ok"
        );
        await renderMemories();
      } catch (e) {
        showToast("保存失败：" + errText(e), "err");
        btn.disabled = false;
      }
      return;
    }

    if (!mid) return;
    btn.disabled = true;
    if (act === "retry") {
      try {
        await apiPost("memory/retry", { scope: state.scope, id: mid });
        showToast("✅ 嵌入成功，该记忆已可参与检索", "ok");
        await renderMemories();
      } catch (e) {
        showToast("❌ 嵌入失败：" + errText(e), "err");
        btn.disabled = false;
      }
    } else if (act === "del") {
      btn.disabled = false;  // 交给确认框，先恢复按钮状态
      askConfirm("确定删除这条记忆？该操作不可恢复。", async function () {
      try {
        await apiPost("memory/delete", { scope: state.scope, id: mid });
        showToast("已删除", "ok");
        await loadScopes();
        await renderMemories();
      } catch (e) {
        showToast("删除失败：" + errText(e), "err");
        btn.disabled = false;
      }
      }, "删除");
    }
  });
}

/* ---------- 启动 ---------- */
async function boot() {
  if (!bridge) {
    setView('<div class="empty">请在 AstrBot Dashboard 中打开本页面（当前未检测到插件页面 bridge）。</div>');
    return;
  }
  try { await bridge.ready(); } catch (e) { /* 拿不到上下文也继续 */ }
  bindEvents();
  try {
    await loadScopes();
    render();
  } catch (e) {
    setView('<div class="empty">加载失败：' + esc(errText(e)) + "</div>");
  }
  // 若上次离开时仍有补嵌任务在跑，自动接续进度显示
  try {
    var st = await apiGet("job/status");
    var jobs = (st && st.jobs) || {};
    var names = Object.keys(jobs);
    for (var i = 0; i < names.length; i++) {
      if (jobs[names[i]] && jobs[names[i]].running) {
        watchedJobs[names[i]] = true;
        showJobBar(names[i], jobs[names[i]]);
      }
    }
    if (Object.keys(watchedJobs).length) pollJobs();
  } catch (e) { /* 忽略 */ }
}
boot();
