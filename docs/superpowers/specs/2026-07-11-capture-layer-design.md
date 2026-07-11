# kb 采集层设计(clip + inbox)

日期:2026-07-11
状态:待用户批准

## 背景与问题

kb 的检索端(语义/hybrid 检索、CJK、增量索引、带引用的 `kb ask`)已经完整,
但摄入端只接受**磁盘上已存在的文件**和 Chrome 书签文件。用户真实的信息流
——网页文章、AI 对话结论、视频内容——都困在浏览器和手机里,要进入 kb
必须手动落地成文件。这个摩擦导致知识库变薄,查询没结果,进而没人查询。

**采集是这个项目当前的瓶颈,不是检索。**

用户环境:PC(Windows)+ 手机,Obsidian vault 已在两端同步。

## 目标

1. 电脑上:复制一个 URL,一条命令把正文抓下来存进知识库。
2. 手机上:把 URL 分享进 Obsidian 的 Inbox 文件夹,回到 PC 后自动补全正文并索引。
3. 纯文字片段(如 AI 对话结论)可以直接粘贴收集。

## 非目标(v1 不做)

- 浏览器扩展 / 本地 HTTP 端口(以后可以作为往收件箱投 URL 的另一个入口加上)
- B 站字幕抓取(留接口,v1 不实现)
- 微信等封闭平台的内容抓取
- 网页快照 / 图片存档(只存正文文字)

## 总体架构

三个新部件,全部搭在现有的缝上:

```
手机: 分享 URL → Obsidian Inbox/ ──(Obsidian 同步)──┐
                                                      ▼
PC:   kb clip <url> ──→ fetch.py ──→ vault/Clips/*.md ──→ 现有 ingest 索引
                            ▲
      kb watch ─ inbox 处理步 ─ 发现裸 URL 笔记 → 原地展开
```

### 1. `kb/fetch.py` — 内容抓取器(新模块)

单一入口:

```python
@dataclass
class FetchedDoc:
    title: str        # 提取的标题,可能为空
    text: str         # 正文(markdown/纯文本)
    url: str          # 原始 URL
    fetched_at: float # unix 时间戳

def fetch_url(url: str, timeout: float = 15.0) -> FetchedDoc | None
```

按 URL 分派:

- **YouTube**(`youtube.com/watch`、`youtu.be`)→ `youtube-transcript-api`
  拉字幕,标题从页面 oEmbed 端点取(无 API key)。
- **其余 HTTP(S) URL** → 抓 HTML,`trafilatura` 提取正文与标题。

依赖放进新的 optional extra `[clip]`(`trafilatura`、`youtube-transcript-api`),
与 `[documents]` / `[synthesis]` 同款模式;核心包纯 Python 红线不变。
库缺失时 `fetch_url` 返回 `None` 并由调用方给出安装提示
(与 `_extract_pdf` 的处理方式一致)。

所有网络/解析错误一律返回 `None`,不抛异常——沿用代码库防御姿态。
超时默认 15 秒。

### 2. `kb clip` — 电脑端采集命令(CLI 新子命令)

```sh
kb clip --set-dir "D:/vault/Clips"   # 一次性配置 clips 目录(写入 config.json 的 clips_dir 键)
kb clip https://example.com/article  # 抓取并保存
kb clip --text "标题" < note.txt      # 从 stdin 收纯文字
```

行为:

- URL 模式:`fetch_url` → 存为 `Clips/<slug>.md`,frontmatter 含
  `title`、`url`、`clipped`(ISO 日期)。文件名 slug 取标题前 60 字符
  (CJK 保留),冲突时追加 `-2`、`-3`。同一 URL 已存在(按 frontmatter
  的 `url` 匹配)则跳过并提示。
- 文字模式:stdin 内容存为笔记,`--text` 的参数作为标题。
- 保存后对 clips 目录跑一次增量 `ingest`,并在首次使用时把它注册为
  `files` source(此后 query/watch 自然覆盖)。
- `clips_dir` 未配置时报错并提示 `--set-dir`。解析顺序:`$KB_CLIPS_DIR`
  → `config.json` 的 `clips_dir`(与 `synthesis_model` 的模式一致)。

### 3. 收件箱处理 — 手机链路

**注册**:`kb add --inbox <path>` 把文件夹以新 kind `"inbox"` 写入
sources.json(`_VALID_KINDS` 增加 `"inbox"`)。inbox 文件夹同时按 files
source 参与索引,因此单独使用也成立;通常它就在已注册的 vault 里。

**识别**:一个 `.md` 文件被认定为"裸 URL 笔记",当且仅当去掉 YAML
frontmatter 和空行后,内容恰好是**一行**,且该行是一个裸 URL 或只含
URL 的 markdown 链接 `[标题](url)`。任何带正文的笔记绝不改动。

**展开(原地)**:抓取成功后重写该笔记:

```markdown
---
title: <提取的标题>
url: <原 URL>
clipped: 2026-07-11
kb-clipped: true
---

<正文>
```

`kb-clipped: true` 防止重复处理。展开后的笔记经 Obsidian 同步回手机,
顺带获得"稍后读"体验。

**重试与放弃**:抓取失败的笔记保持原样,下一轮 watch 重试。失败计数
记录在 `<kb_home>/inbox_state.json`(`路径 → 失败次数`);失败满 3 次
后在笔记 frontmatter 写入 `kb-clip-failed: true` 并停止重试(用户删掉
该键可重新触发)。

**接入 watch**:`watch.run_once` 在快照/ingest 之前先跑一个 inbox 处理
步(新模块 `kb/inbox.py` 的 `process_inbox(folder) -> int`,返回处理条
数)。展开产生的文件变化被同一轮的快照对比捕获,自然触发重新索引。
inbox 步骤的任何失败(含离线)只打印警告,不影响 watch 主循环。

### 4. 配置变更汇总

| 位置 | 变更 |
| --- | --- |
| `config.json` | 新键 `clips_dir`(字符串路径) |
| 环境变量 | `KB_CLIPS_DIR` 覆盖 `clips_dir` |
| `sources.json` | `kind` 新增合法值 `"inbox"` |
| `pyproject.toml` | 新 extra `[clip]`:`trafilatura`、`youtube-transcript-api` |

## 错误处理原则

- 网络不可达 / 超时 / 解析失败 → `fetch_url` 返回 `None`;CLI 报错退出,
  watch 中则计入重试并继续。
- 只有严格匹配"裸 URL 笔记"的文件会被改写;识别器测试覆盖各种边界
  (带正文、多行、frontmatter 已含 kb-clipped、空文件)。
- clips 目录不可写、inbox 文件夹消失 → 打印警告跳过,不中断。

## 测试策略

全程不打真网络:

1. `fetch.py`:用本地 HTML fixture 测 trafilatura 提取路径;URL 分派逻辑
   (YouTube vs 普通网页)纯函数测试;库缺失时返回 `None`。
2. 裸 URL 识别器:表驱动测试(裸 URL / markdown 链接 / 带正文 / 已处理 /
   空文件 / 只有 frontmatter)。
3. `process_inbox`:临时目录 + mock 的 `fetch_url`,验证原地展开、
   frontmatter 写入、失败计数与放弃。
4. `kb clip`:mock `fetch_url`,验证文件落地、slug 生成、URL 去重、
   首次自动注册 source。
5. `watch.run_once`:inbox 步骤挂入后,展开→快照变化→ingest 的联动。

## 实施顺序

1. `fetch.py` + 测试(fixture 驱动)
2. `kb clip` 命令 + `clips_dir` 配置
3. `"inbox"` source kind + `inbox.py` + watch 接入
4. README 文档(Daily workflow 增补手机采集流程)
