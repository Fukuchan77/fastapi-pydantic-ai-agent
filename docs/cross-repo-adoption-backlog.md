# 相互取り込み backlog（fastapi-pydantic-ai-agent）

Agentic AI 系 5 リポジトリを横断で突き合わせた検証の、本 repo 向け抜粋。
**根拠・実測値・全 16 項目の本文は `vaz-agentic-ai-next/docs/cross-repo-adoption-review.md` が正本**。
本文書は重複させず、項目 ID（X-n）で参照する。

先行する片方向レビュー（[reference-repo-review.md](reference-repo-review.md) /
[pydantic-ai-sandbox-comparison-review.md](pydantic-ai-sandbox-comparison-review.md)）はいずれも
「本 repo が何を取り込むか」の**消費側**視点だった。今回の横断レビューは
**本 repo が何を出すか**を初めて明示している。

- 検証日: 2026-09-06
- 兄弟 repo: `beeai-agentic-ai-sandbox` / `pydantic-ai-sandbox` / `vaz-ai-next` / `vaz-agentic-ai-next`

---

## 1. この repo が出す資産

5 repo 中、**リポジトリガードとサプライチェーン規律は本 repo が最も厚い**（`CLAUDE.md` の "Repo guards" 節が列挙するガードテスト 17 件、
Actions SHA 固定 5/5 は 5 repo 中ここだけ）。

| 資産 | 場所 | 何が独自か |
|---|---|---|
| CI ワークフローのガードテスト | `tests/unit/test_ci_workflows.py`（224 行） | YAML を静的パースし全 `uses:` を `_FULL_SHA_PIN_RE`（40 桁）で検証。PyYAML が `on:` を `True` にする癖も `_load_workflow()` で吸収。**他 4 repo すべての X-1 の移植元** |
| ネットワーク遮断ガード | `tests/support/hermetic.py`（47 行） | `AF_INET`/`AF_INET6` の `socket.connect` のみ遮断し `AF_UNIX`（asyncio self-pipe）は素通し。`tests/unit/` に autouse |
| 履歴トリムの不変条件 | `app/stores/session_store/_trim.py`（78 行） | メッセージ境界でのみ切る／tool-call ペアを孤児化しない（理想切点から**前方**探索＝空 tail が解の存在を保証）／`messages[0]` は必ず残す（結果が `max+1` になりうる理由）。**5 repo 中最も厳密な機械トリマ** |
| SSE ライフサイクルの 3 罠 | `app/api/v1/_stream.py:237-304` / `app/api/v1/agent.py:159,175` / `app/patterns/sse.py:6,103` | (1) anyio cancel scope はタスク跨ぎ不可 → 単一の永続駆動タスク、(2) ハートビートは `asyncio.wait()`、(3) U+2028/2029 のため `str.splitlines()` 不可。**再発見のコストが極めて高い知見** |
| ツール設計規約 ＋ enforcement | `docs/tool-design-conventions.md`（144 行）＋ `.pre-commit-config.yaml` の `real-tool-conventions-guard` | 実ツールが現れた瞬間に発火する reminder-and-block スタブ。「静的検証できないものは人間のレビューへ回す」と明示している設計 |
| LLM Top 10 対応表 | `docs/owasp-agentic-llm-mapping.md`（43 行） | 全 10 行に状態（Mitigated / Partial・accepted / Accepted）＋実装モジュール＋**テストファイル**を引用。散文でなく検証可能な主張 |
| Dependabot の `ignore:` 設計 | `.github/dependabot.yml` ＋ `tests/unit/test_dependabot_config.py` | **`fastapi` は minor も major も無視**（Dependabot は 0.x を patch 位置で分類するため `0.136→0.137` が minor 扱い。major だけ無視すると素通りする） |
| 依存境界の強制 | `tests/unit/test_config_dependency_bounds.py` | 全 production 依存に上限宣言を要求 |
| 抑止リストの罠の記録 | `CLAUDE.md`（`pip-audit` 節） | シェルの行継続 `\` が 1 つ落ちると `--ignore-vuln` リストが黙って切り詰められる。抑止リストを持つ全 repo に効く警告 |

## 2. この repo が取り込む項目

S マス 5 件（X-6 / X-2 / X-12 / X-13 / X-16）は 2026-09-08 の作業で実装済み。
M マス 3 件は項目ごとに別 PR で対応: X-9 / X-7 は実装済み。X-8 は未着手（後続 PR）。

| ID | 状態 | 内容 | 出所 | 工数 | 受け入れ条件 |
|---|---|---|---|---|---|
| **X-6** | ✅ 済 | ツール設計規約の**実装**を得る。`docs/tool-design-conventions.md` は `Status: deferred` で、実ツールは 0 件（唯一のツールは `tools_mock.py`） | `pydantic-ai-sandbox/patterns/frameworks/pydantic-ai/src/patterns_pydantic_ai/tool_design.py`（215 行）— `directory_search`/`directory_get`、`_DEFAULT_LIMIT=5`/`_MAX_LIMIT=25`/`_DETAIL_NOTE_CHARS=80`、最終ページで `next_offset=None`、`_coerce_format()`/`_clamp_int()` | S | 初の実ツール設計時の雛形として参照。`real-tool-conventions-guard` が発火したときの参照先を本文書経由で辿れること — [`app/agents/examples/tool_design_reference.py`](../app/agents/examples/tool_design_reference.py)（`@agent.tool` 非装飾・未登録、`docs/tool-design-conventions.md` からリンク、`tests/unit/agents/test_tool_design_reference.py` で回帰検証） |
| X-8 | 未着手 | eval に `Judge` Protocol シームと PR ゲート運用を追加。現状 grader（`evals/graders.py`）はあるが judge の DI シームが無く、evals は pre-push のみで CI 回帰比較が無い | (a) `pydantic-ai-sandbox/patterns/contracts/src/patterns_contracts/eval_graders.py`（93 行、`Rating` に `"unknown"`／`Judge[SubjectT]` Protocol／決定論フェイク judge でネットワーク不要）、(b) `vaz-ai-next/packages/evals/src/pr-gate.ts`（ベースライン差分・over-under-trigger balance・case あたりコスト/レイテンシ・20 件未満は `reportOnly`） | M | (a) は `evals` がネットワーク無しでユニットテスト可能になること。(b) は実モデル課金を伴うため opt-in（ラベル方式）で |
| **X-9** | ✅ 済 | HITL に「サーバ側履歴が正」の構造的封鎖を追加。セッション所有権（HMAC）は**別の**攻撃面を塞いでおり、履歴注入は未対処 | `pydantic-ai-sandbox/patterns/hitl/src/patterns_hitl/app.py:92,157` — リクエストモデルが `extra="forbid"` で `message_history` を持たない（CVE-2026-25580 の経路を構造的に封鎖）。加えて consume-once ステートマシン | M | クライアントが履歴を注入できないことをテストで証明。`AuditRecord` の principal 帰属は維持 — **調査の結果、履歴注入は着手前から `ChatRequest`/`IngestRequest`/`RAGQueryRequest` のいずれにも history/usage/model フィールドが無いことにより既に構造的に不可能だった**（保証は by-omission）。本 PR は `extra="forbid"` を全 3 モデルに追加し、保証を by-construction に格上げ（未知フィールドは 422、黙って捨てない）。`tests/unit/models/test_request_model_strictness.py`（新規、モデル列挙型ガード）と `tests/e2e/test_agent_chat.py`/`tests/e2e/test_rag.py` の注入拒否テストで証明。`AuditRecord` に `principal` フィールドを追加し `_principal_of(deps)` で帰属を明示化（従来は `AgentDeps.principal` にしかなく `AuditRecord` 自体には帰属情報が無かった）。**consume-once ステートマシンは意図的に不採用**: `SessionStore` は Constitution Principle 2 の Protocol 拡張点であり、状態機械メソッドの追加は外部実装への破壊的変更になるため。同一セッションでの並行ターンの last-write-wins は既知の未解決事項として残る |
| **X-2** | ✅ 済 | 自 repo の `hermetic.py` に `connect_ex` と `getaddrinfo` を追加（**DNS 解決だけ通る穴**が残っている） | `pydantic-ai-sandbox/patterns/deep-research/tests/unit/conftest.py:43-49` | S | `test_block_network.py` に新経路の回帰テストを追加 — `tests/support/hermetic.py::block_network()` が `connect`/`connect_ex`/`getaddrinfo` の3経路すべてを遮断、`tests/unit/test_block_network.py` に2件追加 |
| **X-7** | ✅ 済 | コンテキスト管理の残り 2 脚 | (a) `vaz-ai-next/docs/context-budget.md` の段階導入設計（Stage 0/1/2、未指定時 byte 等価のシーム）、(b) `pydantic-ai-sandbox/patterns/deep-research/src/patterns_deep_research/notes.py` の構造化ノート＋`digest_fn` DI シーム | M | 本 repo の `_trim.py` は機械トリムの脚を既に持つ。(a)(b) は導入時に `_trim.py` の不変条件を壊さないこと — `docs/context-budget.md`（新規）に Stage 0/1/2 を明文化。Stage 1 として `SessionStore` 両実装に `history_compactor: HistoryCompactor \| None = None` シームを追加（`_trim.py` に型定義）、`save_history()` で `trim_history()` の**前**に適用、`trim_history()` が最終決定権を持つため既存不変条件は無傷。未設定時（現状の全呼び出し箇所）は Stage 0 と byte 等価 — `tests/unit/stores/test_session_trim_across_backends.py` は無改変のまま緑（Stage 0 の証明）。具体的な compactor 実装（(b) の `ResearchNote`/`distill_notes` 相当）は**未移植**、シームのみ。副次的に RAG 側のハードコード `max_chars=15000`（3 箇所）を `Settings.rag_prompt_max_chars` に設定可能化 |
| **X-12** | ✅ 済 | 抑止方針の段を明示 | `beeai-agentic-ai-sandbox/SECURITY-NOTES.md`（`--ignore-vuln` を使わない最厳格）／`pydantic-ai-sandbox`（日付付きレビュー期限＋追跡参照を必須）／本 repo（1 件ごとの到達可能性理由） | S | 3 段のどれを採るかを明示。`pydantic-ai-sandbox` の**日付付き期限**は本 repo の理由付き抑止に追加する価値がある（現状は期限が無く、無期限に残りうる） — `CLAUDE.md`（pip-audit 節）＋ `AGENTS.md` に3段を明記し、本 repo が tier 3（理由付き・期限無し）であること、tier 2 の日付付き期限が未実装のギャップであることを記録（期限機構自体は未実装） |
| **X-13** | ✅ 済 | OWASP 表を Agentic AI Top 10 まで拡張 | `pydantic-ai-sandbox/patterns/SECURITY-NOTES.md`（Agentic AI Top 10 2025-12 をレイヤ別に） | S | LLM Top 10 は agentic 自律性を LLM06 に畳んでいるため**両表は重複しない**。既存の「各行にテストを引用する」形式を維持 — `docs/owasp-agentic-llm-mapping.md` に ASI01–ASI10（OWASP Top 10 for Agentic Applications, 2025-12）の対応表を追加、全10行に本 repo 固有の実装モジュール＋テストファイルを引用 |
| **X-16** | ✅ 済 | 6 パターン教材への参照 | `beeai-agentic-ai-sandbox/effective_agents/`（[A1] の 6 パターン ＋ `research_system.py:240` の `_print_usage()` によるトークン内訳可視化）、`pydantic-ai-sandbox/patterns/deep-research/COMPARISON.md` | S | **実装は不要**。多エージェント採用を検討する際のゲート判断材料へのリンクを持てばよい — `CLAUDE.md`／`AGENTS.md` に「Multi-agent adoption gate」として参照リンクを追加（実装なし） |

## 3. この repo 固有の注意

- **`CLAUDE.md` と `AGENTS.md` は 1 変更単位**（Req 8.2）。本文書へのリンクを
  `CLAUDE.md` の "Design docs" に追加する場合、`AGENTS.md` も同じコミットで編集すること。
- **`tests/unit/test_file_size_policy.py` が `app/**` / `tests/**` の 1000 行以上を hard fail する**
  （500-999 行は警告）。X-6 / X-9 の取り込みでファイルが伸びる場合は
  参照分割（`app/config/` = ドメイン別 mixin、`app/workflows/` = 関心別 mixin、
  `app/stores/*/` = protocol ＋ backend 別ファイル）に従うこと。
- **`test_contract_drift.py` は一方向**（README は新しいメンバを省略してよいが、
  もう存在しないものを見せてはいけない）。本文書は README の規範フェンスではないため対象外。
- **`Settings.api_key` が 1 個であることに依存している箇所が 2 つある**
  （`/v1/rag/ingest` の principal 未束縛、CRAG 結果キャッシュのキー）。
  X-9 で HITL を触る際、第 2 の鍵を導入する変更と混ぜないこと
  （`app/security/principal.py` のモジュール docstring が正本）。
- **`filterwarnings = ["error::DeprecationWarning"]` は素の設定**であり、
  Python バージョン変更・依存 bump のたびに ignore リストの再棚卸しが必要。
  X-2 / X-8 で新しい依存（テスト用含む）を入れる場合はこの影響を確認する。
