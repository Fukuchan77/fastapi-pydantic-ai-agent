# 参照リポジトリ設計レビュー(Reference Repository Design Review)

Agentic AI アプリ開発のため、以下の参照リポジトリを調査し、本プロジェクト
`fastapi-pydantic-ai-agent` に取り込む価値のある機能・設計を検証した結果をまとめる。

| リポジトリ | 調査時コミット | 性格 | 採用価値 |
|---|---|---|---|
| [agentic-ai-fastapi-playground](https://github.com/Fukuchan77/agentic-ai-fastapi-playground) | `e5ac43b` | ツール設定のみのスキャフォールド | 低(設定のみ) |
| [agentic-ai-bootcamp](https://github.com/Fukuchan77/agentic-ai-bootcamp) | `26e6b09` | Pydantic AI 学習カリキュラム(12 レッスン + LangGraph / LlamaIndex 比較トラック) | 中〜高 |
| [agentic-ai-sandbox](https://github.com/Fukuchan77/agentic-ai-sandbox) | `d6bd987` | 本番品質のリファレンス実装モノレポ(learn + reference 二層) | **最高** |

調査日: 2026-07-26

## Table of Contents

- [1. 現プロジェクトのギャップ分析](#1-現プロジェクトのギャップ分析)
- [2. リポジトリ別評価](#2-リポジトリ別評価)
- [3. 採用推奨機能・設計(優先度付き)](#3-採用推奨機能設計優先度付き)
- [4. 採用を見送るもの](#4-採用を見送るもの)
- [5. 実装ロードマップ案](#5-実装ロードマップ案)

---

## 1. 現プロジェクトのギャップ分析

本プロジェクトは既に多くの点で参照リポジトリより成熟している(セッションストア、
ベクトルストア 3 実装、CRAG ワークフロー、設定バリデーション、120 ファイルのテスト、
セキュリティミドルウェア群)。一方で、参照リポジトリとの比較で判明した主なギャップは以下。

### インフラ・運用

- **CI が存在しない**: `.github/` ディレクトリ、pre-commit、カバレッジゲートがない。
  README の「80%+ カバレッジ」は強制されていない。
- **実装済みストアが未配線**: `RedisSessionStore` / `ChromaVectorStore` /
  `OllamaEmbeddingVectorStore` は実装・テスト済みだが、[`app/main.py`](../app/main.py)
  が `InMemoryVectorStore()` / `InMemorySessionStore()` を直接インスタンス化しており、
  settings(`redis_session_store_enabled` 等)から実装を選択する factory がない。
  対応する設定値はアプリから一切参照されていないデッドコード状態。
- **レートリミットが実質無効**: グローバル 1000/min のみで、コストの高い LLM
  エンドポイントに個別制限がない。slowapi はプロセス内メモリバックエンドのまま
  (Redis は依存に入っているのに未使用)。
- **`/health/ready` が浅い**: `hasattr` チェックのみで、Redis・ベクトルストア・LLM
  プロバイダへの実疎通を検査しない。

### Agent 層

- ツールは開発環境限定の mock 1 個のみ。実ツール・構造化出力(`output_type` は全て
  `str`)・output validator の実利用がない。
- `UsageLimits`(トークン・リクエスト予算)が未設定。`POST /v1/agent/chat` には
  タイムアウトがなく、LLM がハングするとリクエストが無期限に滞留する
  (RAG 側にはタイムアウトあり)。
- 評価(evals)基盤が存在しない。エージェント品質の回帰を検出する仕組みがない。
- human-in-the-loop / ツール承認 / ツール許可リストといったガードレールがない。

### API / ストリーミング

- SSE レスポンスに `Cache-Control: no-cache` / `X-Accel-Buffering: no` がなく、
  nginx 等の背後ではストリームがバッファリングされる。ハートビートもないため、
  アイドルなプロキシ経由の長時間生成で切断されうる。
- SSE イベントが ad-hoc な `{"type", "content"}` dict で、型付きの契約がない。
- `session_id` がクライアント入力のまま所有権バインドなし(共有 API キーを持つ
  誰でも他セッションの履歴を読み書きできる IDOR)。

### 観測性

- ログが非構造化テキストで、`request_id` の付与が手動 2 箇所のみ。
- Logfire にプロンプト/ツール入出力のスクラビング設定がない。

---

## 2. リポジトリ別評価

### 2.1 agentic-ai-fastapi-playground — 採用価値: 低(設定のみ)

1 コミットのスキャフォールドで、`src/app/__init__.py` の docstring 1 行以外に
アプリコードが存在しない。設計・実装パターンの参照元にはならない。

採用できるのはツールチェーン設定のみ:

| 項目 | 参照 |
|---|---|
| `pip-audit` を CI チェーンに組み込む(`audit` タスク + `ci` タスクの `depends` 連鎖) | `mise.toml` |
| コーディングエージェント作業ディレクトリの `.gitignore` 追加(`.claude/`, `.serena/` 等) | `.gitignore` |
| (参考)pydantic-ai 2.0 beta + `pydantic-graph` + `pydantic-evals` という方向性の表明 | `pyproject.toml`, `uv.lock` |

pydantic-ai 2.0 beta / Python 3.14 への追随は「機能の移植」ではなく移行判断であり、
稼働中の CRAG ワークフローを持つ本プロジェクトでは現時点で見送りを推奨(§4)。

### 2.2 agentic-ai-bootcamp — 採用価値: 中〜高

Pydantic AI の段階的学習リポジトリ。API 層は最小限(認証・セッション・レートリミット
なし)で本プロジェクトが先行しているが、**エージェントパターンとテスト手法**に
移植価値の高いコードがある。

主要な発見(パスはリポジトリルートからの相対):

| 項目 | 内容 | 参照 |
|---|---|---|
| 2 軸 LLM ジャッジ評価グレーダー | Outcome 軸(correctness / completeness)+ Behavior 軸(tool_use_discipline / faithfulness)。1–5 + Unknown レーティング、Unknown を平均から除外する部分点集計、自己評価バイアス回避のための独立ジャッジ注入(`judge_model=`) | `lessons/11-evals/evals.py` |
| CRAG 強化 3 点セット | (1) リトライ時に取得件数 `k` を拡大(2→4)、(2) 引用検証 = `VALID_IDS ∩ hit_ids`(実在するだけでなく**今回実際に取得した**文書のみ引用可)、(3) 試行上限到達時はエラーではなく「接地済みサブセットの引用で劣化返却」 | `frameworks/llamaindex-rag-workflows/lessons/04-rag/rag_workflow.py` |
| workflow → SSE 橋渡し | `ctx.write_event_to_stream(ProgressEvent(...))` を全ステージで発行し、公開 API は `on_progress` コールバックを受け取る。llama-index-workflows のイベントを SSE 名前付きイベントに変換する自然な継ぎ目 | 同上 + `frameworks/llamaindex-rag-workflows/lessons/03-streaming-events/` |
| 空振り時の早期終了 | リトリーバル 0 件なら LLM を呼ばず即 `StopEvent` | `rag_workflow.py`(retrieve ステップ) |
| `create_app(model=...)` ファクトリ | import 時にモデルを構築せず、`uvicorn --factory` で起動。テストは `create_app(model=TestModel(...))` で注入 | `lessons/10-production/app.py` |
| output validator + `ModelRetry` | 未知の引用 ID を検出したら `ModelRetry` を投げてモデルに自己修正させる | `lessons/08-rag/rag.py` |
| ガードレール 3 層とその終了性テスト | `max_iters` / `UsageLimits` / `asyncio.Semaphore(max_parallel)`。「上限で必ず停止する」「並列制限してもカバレッジが落ちない」ことをテストで証明 | `lessons/07-advanced-agents/`, `lessons/09-multi-agent/` |
| SSE ワイヤ形式のテスト | SSE クライアントなしで `TestClient` のレスポンス本文に対し `"event: token" in body` を直接アサート | `lessons/10-production/test_app.py` |

注意点: README が参照する `Makefile` が実在しない等のドキュメントドリフトがあり、
その点は真似しない。

### 2.3 agentic-ai-sandbox — 採用価値: 最高

learn(教材)+ reference(pyright strict・カバレッジ 98% ゲートの本番品質実装)の
二層モノレポ。本プロジェクトに不足している領域(SSE 契約、ガードレール、引用検証、
CI、テスト衛生)のほぼ全てに参照実装がある。

主要な発見:

#### SSE / ストリーミング

| 項目 | 内容 | 参照 |
|---|---|---|
| 型付き SSE イベント契約 | `step_started` / `tool_called` / `token` / `completed` / `error` の 5 種 discriminated union。`event:` 名 = `type` 判別子、`data:` = `model_dump_json()`。最小フィールドのみで生プロンプト・トレースバック・資格情報を載せない | `reference/patterns/contracts/src/patterns_contracts/sse.py` |
| 双方向ワイヤ変換 + U+2028 バグ対策 | `to_sse()` / `parse_sse_events()`。パーサは SSE 仕様の行終端(`\r\n|\r|\n`)のみで分割する。`str.splitlines()` は U+2028/U+2029 でも分割するため、`model_dump_json()` が文字列内に生で出力しうるこれらの文字でペイロードが破断する — 自前実装なら踏むバグ | `reference/patterns/sse/src/patterns_sse/events.py` |
| ストリームライフサイクル堅牢化 | 例外時は必ず terminal `error` イベントを送出(無言の切断禁止)/ `_MAX_EVENTS = 1000` の暴走バックストップ / `send_timeout=60s` / `request.is_disconnected()` による協調切断 / `asyncio.CancelledError` は必ず再送出 / `finally: await agen.aclose()` | `reference/patterns/sse/src/patterns_sse/app.py` |
| ASGI レベル切断注入テスト | `httpx.ASGITransport` は全レスポンスをバッファし切断を再現できないため、`app(scope, receive, send)` を直接駆動し N フレーム後に `http.disconnect` を返すカスタムドライバでクリーンアップを検証 | `reference/patterns/sse/tests/support/asgi_driver.py` |

#### RAG / 引用

| 項目 | 内容 | 参照 |
|---|---|---|
| 引用健全性の loud-fail | `validate_citations()` が `EmptyCitationError` / `DanglingCitationError`(全ての未知 ID と既知集合をメッセージに列挙)を送出。もっともらしい未接地回答を型付き例外に変える | `reference/patterns/rag/src/patterns_rag/citation.py` |
| 決定的リトリーバル | `(-score, chunk_id)` で再ソートしてから truncate(同点スコアで引用が実行ごとに揺れない)。同期リトリーバは `asyncio.to_thread` でイベントループ外へ | `reference/patterns/rag/src/patterns_rag/retrieval.py` |
| chunk_id / locator スキーム | `chunk_id = f"{source}::{ordinal:04d}"`、locator は `page → section → char` の優先順位で導出しアンカー不在なら loud-fail。プロンプトには `chunk_id | source | locator | score` をラベル付けして「フィールドを逐語コピーで引用」させる | `reference/patterns/rag/src/patterns_rag/chunking.py`, `rag.py` |

#### Agent ガードレール

| 項目 | 内容 | 参照 |
|---|---|---|
| 4 ガードレール | (1) `max_iterations`、(2) `allowed_tools` 許可リスト(拒否は記録して即停止)、(3) `approval_hook(tool, args) -> bool` による危険ツールの human-in-the-loop、(4) トークン `budget` — 予算チェックは**ツール実行(副作用)前**に行う | `reference/patterns/frameworks/pydantic-ai/src/patterns_pydantic_ai/autonomous_agent.py` |
| 閉じた stop_reason 語彙 | `Literal["completed", "max_iterations", "budget_exceeded", "denied", "disallowed_tool"]` — 終了理由を型で列挙し、監査証跡として拒否・否認の試行を全て記録 | `reference/patterns/contracts/src/patterns_contracts/autonomous_agent.py` |
| 条件付き `NativeOutput` | `model.profile["supports_json_schema_output"]` が真のときのみ `NativeOutput(Schema)` でラップ(本番は grammar-constrained decoding、`TestModel`/`FunctionModel` はプレーンのまま)。ローカルモデルの構造化出力フレーク対策 | `reference/app/src/pydantic_ai_sandbox/agents/chat_agent.py` |
| `FallbackModel` + 起動時 dry-run | lifespan でフォールバックチェーンを eager 構築し、設定不備を初回リクエスト 500 ではなく起動失敗にする | `reference/app/src/pydantic_ai_sandbox/llm/factory.py`, `main.py` |

#### 観測性・テスト衛生

| 項目 | 内容 | 参照 |
|---|---|---|
| Logfire スクラビング | スクラビング既定 ON + `extra_patterns=["prompt", "tool_input", "tool_output"]`。`LOG_SENSITIVE_PAYLOADS=true` での無効化時は監査警告を追加送出。観測初期化全体を scoped `except Exception` で包み、観測障害で起動を落とさない(fail-soft) | `reference/app/src/pydantic_ai_sandbox/logging_setup.py` |
| hermetic テストガード | autouse `block_network` fixture が `socket.connect` 等を monkeypatch し、AF_INET/AF_INET6 のみ loud-fail(AF_UNIX は asyncio self-pipe のため素通し)。さらに「ガードが空振りしていないこと」自体のテストを同梱 | `reference/patterns/deep-research/tests/unit/conftest.py`, `tests/support/hermetic.py` |
| anti-false-green ガード | `EXPECT_LIVE_TESTS=<n>` に対し実際に実行された（`when == "call"`）テスト数を検査し、ゲートされたレーンが無言でスキップされ続ける事故(実際に起きた)を防ぐ pytest プラグイン | `reference/patterns/contracts/src/patterns_contracts/pytest_live_guard.py` |
| README↔コード契約ドリフトテスト | README の normative なコードフェンスを `ast` でパースし、クラス集合・フィールド集合・`Literal` 語彙をランタイム実体と突合。1 テストで 10 README を守る | `reference/patterns/contracts/tests/unit/test_contract_drift.py` |
| ハードコードモデル ID 禁止 | pre-commit の pygrep フック + ランタイムテストの二重ガード | `reference/.pre-commit-config.yaml`, `reference/app/tests/unit/test_no_hardcoded_model_ids.py` |

#### CI 構成(8 workflows)

- path フィルタで対象面ごとに分離、action は SHA ピン、`concurrency` で PR ランのみキャンセル
- 従量課金プロバイダ(watsonx)の統合テストは **`workflow_dispatch` 限定**(コスト管理)+
  「secrets が空文字で注入されて fail-open する」GitHub の仕様に対する明示的検証ステップ
- ライブモデルテストは PR トリガなしの nightly マトリクス(1 レーンの失敗が他を隠さない
  `fail-fast: false`)
- `security.yml`: pip-audit + gitleaks を push / PR / 週次 cron で実行
- Ollama モデル blob を `hashFiles`(workflow 自身)キーで `actions/cache` — モデル更新で
  キャッシュ自動ローテーション
- 参照: `.github/workflows/`, `.github/dependabot.yml`, `.gitleaks.toml`

#### 参考ドキュメント

- OWASP Agentic AI / LLM Top 10 と CVE フロアの対応表: `reference/patterns/SECURITY-NOTES.md`
- 評価グレーダー設計論: `reference/patterns/EVAL-GRADERS.md`

なお sandbox には永続化・セッション・会話履歴が一切なく、この領域は本プロジェクトが先行している。

---

## 3. 採用推奨機能・設計(優先度付き)

### P1 — 高優先(直接的な効果・リスク低減)

| # | 項目 | 主な参照元 | 本プロジェクトでの適用 |
|---|---|---|---|
| 1 | **CI パイプライン導入** | sandbox `.github/workflows/`, playground `mise.toml` | GitHub Actions で `mise run lint`(ruff + ty)→ `pytest`(unit/integration/e2e)→ `pip-audit` を実行。既存の mise タスクをそのまま呼べるため導入コストが低い。カバレッジ閾値(`fail_under`)も同時に設定し README の「80%+」を強制化 |
| 2 | **型付き SSE イベント契約 + ライフサイクル堅牢化** | sandbox `contracts/sse.py`, `patterns_sse/{events,app}.py` | 現行の `{"type","content"}` dict を discriminated union に置換し、`to_sse()` で送出。terminal `error` イベント保証・`_MAX_EVENTS`・`is_disconnected()`・`finally: aclose()` を [`app/api/v1/agent.py`](../app/api/v1/agent.py) のストリームに追加。同時に `Cache-Control: no-cache` / `X-Accel-Buffering: no` ヘッダとハートビートコメントを付与([`docs/production_deployment.md`](production_deployment.md) の nginx 節とも整合) |
| 3 | **CRAG 強化 3 点セット + 引用の型付き例外** | bootcamp `rag_workflow.py`, sandbox `citation.py` | [`app/workflows/corrective_rag.py`](../app/workflows/corrective_rag.py) に (1) リトライ時の widen-k、(2) `hit_ids` 交差による引用検証、(3) 上限到達時の接地済みサブセット劣化返却、(4) 0 件時の LLM スキップ早期終了を追加。`DanglingCitationError` 等は既存の `RAGWorkflowError` 階層([`app/workflows/exceptions.py`](../app/workflows/exceptions.py))に追加 |
| 4 | **Agent ガードレール** | sandbox `autonomous_agent.py`, bootcamp lessons 07/09 | `agent.run()` / `run_stream()` に `UsageLimits` を設定し、`POST /v1/agent/chat` に `asyncio.wait_for`(RAG 側と同じパターン)を追加。将来の実ツール導入に備え、閉じた `stop_reason` 語彙・`allowed_tools`・`approval_hook` の設計を採用 |
| 5 | **ストア factory 配線** | sandbox `llm/factory.py` の構成パターン | settings(`redis_session_store_enabled` 等)→ ストア実装を選択する factory を作り、[`app/main.py`](../app/main.py) の in-memory 固定を解消。実装・テスト済みの `RedisSessionStore` / `ChromaVectorStore` が初めて本番導線に乗る。起動時 dry-run(疎通確認)も同時に導入 |

### P2 — 中優先(品質・信頼性の強化)

| # | 項目 | 主な参照元 | 本プロジェクトでの適用 |
|---|---|---|---|
| 6 | **評価(evals)基盤** | bootcamp `lessons/11-evals/`, sandbox `eval_graders.py` + `EVAL-GRADERS.md` | 2 軸グレーダー契約(Outcome/Behavior、1–5 + unknown、根拠必須、`Judge[T]` Protocol 注入)を導入し、ゴールデンデータセットに対するオフライン評価を CI のゲート(または nightly)に。bootcamp 版 → sandbox 版への強化パスが README に文書化済みでそのまま辿れる |
| 7 | **Logfire スクラビング + fail-soft 初期化** | sandbox `logging_setup.py` | [`app/observability.py`](../app/observability.py) に `extra_patterns` スクラビングとオプトアウト監査警告、観測初期化の例外吸収を追加 |
| 8 | **`create_app()` ファクトリ化** | bootcamp `lessons/10-production/app.py` | [`app/main.py`](../app/main.py) の module-level `app` + import 時 `get_settings()` を解消し、`conftest.py` が import 前に env を設定する回避策を不要にする |
| 9 | **`block_network` hermetic fixture** | sandbox `deep-research/tests/unit/conftest.py` | unit テストが LLM / ベクトルストアへ無言で到達しないことを構造的に保証。「ガードが空振りしていない」テストも同梱 |
| 10 | **`FallbackModel` + 条件付き `NativeOutput`** | sandbox `llm/factory.py`, `agents/chat_agent.py` | プロバイダ障害時のフォールバックチェーン(起動時 dry-run 付き)と、`supports_json_schema_output` プロファイルゲートによる構造化出力の安全な導入(`output_type=str` からの脱却) |
| 11 | **セッション所有権 + レートリミット強化** | (ギャップ分析より。参照実装はなし — 両参照リポジトリとも認証レイヤは対象外) | `session_id` をサーバ発行 + 認証主体へのバインドに変更(IDOR 解消)。slowapi にエンドポイント別制限(LLM 系を厳しく)と Redis `storage_uri` を設定 |

### P3 — 低優先(運用・衛生)

| # | 項目 | 主な参照元 |
|---|---|---|
| 12 | pip-audit / gitleaks / pre-commit / dependabot、`.gitignore` へのエージェント作業ディレクトリ追加 | playground `mise.toml`, sandbox `security.yml`, `.pre-commit-config.yaml` |
| 13 | `/health/ready` の実依存チェック化(Redis ping / ベクトルストア疎通)、構造化 JSON ログ + logging Filter による `request_id` 全域相関 | sandbox の boot チェーン思想(`main.py`) |
| 14 | README↔コード契約ドリフトテスト、ハードコードモデル ID 禁止の二重ガード、anti-false-green ライブテストガード | sandbox `test_contract_drift.py`, `pytest_live_guard.py` |
| 15 | ツール設計規約の採用(実ツール導入時): `<resource>_<verb>` 命名、ページネーション + `next_offset`、`response_format: concise/detailed`、寛容な引数パース | sandbox `tool_design.py` |
| 16 | OWASP Agentic AI / LLM Top 10 対応表の自プロジェクト版作成 | sandbox `SECURITY-NOTES.md` |

---

## 4. 採用を見送るもの

| 項目 | 理由 |
|---|---|
| pydantic-ai 2.0 beta / Python 3.14 への移行(playground の方向性) | 機能移植ではなく移行判断。`prerelease = "allow"` を稼働中の CRAG ワークフローに持ち込むリスクが利益を上回る。2.0 GA 後に別途検討 |
| モノレポ / mise 多層構成・uv workspace(sandbox, bootcamp) | 単一サービスには過剰。現行の単一 `pyproject.toml` + mise タスクで十分 |
| SDD スペック体系(sandbox `reference/specs/`) | spec/plan/tasks/PDCA のフル体系は重い。ADR 的な軽量ドキュメント(設計判断の記録)のみ推奨 |
| MCP 統合 | 3 リポジトリのいずれにも実装が存在せず、参照できるものがない(playground の依存宣言のみ)。必要になった時点で公式ドキュメントから直接導入 |
| pyright strict + ty の二重型チェック(sandbox / playground) | 本プロジェクトは ty strict 導入済み。二重運用は保守コストが利益を上回る |
| デュアル LLM ジャッジ以前の重厚な評価インフラ一式 | まず P2-#6 の最小構成(グレーダー契約 + ゴールデンセット)から始める |

---

## 5. 実装ロードマップ案

```
Phase 1(P1: 基盤とリスク低減)
  1. CI パイプライン(#1)          — 他の全変更の安全網。最初に入れる
  2. ストア factory 配線(#5)      — デッドコード解消。CI があれば安全に着手可能
  3. SSE 契約 + 堅牢化(#2)
  4. CRAG 強化 + 引用検証(#3)
  5. ガードレール(#4)

Phase 2(P2: 品質強化)
  6. create_app() ファクトリ化(#8)— テスト構造が簡潔になり以降の変更が楽になる
  7. block_network fixture(#9)
  8. Logfire スクラビング(#7)
  9. FallbackModel + NativeOutput(#10)
 10. 評価基盤(#6)
 11. セッション所有権 + レートリミット(#11)

Phase 3(P3: 運用衛生)
 12. pre-commit / gitleaks / dependabot(#12)
 13. readiness 実チェック + 構造化ログ(#13)
 14. ドリフトテスト等のガード群(#14)
```

各フェーズは独立して価値を出せる順に並べている。特に **CI(#1)を最初に導入する**ことで、
以降の全ての取り込み作業が既存 120 テストファイルの回帰検出下で行える。
