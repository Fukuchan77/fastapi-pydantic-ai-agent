# コードレビュー & pydantic-ai-sandbox 比較・検証レポート

`fastapi-pydantic-ai-agent`(調査時コミット `fd6ec5a`)をコードレビューし、リファレンス実装集
[pydantic-ai-sandbox](https://github.com/Fukuchan77/pydantic-ai-sandbox)(調査時コミット `eea6f07`)
が確立しているパターンと比較・検証した結果をまとめる。

- 調査日: 2026-08-02
- 関連文書: [reference-repo-review.md](reference-repo-review.md)(別リポジトリ群の先行レビュー。
  本レポートと重複する指摘は「先行レビュー済み」と付記)

## Table of Contents

- [1. サマリ](#1-サマリ)
- [2. 前提: 両リポジトリの役割とバージョン差](#2-前提-両リポジトリの役割とバージョン差)
- [3. レビュー指摘一覧](#3-レビュー指摘一覧)
- [4. pydantic-ai-sandbox との観点別比較](#4-pydantic-ai-sandbox-との観点別比較)
- [5. 検証結果](#5-検証結果)
- [6. 推奨ロードマップ](#6-推奨ロードマップ)
- [7. 追補: 001-agent-architecture-enhancements ブランチとの照合](#7-追補-001-agent-architecture-enhancements-ブランチとの照合)

---

## 1. サマリ

**総評**: 本プロジェクトはミドルウェア・ストア・設定バリデーション・テスト量(823 テスト)の面で
よく作り込まれており、ユニット/E2E/統合テストと lint / 型チェックは全てグリーン
([§5](#5-検証結果))。一方で、**再現確認済みの実バグが 1 件**(RAG キャッシュの恒久汚染)、
本番ガードをすり抜ける設定不備が 1 件あり、pydantic-ai の安全装置
(`UsageLimits` / `ModelSettings` / タイムアウト / 構造化出力)がチャット経路にほぼ皆無という
ギャップがある。pydantic-ai-sandbox は pydantic-ai **v2** 前提のリファレンスであり
(本プロジェクトは v1.70)、API 差分を踏まえた上で採用すべき設計規律
(予算管理、SSE の切断処理、fail-fast 起動、テスト規律)が多数ある。

**最重要指摘 TOP 5**(詳細は [§3](#3-レビュー指摘一覧)):

| # | 深刻度 | 指摘 | 場所 |
|---|---|---|---|
| H-1 | High | タイムアウトで中断された RAG リクエストが in-flight キャッシュを恒久汚染し、以後同一クエリが常に 504(**再現スクリプトで実証済み**) | [corrective_rag.py:294](../app/workflows/corrective_rag.py) |
| H-2 | High | `app_env` が自由文字列のため `Production`/`prod` 等の表記ゆれで本番ガードが無効化され、mock ツールが本番登録されうる | [config.py:308](../app/config.py) |
| H-3 | High | `/v1/rag/ingest` が RAG 結果キャッシュを無効化せず、新規文書が最大 TTL 300 秒間クエリ結果に反映されない | [corrective_rag.py:134](../app/workflows/corrective_rag.py) |
| H-4 | High | チャット経路に `UsageLimits`・`ModelSettings`・タイムアウトが一切なく、トークン消費と滞留時間が無制限 | [chat_agent.py:119](../app/agents/chat_agent.py) |
| H-5 | High | セッション履歴が全量リプレイ+上限 1000 件到達で `ValueError` となり、そのセッションが恒久的に使用不能になる | [session_store.py:281](../app/stores/session_store.py) |

---

## 2. 前提: 両リポジトリの役割とバージョン差

| | fastapi-pydantic-ai-agent | pydantic-ai-sandbox |
|---|---|---|
| 性格 | 本番指向の FastAPI + PydanticAI アプリ(chat / SSE / CRAG) | リファレンス実装モノレポ(ルートアプリ + 10 パターンレーン) |
| pydantic-ai | **pydantic-ai-slim 1.70.0(v1 系)** + pydantic-ai-litellm 0.2.3 | **pydantic-ai-slim 2.3.0(v2 系)**、hitl レーンのみ 2.9.x |
| Python | >=3.13 | ルート >=3.13、hitl / sse レーン >=3.14 |
| モデル解決 | `LiteLLMModel` 経由の一元ルーティング | provider 別 factory + `FallbackModel`(watsonx はカスタム `Model` 実装) |
| 品質ゲート | ruff + ty、カバレッジゲートなし | ruff(bandit/複雑度 10 含む)+ pyright strict + カバレッジ 98% + CI/pre-commit で機械強制 |

**比較上の注意**: sandbox のパターンには v2 でのみ成立するもの(`instructions=`、
`NativeOutput`、deferred tools、`run_stream_events` 等)が含まれる。本レポートでは
「**v1.70 のまま今すぐ採用できる規律**」と「**v2 移行時に効いてくる差分**」を区別して記載する。
v2 移行時の破壊的変更チェックリストは sandbox の
`specs/document-review/agentic-ai-design-v2-review.md` が実機検証ログ付きでまとめており、
そのまま移行計画の下敷きにできる。

---

## 3. レビュー指摘一覧

### High

#### H-1. キャンセルされた RAG リクエストが in-flight キャッシュを恒久汚染する【実証済み】

- 場所: [app/workflows/corrective_rag.py:251-300](../app/workflows/corrective_rag.py)
- thundering herd 対策として `_pending_futures[cache_key]` に Future を登録するが、
  失敗時のクリーンアップが `except Exception`(L294)であり、`BaseException` 派生の
  `asyncio.CancelledError` を捕捉しない。RAG エンドポイントは
  [app/api/v1/rag.py:76](../app/api/v1/rag.py) で `asyncio.timeout()` により run を
  キャンセルするため、タイムアウト発生時に Future が未解決のまま残留する。
- ワークフローインスタンスは [app/deps/workflow.py](../app/deps/workflow.py) で
  ベクトルストアをキーにプロセス全体でキャッシュされるため、以後同一
  `(query, max_retries)` のリクエストは全て死んだ Future を await し続け、
  自身のタイムアウトまでハングして **504 を返し続ける**。
- 再現: [§5.2](#52-h-1-の再現実証) の通り、1 回目のタイムアウト後はバックエンドが
  即応答できる状態でも 2 回目の同一クエリがタイムアウトすることを確認した。
- 修正方針: `except Exception` → `try/finally`(または `except BaseException` で
  Future 解決+再 raise)に変更し、キャンセル経路でも必ず
  `future.cancel()`/`set_exception()` と `del _pending_futures[key]` を実行する。
  sandbox の SSE レーンが規範とする「`CancelledError` は握りつぶさず再送出、
  クリーンアップは `finally`」([§4.4](#44-sse-ストリーミング))と同じ規律。

#### H-2. `app_env` が未検証の自由文字列で、本番ガードが表記ゆれで無効化される

- 場所: [app/config.py:308-311](../app/config.py)
- `app_env: str = Field(default="development", ...)` に値検証がない。一方、本番ガードは
  `== "production"` の完全一致比較([config.py:552](../app/config.py) の mock ツール禁止
  validator、[chat_agent.py:133](../app/agents/chat_agent.py) の mock ツール登録ガード)。
- `APP_ENV=Production` や `APP_ENV=prod` と設定された本番環境では両ガードが素通りし、
  **mock ツールが本番エージェントに登録されうる**。`extra="forbid"` や SecretStr 検証など
  他の設定が厳格なだけに、ここだけ抜けているのは危険。
- 修正方針: `Literal["development", "staging", "production"]` にする。sandbox は
  provider 名を `Literal` で閉じ、dispatch テーブルとの整合をテストで固定している
  (`test_factory_dispatch.py`)。sandbox のレビューチェックリストにも
  「閉じた語彙は `str`+説明ではなく `Literal`」の規則がある。

#### H-3. `/v1/rag/ingest` が RAG 結果キャッシュを無効化しない

- 場所: キャッシュキー生成 [corrective_rag.py:134-150](../app/workflows/corrective_rag.py)、
  ingest [app/api/v1/rag.py:46](../app/api/v1/rag.py)
- キャッシュキーは `sha256(query|max_retries)` のみで、ベクトルストアの内容バージョンを
  含まない。ingest は同じベクトルストアを変更するがキャッシュには触れないため、
  **新規文書投入後も最大 `rag_cache_ttl`(既定 300 秒)の間、同一クエリは古い回答を返す**。
- 修正方針: ingest 時に該当ワークフローの `_cache` / `_pending_futures` をクリアする
  (ストアに世代カウンタを持たせキーに混ぜる方法でも可)。

#### H-4. チャット経路に予算・時間の上限が一切ない

- 場所: [app/agents/chat_agent.py:119-124](../app/agents/chat_agent.py)、
  [app/api/v1/agent.py:154, 239](../app/api/v1/agent.py)
- `Agent` 構築時に `ModelSettings`(temperature / max_tokens)がなく、`run` / `run_stream`
  呼び出しに `usage_limits=` も `asyncio.timeout()` もない(RAG 経路には両方ある)。
  ツール呼び出しループの暴走や低速プロバイダで、トークン消費・接続滞留が無制限になる。
  先行レビューでも指摘済みだが未対応。
- 修正方針: sandbox hitl レーンの規律を採用 —
  `UsageLimits(request_limit=..., tool_calls_limit=..., total_tokens_limit=...)` を
  モジュール定数として定義し全 run に渡す(v1.70 でも `usage_limits=` は利用可能)。
  併せて `asyncio.timeout(settings.llm_request_timeout)` でルート層を包む。

#### H-5. セッション履歴の無制限成長と上限到達時の恒久破損

- 場所: [app/api/v1/agent.py:148-165](../app/api/v1/agent.py)、
  [app/stores/session_store.py:104, 281-282](../app/stores/session_store.py)
- 毎ターン全履歴を `message_history=` でリプレイし、`result.all_messages()` を全量保存する。
  唯一の上限は 1000 件で、トークン予算・要約・トリミングはない。コストはターン数に
  比例して増加し、**1001 件目の保存で `ValueError` が発生した後はそのセッションが
  読み書きとも恒久的に失敗する**(クライアントには 500 が返り続ける)。
- 修正方針: 上限到達時は例外ではなく古いメッセージの切り捨て(ツール往復の整合を保つ
  境界で切る)へ変更。中期的には sandbox が `docs/context-engineering.md` で示す
  コンパクション/構造化ノートの導入を検討。

### Medium

#### M-1. lifespan 後半に try/finally がなくリソースリークしうる

- [app/main.py:222-304](../app/main.py)。`build_chat_agent()`(L269)や
  `configure_logfire()`(L273)が raise すると、teardown(L287-304)が実行されず
  `cleanup_task` と `httpx.AsyncClient` がリークする。設定検証ブロック(L181-220)のみ
  try で保護されている。sandbox は「起動時 fail-fast」をパターン化しているが
  (`_lifespan` 内の eager dry-run)、失敗経路でも後始末が走る構造にしている。

#### M-2. Redis / Chroma / embedding 設定がデッドコード(先行レビュー済み・未対応)

- [config.py:262-270, 392-401](../app/config.py) の `redis_url` /
  `redis_session_store_enabled` / `embedding_model` / `embedding_base_url` は
  どこからも参照されず、[main.py:223, 227](../app/main.py) は
  `InMemoryVectorStore` / `InMemorySessionStore` をハードコードする。
  実装・テスト済みの `RedisSessionStore` / `ChromaVectorStore` /
  `OllamaEmbeddingVectorStore` に到達経路がない。settings から実装を選ぶ factory を
  設ける(sandbox の `llm/factory.py` の dispatch テーブル+ Literal 整合テストが手本)。

#### M-3. `InMemoryVectorStore.query` がイベントループ上で CPU バウンド処理を行い、ロックもない

- [app/stores/vector_store.py:122-231](../app/stores/vector_store.py)。クエリごとに
  最大 1000 文書分の TF-IDF ベクトルを同期再計算し(L222-225)、`add_documents` は
  `_documents` / `_doc_tokens` / `_memory_usage` を無ロックで変更する。並行リクエストで
  イベントループ停止と不整合読み取りが起こりうる。同ファイルの Chroma 実装は
  `run_in_executor` を正しく使っており、それに合わせる。

#### M-4. リトライの transient 判定がメッセージ文字列の部分一致

- [app/workflows/exceptions.py:56-70](../app/workflows/exceptions.py)。
  `"connection"` / `"503"` 等を含むメッセージは認証エラーでもリトライ対象になる。
  sandbox は例外**型**で分類する規律(`FallbackModel` は `ModelAPIError` のみ回復、
  カスタムトランスポートは SDK 例外を `ModelAPIError` にラップ、プロバイダ側リトライは
  `max_retries=0` で無効化して責務を一元化)。型ベース分類へ移行すべき。

#### M-5. エラーレスポンスのエンベロープが 2 種類ある

- 401 は `{"detail": {"message", "code"}}`([app/deps/auth.py:70-73](../app/deps/auth.py))、
  413/429/500 はフラットな `ErrorResponse`。クライアントに 2 系統のパーサを強いる。
  どちらかに統一する。

#### M-6. `secrets.compare_digest` に非 ASCII ヘッダで 500

- [app/deps/auth.py:54](../app/deps/auth.py)。Starlette はヘッダを latin-1 でデコード
  するため、非 ASCII の `X-API-Key` は `compare_digest` が `TypeError` を投げ、
  401 ではなく 500 になる。encode してから比較する。

#### M-7. レートリミットがプロセスローカルかつプロキシ背後で全クライアント共有

- [app/middleware/rate_limit.py:94-98](../app/middleware/rate_limit.py) の `Limiter` に
  `storage_uri` がなくワーカー単位の制限になる。また既定 `trusted_proxies=[]`
  ([config.py:340](../app/config.py))では LB 背後で全クライアントがプロキシ IP の
  単一バケット(1000/min)を共有する。Redis バックエンド化と、デプロイ手順書での
  `trusted_proxies` 設定必須化を。

#### M-8. SSE 経路が `BaseHTTPMiddleware` 2 枚に包まれ、キャッシュ制御ヘッダもない

- [app/main.py:343-355](../app/main.py)、[app/api/v1/agent.py:306](../app/api/v1/agent.py)。
  `BaseHTTPMiddleware` は `StreamingResponse` を再ラップし、長時間 SSE の切断伝播と
  バックプレッシャに干渉することが知られる。`Cache-Control: no-cache` /
  `X-Accel-Buffering: no` も未設定(先行レビュー済み)。sandbox の SSE レーンとの
  詳細比較は [§4.4](#44-sse-ストリーミング)。

#### M-9. `_get_cached_model` が自身の引数を無視する

- [app/deps/workflow.py:29-56](../app/deps/workflow.py)。`llm_model` / `llm_base_url`
  引数は `lru_cache` のキーとしてのみ機能し、本体は `get_settings()` を呼び直す。
  また呼び出し側(L90)は `str | None` 注釈に `HttpUrl` を渡している。引数を実際に
  使うか、キー引数をやめて設定ハッシュをキーにする。

### Low(コードスメル)

| # | 指摘 | 場所 |
|---|---|---|
| L-1 | `result.data` フォールバックはデッドコード(v1 で `data` は削除済み、かつ `output_type=str`) | [agent.py:181-186](../app/api/v1/agent.py) |
| L-2 | 未使用の `await limiter.hit(...)` — slowapi の `hit` は同期関数 | [rate_limit.py:184](../app/middleware/rate_limit.py) |
| L-3 | `save_history` が `_last_access` をロック外で更新(`get_history` は修正済みの同じレース) | [session_store.py:182](../app/stores/session_store.py) |
| L-4 | `RedisSessionStore.close()` が deprecated な `close()` を使用し、かつ lifespan から呼ばれない | [session_store.py:500](../app/stores/session_store.py) |
| L-5 | `readiness_check` が sync `def` で `hasattr` チェックのみ(先行レビュー済み) | [health.py:24](../app/api/health.py) |
| L-6 | ワークフローのモデル解決フォールバックが `build_model()` を迂回し、素の設定文字列を `Agent` に渡す(現状は呼び出し側が常にモデルを渡すため潜在) | [corrective_rag.py:110](../app/workflows/corrective_rag.py) |
| L-7 | CSP に末尾スペース・`'unsafe-inline'`、HSTS を平文 HTTP でも送出 | [security_headers.py:50-55](../app/middleware/security_headers.py) |
| L-8 | `Vary` ヘッダを追記でなく上書き | [cors.py:91](../app/middleware/cors.py) |
| L-9 | 関数内で `StreamingResponse` 等を重複 import | [agent.py:217-219](../app/api/v1/agent.py) |
| L-10 | `patch("app.api.v1.agent.get_agent_deps")` は束縛済み `Depends` に無効(app.state パッチで偶然成立) | tests/unit/api/v1/test_agent_endpoints.py:89 ほか |
| L-11 | README の `/health` レスポンス例(`{"status": "healthy", ...}`)が実装(`{"status": "ok"}`)と不一致 | README.md / [health.py:20](../app/api/health.py) |
| L-12 | `tests/unit/middleware/` のみ `__init__.py` 欠落、`test_naming_conventions.py` が CWD 依存 | tests/unit/ |

---

## 4. pydantic-ai-sandbox との観点別比較

### 4.1 エージェント定義

| 観点 | 本プロジェクト | sandbox の規範 |
|---|---|---|
| 型パラメータ | `Agent[AgentDeps, str]`(chat)/ 素の `Agent`(RAG 内部 2 体) | 全構築箇所で `Agent[DepsT, OutputT]` を明示。deps なしは `deps_type=type(None)`(pyright strict 対応) |
| システムプロンプト | `agent.system_prompt(...)` 動的登録 | `instructions=`。`system_prompt` は message_history に残留し再開境界でリークするため非推奨(v2 の知見だが、履歴をまたぐ設計への示唆として有効) |
| ツール登録 | `@agent.tool` デコレータ(mock 1 個、条件付き登録) | モジュールレベル関数+コンストラクタ `tools=[...]`(デコレータのみだと pyright strict で未使用扱いになるため)。設定が要る場合は factory がクロージャを返す |
| ファクトリ | `build_chat_agent(model=None)` — **同型で一致** | `build_chat_agent(model: Model \| None = None)`。「新規構築は `model=`、既存の一時差し替えは `agent.override`」の使い分けを明文化 |

RAG 内部の `_eval_agent` / `_synth_agent`
([corrective_rag.py:111-118](../app/workflows/corrective_rag.py))が素の `Agent` である点、
および L-6 のフォールバック経路は sandbox 流に揃えると型と設定の一貫性が上がる。

### 4.2 出力型と検証

- 本プロジェクトは全エージェントが `output_type=str` で、`output_retries` を設定して
  いるものの検証対象が事実上ない。RAG の関連性評価
  ([corrective_rag.py:574-628](../app/workflows/corrective_rag.py))は LLM の文字列出力を
  自前パースしており、構造化出力にすれば自前リトライの多くが不要になる。
- sandbox は構造化 Pydantic モデルが既定(`ChatResponse` 等)で、
  `agent.output_validator` + `ModelRetry`(実行可能な指示文言つき)を検証の第一線に置く。
  v1.70 でも `output_type=PydanticModel` と `@agent.output_validator` は利用可能であり、
  今すぐ採用できる。
- sandbox 固有の注意点として、`NativeOutput` は `TestModel`/`FunctionModel` が
  `supports_json_schema_output=False` を返すため条件付きでのみラップする、という
  非自明パターンをテストで固定している(v2 移行時に踏む地雷の先回り)。

### 4.3 予算・安全装置

- 本プロジェクト: チャット経路に `UsageLimits` / `ModelSettings` / タイムアウトなし(H-4)。
- sandbox: `UsageLimits(request_limit=8, tool_calls_limit=10, total_tokens_limit=20_000)` を
  モジュール定数化して全 run に渡し、テストで上書き可能にする。予算超過は HTTP 429 に
  マップ。マルチターンでは `UsageLimits` が run ごとにリセットされるため
  `usage=stored.usage` で予算を持ち越す。タイムアウトは設定駆動
  (`WATSONX_TIMEOUT_CONNECT/READ`)で正値検証つき。
- 本プロジェクトの RAG 経路(`asyncio.timeout` + ワークフロータイムアウト設定)は
  この規律に近く、チャット経路にも同じ構造を展開すればよい。

### 4.4 SSE ストリーミング

| 観点 | 本プロジェクト([agent.py:195-306](../app/api/v1/agent.py)) | sandbox `patterns/sse/` |
|---|---|---|
| 配信 | 手書き `data: {...}\n\n` + `StreamingResponse` | `sse_starlette.EventSourceResponse`(`send_timeout` つき) |
| イベント型 | ad-hoc な `{"type", "content"}` dict | 判別共用体 `SseEvent`(契約パッケージで一元定義、README との drift をテストで検出) |
| 切断処理 | `CancelledError` を捕捉してログ | `request.is_disconnected()` で協調的 break、`CancelledError` は**再送出**、`finally` で `agen.aclose()` |
| 暴走対策 | なし | `_MAX_EVENTS=1000` バックストップ |
| プロキシ対策 | ヘッダなし(M-8) | EventSourceResponse が `Cache-Control` 等を設定 |

H-1 の根因(キャンセルを想定しないクリーンアップ)と M-8 は、この
「キャンセル安全 + 明示的上限 + 型付きイベント」の規律の不在という同じ穴から来ている。
`sse-starlette` の採用は先行レビューでも P2 提案済み。

### 4.5 メッセージ履歴

- 共通点: 履歴は `result.all_messages()` をサーバ側ストアで管理し、クライアントからは
  受け取らない。本プロジェクトのこの設計は sandbox のセキュリティ規範
  (「`message_history` / `usage` をクライアントから受けない」— リクエストモデルの
  `extra="forbid"` で機械的に排除)と一致しており、**良い点**。
- 相違点: sandbox は履歴持ち越しに予算持ち越し(`usage=`)とコンパクション指針
  (`docs/context-engineering.md`)を伴わせる。本プロジェクトは上限 1000 件の
  ハードエラーのみ(H-5)。

### 4.6 エラーハンドリング方針

- sandbox の規律: (1) ルート層はエージェント例外を握らず FastAPI の 500 に伝播、
  (2) 起動時 fail-fast(設定 validator + lifespan での eager dry-run)、
  (3) fail-soft は観測性ブートストラップのみに限定し、broad-except には必ず
  scoped `noqa` + 理由コメント(ruff `BLE` で機械強制)。
- 本プロジェクトは (2) の設定 validator は充実しているが、lifespan の失敗経路に
  後始末がなく(M-1)、リトライ分類が文字列一致(M-4)、エラーエンベロープが
  不統一(M-5)。グローバル 500 ハンドラで一律に丸める設計は sandbox と思想が
  異なるが、それ自体は選択の範囲。問題は分類と後始末の精度。

### 4.7 観測性

- 両者とも `logfire.instrument_pydantic_ai()` + `instrument_fastapi()` を使用。
- sandbox が上乗せしている規律: `send_to_logfire="if-token-present"` +
  トークン未設定時の 1 回警告、**スクラビング既定オン**
  (`ScrubbingOptions(extra_patterns=["prompt", "tool_input", "tool_output"])`、
  無効化には明示フラグ+警告)、configure/instrument の例外は 1 回の WARNING に畳み
  絶対に伝播させない(テストで固定)。本プロジェクトの
  [observability.py](../app/observability.py) にはスクラビング設定がなく、
  プロンプト・ツール入出力が平文でテレメトリに乗る。**採用推奨**。

### 4.8 テスト規律

- 共通の良い点: `TestModel` / `FunctionModel` によるハーメチックなエージェントテスト、
  dependency_overrides の活用、設定キャッシュのクリア fixture。
- sandbox のみにある規律で採用価値が高いもの:
  1. **`models.ALLOW_MODEL_REQUESTS = False`** を conftest で設定し、テストが実 LLM に
     アクセスした瞬間に失敗させる(本プロジェクトは未設定。現状は env ガードのみ)。
  2. **anti-false-green**: live テストは「到達不能ならスキップでなく FAIL」+
     `EXPECT_LIVE_TESTS=<n>` で実行数を強制。本プロジェクトの `-m ollama` は
     全スキップでもグリーンになる。
  3. **API サーフェスロックテスト**(`test_chat_agent_v2_surface.py`): pydantic-ai の
     公開 API 形状をアプリコードと独立に固定し、依存更新の破壊を早期検出。
     v1→v2 移行を控える本プロジェクトには特に有効。
  4. `pyproject.toml` に `asyncio_mode = "auto"` を設定(本プロジェクトは未設定で、
     約半数のファイルが明示 `@pytest.mark.asyncio` に依存。付け忘れたテストは
     コルーチンが実行されないままパス扱いになるリスクがある)。
  5. モデル ID ハードコード禁止の静的スキャンテスト + pre-commit フック。
- L-10(効果のない `patch`)のような「偶然通っているテスト」は、sandbox の
  「公開フックのみに触る」(`TestModel.last_model_request_parameters` 等)方針で防げる。

### 4.9 設定管理

- 共通: pydantic-settings + `SecretStr` + cross-field validator + キャッシュされた
  `get_settings()`。本プロジェクトの placeholder 検出・HTTPS 強制はむしろ sandbox より
  厳しく、**良い点**。
- 相違: sandbox は「設定に書けることは必ず配線されている」ことをテストで保証する
  (dispatch テーブルと `Literal` の drift テスト)。本プロジェクトは M-2 の通り
  未配線設定が残る。また `app_env` の `Literal` 化(H-2)も sandbox 流。

---

## 5. 検証結果

### 5.1 品質ゲート実行(2026-08-02、Python 3.13 / uv 0.8.17)

| コマンド | 結果 |
|---|---|
| `uv sync --frozen` | 成功 |
| `uv run pytest tests/unit/ tests/integration/ tests/e2e/ -q` | **789 passed, 1 skipped, 6 errors**(80.2 秒) |
| `uv run ruff check app/ tests/` | All checks passed |
| `uv run ty check app/` | All checks passed |

- 6 件のエラーは全て `tests/integration/test_docker_deployment.py` で、検証環境に
  Docker デーモンがないことによる環境起因(コード不良ではない)。
- `tests/local/`(Ollama 必須)と `tests/benchmarks/` は対象外とした。

### 5.2 H-1 の再現実証

`CorrectiveRAGWorkflow` に「1 回目はハングする LLM、2 回目は即応答する LLM」を注入し、
RAG エンドポイントと同じ `asyncio.timeout()` でラップして実行した結果:

```text
request 1: timed out (would be HTTP 504)
pending futures after cancel: 1  <- leaked if > 0
request 2: timed out despite fast backend -> cache poisoned, every identical query 504s forever
```

1 回目のタイムアウト後に `_pending_futures` へ未解決 Future が残留し、バックエンドが
即応答できる状態でも 2 回目の同一クエリが 504 相当になることを確認した。
`except Exception`(corrective_rag.py:294)が `CancelledError` を捕捉しないことが根因。

---

## 6. 推奨ロードマップ

### P1 — 実バグ修正(小さく、即効)

1. **H-1**: `_pending_futures` のクリーンアップを `try/finally` 化(キャンセル時は
   `future.cancel()` + 削除)。§5.2 の再現手順をそのまま回帰テスト化できる。
2. **H-2**: `app_env` を `Literal["development", "staging", "production"]` に変更。
3. **H-3**: ingest 時に RAG キャッシュをクリア(または世代カウンタをキーに混入)。
4. **H-5**: 履歴上限到達時の `ValueError` を古いメッセージの切り捨てに変更。
5. **M-1**: lifespan の agent 構築以降を `try/finally` で保護。
6. **M-6**: API キー比較を bytes 化して非 ASCII ヘッダの 500 を解消。

### P2 — sandbox パターンの採用(v1.70 のままで可能)

1. **H-4**: `UsageLimits` 定数 + `ModelSettings`(max_tokens / temperature)+
   チャット経路の `asyncio.timeout`。
2. `models.ALLOW_MODEL_REQUESTS = False` を conftest に追加、
   `asyncio_mode = "auto"` を pyproject に設定。
3. Logfire スクラビング(`ScrubbingOptions(extra_patterns=[...])`)を既定オンに。
4. SSE を `sse-starlette` + 型付きイベント + `is_disconnected()` + イベント上限に刷新
   (M-8 と先行レビュー P2 の統合対応)。
5. RAG の関連性評価を構造化出力(`output_type=EvalResult` + `output_validator`)に変更し、
   文字列パースと substring リトライ分類(M-4)を型ベースに置換。
6. ストア factory を導入して M-2 の未配線設定を解消(または設定ごと削除)。
7. live テストの anti-false-green 化、モデル ID ハードコード禁止スキャンの導入。

### P3 — v2 移行準備

1. sandbox の `specs/document-review/agentic-ai-design-v2-review.md` を移行チェック
   リストとして採用。特に本プロジェクトに直撃するもの:
   - `result.data` 完全廃止(L-1 のデッドコード削除で先行対応)
   - `system_prompt` → `instructions` への移行
   - `pydantic-ai-slim[extras]` 維持(メタパッケージ不使用) — 現状のままで OK
   - `FunctionModel` の終端応答は `ToolCallPart("final_result", ...)` にする
     (構造化出力導入後のテスト書き換え時に必要)
2. API サーフェスロックテストを v1.70 の現行 API で先に作成し、v2 更新時の破壊を
   このテストの差分として観測する。
3. `pydantic-ai-litellm`(v1 系依存)の v2 対応状況の確認が移行の前提条件になる点に注意。

---

## 7. 追補: 001-agent-architecture-enhancements ブランチとの照合

追補日: 2026-08-02。本章は `origin/001-agent-architecture-enhancements`
(HEAD `aaabdea`、本レポートのレビュー対象 `fd6ec5a` の上に 30 コミット)を
§3 の指摘・§4 の比較観点と照合し、検証した結果である。

### 7.1 総評

このブランチは本レポートの P1/P2 ロードマップと pydantic-ai-sandbox の規範の
**大部分を先行実装している**。特に H-1(RAG キャッシュの CancelledError 汚染)は
`except BaseException` + await 不在のクリーンアップ + フォロワーへの `TimeoutError`
伝播という模範的な修正になっており、専用回帰テスト
(`tests/unit/workflows/test_rag_cache_cancellation.py`)も付いている。
一方で **H-2 / H-3 / H-5 の 3 件の High と、M-3 / M-4 / M-5 / M-6(API キー側)は
未解消のまま残っている**。

### 7.2 ブランチが新たに実装したもの(sandbox 規範との対応)

| ブランチの実装 | 対応する sandbox 規範 / 本レポート指摘 |
|---|---|
| `run_guarded()`: `UsageLimits` + ツール許可リスト + 承認フック + 監査証跡(`app/agents/guardrails.py`、toolset ラッパー方式、閉じた `StopReason` 語彙) | §4.3(H-4 解消)+ hitl レーンの承認パターン |
| チャット経路の `chat_request_timeout` → 504(`app/api/v1/agent.py`) | §4.3(H-4 解消) |
| ストア factory + 起動時接続 dry-run(`app/stores/factory.py`、Redis/Chroma/Ollama が設定から選択可能に) | §4.9(M-2 解消)+ fail-fast at boot |
| `FallbackModel` チェーン + `NativeOutput` の条件付きゲート(`app/llm/factory.py`) | §4.1 / sandbox `llm/fallback.py` の規範 |
| 型付き SSE イベント契約 + ライフサイクル保証(`app/patterns/sse.py`, `app/api/v1/_stream.py`: `sse_max_events` 上限、`is_disconnected()` ポーリング、`CancelledError` 再送出、`finally` で generator クローズ、送信タイムアウト、`Cache-Control: no-cache` / `X-Accel-Buffering: no`) | §4.4(M-8 をほぼ解消) |
| Logfire スクラビング + fail-soft 初期化 | §4.7 の採用推奨 |
| サーバ発行・HMAC 署名つき session_id(`app/services/session_service.py`、principal 束縛、非 ASCII ガードつき定数時間比較) | hitl レーンの「セッションをクライアントに委ねない」規範の発展形 |
| エンドポイント別 LLM レート制限 + `storage_uri`(Redis)対応 | M-7 解消 |
| `/health/ready` の実疎通プローブ | L-5 解消 |
| contract-drift テスト、モデル ID ハードコード禁止テスト、`EXPECT_LIVE_TESTS` プラグイン、ファイルサイズポリシー、`block_network` fixture(unit 層のソケット遮断) | §4.8 のテスト規律(anti-false-green、ハーメチック化)|
| 二軸 LLM ジャッジ評価基盤(`evals/graders.py` ほか) | sandbox `EVAL-GRADERS.md` の outcome/behavior 二軸 |
| pre-commit gitleaks / pip-audit + dependabot + CI(カバレッジゲート含む) | sandbox のセキュリティ運用規範(§1 補足の脆弱性 85 件への対処経路) |
| `create_app()` factory 化(設定をモジュール import 時でなく factory 引数で解決) | sandbox `main.py` の app factory 規範 |
| pydantic-ai-slim を 1.70.0 → **1.107.1** に更新 | sandbox セキュリティチェックリストの「v1 系は >=1.99.0 フロア」を充足 |

### 7.3 指摘の解消状況マトリクス

| 指摘 | 状態 | 備考 |
|---|---|---|
| H-1 キャッシュ CancelledError 汚染 | ✅ 解消 | `rag_cache.py` の `except BaseException`。リーダーキャンセル時はフォロワーに `TimeoutError` を渡し 504 経路に載せる設計まで含めて適切 |
| H-2 `app_env` 自由文字列 | ❌ 未解消 | `app/config/security.py:133` に同一コードが残存。`Literal` 化されていない |
| H-3 ingest がキャッシュ無効化しない | ❌ 未解消 | キャッシュキーは `sha256(query\|max_retries)` のまま、ingest はキャッシュに触れない |
| H-4 チャット経路の予算・時間上限なし | ✅ 解消 | `run_guarded` + `chat_request_timeout`。監査証跡・許可リスト・承認フックまで実装 |
| H-5 履歴 1000 件で恒久破損 | ❌ 未解消 | `session_store/in_memory.py:210` が `ValueError` のまま。トリミングなし |
| M-1 lifespan の try/finally | 🔶 部分 | teardown は `_close_quietly` で個別保護されたが、startup 後半(agent 構築・logfire)の失敗では yield 前例外となり teardown 自体が走らず `cleanup_task`/HTTP client がリークする構造は残る |
| M-2 ストア設定のデッドコード | ✅ 解消 | factory + dry-run で配線 |
| M-3 InMemoryVectorStore の CPU バウンド・無ロック | 🔶 部分 | IDF キャッシュ追加で再計算は軽減、ただしイベントループ上の同期実行と無ロック変更は残存 |
| M-4 substring による transient 判定 | ❌ 未解消 | `exceptions.py` 同一実装。なお新規の `classify_usage_limit_exceeded`(guardrails.py)も例外メッセージ文字列で分類しており、pydantic-ai 更新でメッセージが変わると分類が壊れる同種の脆さがある |
| M-5 エラーエンベロープ 2 系統 | ❌ 未解消 | 401/403 は `{"detail": {...}}`、413/429/500 はフラットのまま |
| M-6 非 ASCII ヘッダで 500 | 🔶 部分 | session_id 側は `isascii()` ガードを実装(教訓は反映済み)、しかし `X-API-Key` の `compare_digest`(`deps/auth.py`)は未対策のまま |
| M-7 レート制限のプロセスローカル | ✅ 解消 | `storage_uri` 対応 + LLM エンドポイント別制限 |
| M-8 SSE のミドルウェア干渉・ヘッダ欠如 | 🔶 ほぼ解消 | ヘッダ・ライフサイクルは解消。`BaseHTTPMiddleware` 系スタックが SSE を包む構造自体は残る |
| M-9 `_get_cached_model` が引数無視 | ✅ 解消 | 引数が実際のキャッシュキー兼構築入力に |
| L-1 / L-2 / L-3 / L-4 / L-5 / L-9 | ✅ 解消 | dead `result.data` 削除、`limiter.hit` 適正化、`_last_access` ロック順序修正、Redis `aclose()` + lifespan クローズ、readiness 実疎通、重複 import 解消 |
| L-6 workflow のモデル解決フォールバック | ❌ 未解消 | `corrective_rag.py:89` に同一コード |
| L-7 CSP `'unsafe-inline'` / 無条件 HSTS | ❌ 未解消 | |
| L-8 `Vary` 上書き | ❌ 未解消 | |
| L-11 README の /health 例と実装の不一致 | ❌ 未解消 | README は `"healthy"`、実装は `"ok"` |
| §4.8 `asyncio_mode = "auto"` | ❌ 未採用 | `block_network` fixture(ソケット遮断)は `ALLOW_MODEL_REQUESTS=False` 相当以上の保護として採用済み |

### 7.4 ブランチの検証結果(2026-08-02)

| コマンド | 結果 |
|---|---|
| `uv sync --frozen` | 成功(pydantic-ai-slim 1.107.1) |
| `uv run ruff check app/ tests/` | All checks passed |
| `uv run ty check app/` | All checks passed |
| `uv run pytest tests/unit/ tests/integration/ tests/e2e/ -q` | **1113 passed, 7 failed, 1 skipped, 6 errors**(114 秒) |

失敗・エラーは全て検証環境起因でコード不良ではない:

- 6 errors: `test_docker_deployment.py`(Docker デーモンなし)
- 6 failed: `test_chroma_query_with_scores.py` — `ChromaVectorStore` 構築時の
  sentence-transformers モデルダウンロードがプロキシで 403(オフライン環境では実行不能)
- 1 failed: `test_block_network.py::test_block_network_blocks_af_inet6_connect` —
  IPv6 のない環境ではフィクスチャの遮断例外より先にソケット生成自体が
  `OSError(EAFNOSUPPORT)` で失敗する

ただし後 2 者は**テスト設計上の新規指摘**でもある: Chroma 統合テストは
インターネット必須(HF ダウンロード)なのにデフォルト実行層に置かれており、
ブランチ自身が導入した「blocking 層は決定的・オフラインのみ」という sandbox 由来の
CI 原則と矛盾する。IPv6 テストは `socket.has_ipv6` 等での skip ガードが必要。

### 7.5 残タスク(このブランチ取り込み後の優先度)

1. **P1**: H-2(`app_env` の `Literal` 化)、H-3(ingest でのキャッシュ無効化)、
   H-5(履歴上限のトリミング化)— いずれも小さい修正で、ブランチの新テスト基盤に
   回帰テストを足しやすい。
2. **P1**: M-6 の残り(`X-API-Key` の非 ASCII ガード。session_service に同じ対策の
   実装例がありコピーで済む)。
3. **P2**: Chroma 統合テストの環境ゲート化(`RUN_INTEGRATION_CHROMA=1` 等)と
   IPv6 テストの skip ガード。M-4 の型ベース分類化(新設 `classify_usage_limit_exceeded`
   も `UsageLimitExceeded` の属性ベース判定へ)。lifespan startup 後半の try/finally。
4. **P3**: 残りの Low(L-6/7/8/11)と `asyncio_mode = "auto"`。v2 移行は
   1.107.1 への更新で距離が縮まっており、§6 P3 のチェックリストが引き続き有効。
