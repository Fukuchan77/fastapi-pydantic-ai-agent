# 003-pydantic-ai-v2-migration → main マージ可否レビュー

**対象**: `003-pydantic-ai-v2-migration` (87 コミット / 235 ファイル / +26,456 −6,844)
**マージベース**: `956d9e2`（`main` の先端。`git rev-list --left-right --count` = `0 87`）
**レビュー日**: 2026-08-15
**観点**: コード品質・セキュリティ

---

## 1. 判定

**PR #21 の CI が green になり次第、マージを推奨する。**

当初は Blocker 3 件を検出し、うち実装欠陥 2 件と High 2 件を修正した。
その後 BL1 の解消のために PR を開いたところ、**PR CI で 4 件目の Blocker（BL0）が
発覚**した — Python 版数が固定されておらず、CI が 3.14 系を解決した結果
75 件のテストが同一の DeprecationWarning で落ちる、というものである。
ローカルでは 3.13 だったため一度も再現しなかった。

これは本レビューの最重要の教訓でもある。**ローカルの green は CI の green ではない。**
BL1（CI が一度も実行されていない）を「手続き上の事項」と評したのは過小評価であり、
実際にはそれ自体が未検出の Blocker を隠していた。

| 区分 | 件数 | 状態 |
|------|------|------|
| Blocker | 4 | すべて修正済み（BL0 は PR CI で初めて発覚） |
| High | 3 | 2 件修正済み / 1 件は運用対応 |
| Medium | 6 | 記録のみ（マージの障害ではない） |

### 検証結果（実測）

ブランチが要求する `pydantic-ai-slim>=2.27.0` を実際に導入し、全レーンを完走させた。

| チェック | コマンド | 修正前 | 修正後 |
|----------|----------|--------|--------|
| Lint | `ruff check app/ evals/ tests/` | All checks passed | All checks passed |
| Format | `ruff format --check` | clean | clean |
| 型検査 | `ty check app/ evals/` | All checks passed | All checks passed |
| テスト | `pytest tests/unit tests/integration tests/e2e` | **1431 passed / 22 skipped** | **1490 passed / 23 skipped** |
| カバレッジ | 80% ゲート | 96.44% | 96.19% |
| 依存監査 | `pip-audit`（除外 5 件） | No known vulnerabilities | No known vulnerabilities |

解決された実バージョン: `pydantic-ai 2.30.0` / Python 3.13.12。
ローカルで未実行のレーン: `test_docker_deployment.py`（Docker 不可）、
`test:redis`（Redis 不在）、`tests/local`（Ollama 不在）、`evals`（実 LLM 必要）。
このうち Docker レーン（7 件）と Redis レーンは **PR CI 側で実行され、いずれも通っている**
（run 31862477318 で Docker 7 件 PASSED、Redis サービスコンテナ起動成功）。

> **この表だけでマージ可否を判断してはならない。** BL0 が示すとおり、
> ローカルの green は CI の green を意味しない。判定の根拠は PR #21 の CI である。

---

## 2. 評価できる点

指摘の前に、このブランチの水準を明記しておく。エンジニアリング品質は例外的に高い。

- **依存制約に実証的根拠がある。** `fastapi<0.137` は二分探索の結果として
  「0.136.3 は router を flatten するが 0.137.2 はしない → slowapi の
  `_find_route_handler` が `app.routes` を非再帰走査して endpoint を見つけられず、
  `_should_exempt` が全リクエストを免除扱いにし、レート制限が**静かに**無効化される」
  という因果まで特定し、カナリアテスト名まで併記している。
  `pydantic-ai-litellm<0.3.0` は「0.x パッケージが private API 6 個に依存するので
  minor が breaking」という正しい理由で major ではなく minor を上限にしている。
- **v2 移行の手続きが厳格。** adapter-compatibility gate を PASS させてから制約を上げ、
  5 レーンで deprecation warning のセンサスを取り直したうえで
  `filterwarnings = ["error::DeprecationWarning"]` を維持。
  `tests/unit/test_pydantic_ai_api_lock.py`（380 行）が依存 API 表面を固定している。
- **grep で見えない挙動変化を捕捉している。** `docs/pydantic-ai-v2-behaviour-notes.md` は
  `InstrumentationSettings.use_aggregated_usage_attribute_names` の既定変更で
  agent-run スパンの累積トークン属性が `gen_ai.usage.*` → `gen_ai.aggregated_usage.*` に
  改名される件を、インストール済み 2.30.0 のソースを引用して記録している。
  アプリ側のコード変更ゼロで起きる影響であり、通常の差分レビューでは見落とす類の変化。
- **`pip-audit` 除外 5 件すべてに到達可能性分析がある。** PYSEC-2026-161 は
  「到達可能だが TrustedHostMiddleware で緩和」、他 4 件は
  「HTTPEndpoint 不使用 / `request.form()` 不使用 / StaticFiles 不使用」と個別判定。
- **Corrective RAG のキャンセル処理が正確。** `app/workflows/rag_cache.py` は
  `asyncio.shield` で follower が共有 future をキャンセルするのを防ぎ、
  `CancelledError` を `TimeoutError` に変換して follower が自身の 504 経路に乗れるようにし、
  後始末ブロックを await-free にして二重キャンセルでも孤児化しないようにしている。
  3 つの罠すべてに個別の根拠コメントがある。
- GitHub Actions は SHA ピン留め、pre-commit に gitleaks/pip-audit、
  `evals/` に LLM-judge 評価基盤、OWASP LLM Top 10 マッピング。

### 参照ガイドとの整合

- **PydanticAI 公式**: 評価器を文字列マッチから構造化 verdict へ移行、
  `NativeOutput` をモデルプロファイルの `supports_json_schema_output` で条件付き適用、
  `FallbackModel` チェーン、`UsageLimits` によるネイティブ予算制御、
  非推奨の `output_retries=` を避けて `retries={"output": ...}`、
  v2 で既定が反転する `end_strategy` を明示的に `"early"` に固定。いずれも定石に沿う。
- **Anthropic "Building effective agents"**: ツール許可リスト・承認フック・監査証跡
  （`app/agents/guardrails.py`）はガードレール指針に正面から対応。
  ただし実ツールは依然 `tools_mock.py` のみで、`chat_agent.py` の実検索は TODO のまま。
  **tool-calling パターンは本番構成で一度も実行されていない**。
- **Anthropic multi-agent research system**: 状態の永続化（Redis セッションストア配線）、
  ステップ単位のエラー分類、中断からの復旧可能性が改善されている。

---

## 3. Blocker

### BL0. Python 版数が固定されておらず、CI で全 RAG/レート制限系テストが落ちる（修正済み）

**BL1 を解消するために PR #21 を作成して初めて発覚した。** つまりこれは
「CI が一度も走っていない」ことが理論上のリスクではなく実在のリスクだった、という実証である。

初回 run 31862477318 は**失敗ではなく 6 時間のジョブタイムアウトで cancelled**。
打ち切りはキャンセルとして届くため pytest が失敗サマリに到達できず、
**その時点で失敗していたテストのトレースバックは 1 件も残らなかった**。
無制限の `await eval_started.wait()`（`test_rag_cache_generation.py`）が
高速な失敗を無限ハングに変えていたためである。

そのハングを潰した run 31880991303 で全体像が出た:

```
63 failed, 1426 passed, 8 skipped, 12 errors
```

**75 件すべてが同一原因**だった:

```
DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated and
slated for removal in Python 3.16; use inspect.iscoroutinefunction() instead
```

原因の連鎖:

1. `mise.toml` は `uv = "latest"` を固定するが **Python は固定していない**。
   `pyproject.toml` の `requires-python = ">=3.13"` は下限であって上限ではない。
2. そのため CI では uv が最新の互換 CPython（3.14 系）を解決し、
   ローカル開発機は 3.13.12 のままという**インタプリタの不一致**が生じた。
3. Python 3.14 は `asyncio.iscoroutinefunction` を非推奨化した。
4. `starlette` 0.52.x（`starlette/routing.py`）と `slowapi` 0.1.10
   （`slowapi/extension.py`）は依然としてこれを呼ぶ。しかも
   **`starlette<1.0` は本ブランチが意図的に置いた load-bearing なピン**
   （slowapi 0.1.10 が starlette 1.x でレート制限を静かに無効化するため）であり、
   アップグレードで回避できない。
5. `filterwarnings = ["error::DeprecationWarning"]`（意図的に module 非依存）が
   その呼び出しを 1 件残らず hard failure に変換した。

**修正**: インタプリタを下限ではなく固定にした。

- `.python-version`（uv が最優先で読む）と `mise.toml [tools] python`（CI の
  ツールチェーン導入元）の双方を `3.13` に固定。両方揃えたのは、uv の既定
  `python-preference = managed` が PATH 上の Python より uv 管理版を優先しうるため。
- `tests/unit/test_python_version_pin.py` を追加。`.python-version` /
  `mise.toml` / `requires-python` の整合に加え、
  **実行中のインタプリタ自体が固定版であること**を検証する
  （CI では無関係な 63 件ではなく、この 1 件が落ちるべきだった）。
- `CLAUDE.md` / `AGENTS.md`（規約により対で更新）に、
  deprecation センサスは依存バンプだけでなく**Python 版数変更でも無効化される**
  ことと、固定を解除してよい条件（starlette と slowapi が
  `inspect.iscoroutinefunction` へ移行したとき）を明記。

> **診断過程の訂正**: 初回 run の切り詰められたログでは失敗が
> eval agent の構造化出力系に集中して見えたため、当初その経路を疑うと報告した。
> これは誤りだった。完全なログでは失敗はレート制限・エラー封筒・Chroma ストア・
> RAG・e2e に広く分布しており、eval agent とは無関係である。
> 見えていたのはハング直前までに到達した部分集合にすぎなかった。

### BL1. このブランチの CI は一度も実行されていない（PR 作成で解消済み）

```
actions_list(list_workflow_runs, branch=003-pydantic-ai-v2-migration) → {"total_count": 0}
```

003 に対する PR も存在しない。追加された `.github/workflows/pr.yml` は
`on: pull_request: branches: [main]` のみをトリガとするため、**PR を開くまで走らない**。
87 コミット・pydantic-ai のメジャーバージョン移行が、自動検証を一度も経ずに
マージ判断の対象になっていた。

**本レビューでの緩和**: 上表のとおりローカルで lint / 型検査 / 1483 テスト /
カバレッジゲート / pip-audit を完走させ、すべて green であることを確認した。
ただしこれは PR CI の代替ではない（Redis レーンと Docker レーンは未実行）。

**残作業**: 003 → main の PR を開き、Redis サービスコンテナ付きの
`test:ci` + `test:redis`（`EXPECT_LIVE_TESTS=7`）+ `pip-audit` を green にする。

> `pr.yml` に `push: branches: [main]` トリガがないため、マージ後の main では
> CI が走らない。マージ後の回帰検知のために追加を推奨する。

### BL2. TRUSTED_PROXIES が CIDR 非対応 — 003 では実害が拡大していた（修正済み）

`app/middleware/rate_limit.py::get_client_identifier` は
`direct_client_ip in trusted_proxies` の**完全一致文字列比較**で、
`trusted_proxies` に書式検証もなかった。一方 `docs/production_deployment.md` は
全デプロイ先で CIDR を指示している:

| 行 | 記載 |
|----|------|
| 132 | `TRUSTED_PROXIES=["127.0.0.1", "10.0.0.0/8"]` (Nginx) |
| 277 | `TRUSTED_PROXIES=["10.0.0.0/8"]` (ALB の VPC CIDR) |
| 318 | Cloudflare の 15 CIDR レンジ |

文書どおりに設定すると membership テストは決して真にならず、`X-Forwarded-For` は
常に無視され、プロキシ配下の全クライアントがプロキシ IP という**単一バケット**に集約される。

**003 でこの欠陥は重くなっていた。** 003 は `enforce_llm_rate_limit`
（既定 30/minute）を chat / stream / RAG query に追加し、これを LLM コスト DoS の
主要防御に位置づけている（Req 11.3）。クライアント識別が壊れている以上、
**新設のコスト防御も同時に無効**であり、`storage_uri` による Redis 共有バケット化
（Req 11.4）も同じ理由で効果を失っていた。2 つの制御が同時に、静かに fail-open する。

**修正**:
- `app/middleware/rate_limit.py` — `ipaddress` によるネットワーク包含判定
  （`_is_trusted_proxy` / `_parse_trusted_proxies`）。裸の IP も CIDR も IPv6 も受理。
  解析結果はリクエスト毎に走るため `@cache`。
- `app/config/security.py` — `validate_trusted_proxies` で不正エントリを**起動時に拒否**。
  「決してマッチしない設定」で静かに動き続ける事態を防ぐ。
- 転送値も IP アドレスとして検証してからバケットキーにする。信頼済みプロキシが
  不正な値を中継しても無制限にバケットが増えない。
- 回帰テスト `tests/unit/test_middleware_rate_limit_cidr.py`（18 ケース）。
  Cloudflare の実レンジを含む CIDR 一致・不一致、IPv6、不正エントリの起動時拒否を検証。
- 既存 2 テストが `TRUSTED_PROXIES=["testclient"]`（Starlette TestClient の既定ホスト名、
  IP ではない）を使っていたため、`TestClient(app, client=("10.0.0.1", 12345))` +
  実 IP / CIDR 設定へ更新した。

### BL3. `.env.example` の SESSION_SIGNING_KEY プレースホルダが検証を通過していた（修正済み）

```
.env.example:13  API_KEY=your-api-key-here                     → 拒否される（正しい）
.env.example:18  SESSION_SIGNING_KEY=your-session-signing-key-here → 通過してしまう
```

`validate_session_signing_key_strength` は `api_key` の列挙集合（8 語）をそのまま
再利用しており、`your-session-signing-key-here` はそこに含まれず、
30 文字で 16 文字下限も満たす。実測:

```
old validator: placeholder? False | length ok? True -> ACCEPTED
```

`session_signing_key` は session_id を principal に束縛する唯一の秘密
（`app/services/session_service.py::_sign`）。既知の鍵なら任意の principal 向けの
session_id を偽造でき、Req 11.1/11.2 の IDOR 防御が成立しない。
`.env.example` をコピーして `API_KEY` だけ直したデプロイ — つまり最も自然な手順 —
が、**公開リポジトリに書かれた HMAC 鍵で稼働**することになる。
`API_KEY` 側が同種のプレースホルダを正しく弾くだけに、この非対称性が事故を招く。

**修正**: 列挙の追加ではなく、欠陥の「クラス」を閉じた。

- `app/config/_secret_placeholders.py` を新設。共通プレースホルダ集合に加えて
  `SHAPE_PATTERN`（`<words>-here` 形状）を持ち、`.env.example` の命名規約に従う値を
  列挙漏れに関係なく検出する。4 箇所に重複していた集合を 1 箇所に統合した
  （その重複こそが本欠陥の原因）。
- `.env.example` の値を `replace-me` に変更し、`openssl rand -hex 32` を併記。
- 回帰テスト `tests/unit/test_config_env_example_placeholders.py`。
  **コミット済みの `.env.example` を実際に読み**、そこに書かれた各 secret 値が
  `Settings()` で `ValidationError` になることを検証する。個別の文字列ではなく
  「`.env.example` の値は必ず拒否される」という不変条件を固定した。

---

## 4. High

### H1. `/rag/ingest` が過大チャンクで 500 を返していた（修正済み）

`IngestRequest.chunks` は `min_length=1, max_length=1000` で**リスト長**しか制約せず、
各チャンクの文字数制約がなかった。`InMemoryVectorStore.add_documents` は
`max_chunk_size`（既定 100,000 文字）超過で `ValueError` を送出するが、
route はこれを捕捉していない。リクエストボディ上限は 10 MB なので
100,001 文字のチャンクは容易に到達可能で、クライアント起因の入力エラーが
500 + トレースバックになる。003 は `app/api/errors.py` で
「あらゆるエラーを 1 つの平坦な封筒に収束させる」方針（Req 8）を徹底しているだけに、
ここだけ方針から外れていた。

**修正**: 二層で対応した。

- `app/models/rag.py` — `MAX_CHUNK_CHARS = 100_000` と `validate_chunk_sizes`
  バリデータを追加し、**入口で**弾く。Chroma / Ollama バックエンドは per-chunk 制限を
  持たないため、ストア非依存の wire contract として宣言する意味がある。
- `app/api/v1/rag.py` — ストアが wire contract より**厳しい**上限で構成された場合の
  受け皿として `ValueError` → `HTTPException(422, code="INVALID_DOCUMENT_CHUNK")`。
- 回帰テスト `tests/unit/api/v1/test_rag_ingest_chunk_size.py`。
  境界値、エラーメッセージが該当インデックスを示すこと、
  平坦な封筒（`detail` にネストしない）であることを検証。

### H2. TF-IDF 文書ベクトルがクエリ毎に全件再計算されていた（修正済み）

`_score_snapshot` は `snapshot.doc_tokens` を走査して**各文書の TF-IDF ベクトルを
毎回再構築**していた。`_idf_cache` は保持されるが、文書ベクトルのキャッシュはない。

003 は `asyncio.to_thread` + ロック + 不変スナップショットを導入しており、
これは正しい前進で、イベントループのブロックとリード時の破れ（tear）は解消されていた。
残っていたのは **1 クエリあたり O(コーパス全体) の CPU コストそのもの**。
既定上限は 1000 文書 × 100,000 文字で、CPU バウンドな Python 処理をスレッドに
逃がしても GIL 競合は残り、同時クエリはデフォルトのスレッドプール
（`min(32, cpu+4)`）を飽和させる。

**修正**: 003 のスナップショット方式を尊重し、その上にキャッシュを重ねた。

- `_doc_vectors` / `_doc_norms` を `_idf_cache` と同一ライフサイクルで保持し、
  `_invalidate_derived_caches()` で一括無効化。
- **ベクトルの構築自体もワーカースレッド内で行う。** ロック下で構築すると
  ingest 直後の最初のクエリだけイベントループを塞ぐことになり、003 の設計意図に反する。
  `_score_snapshot` が構築して返し、呼び出し側が `generation` 一致を確認してから
  書き戻す。スコアリング中にコーパスが変わった場合は破棄する。
- `_cosine_similarity_with_norms` でノルム再計算を排除し、内積は小さい方の
  ベクトルを走査する。スコアリングが O(文書数 × クエリ語数) に落ちる。
- 回帰テスト `tests/unit/stores/test_in_memory_vector_store_doc_vector_cache.py`（11 ケース）。
  再利用・無効化・**キャッシュ経路と非キャッシュ経路のスコア一致**・
  スコアリング中の並行 ingest で不整合なキャッシュが公開されないことを検証。

### H3. Dependabot の未処理 PR がブランチの load-bearing なピンと矛盾する（運用対応）

open PR 10 件のうち 2 件が 003 の意図的なピンと衝突する:

- **#17** `starlette 0.52.1 → 1.3.1` — 003 は `starlette>=0.52.1,<1.0` に固定。
  理由（slowapi 0.1.10 が starlette 1.x で例外ハンドラのディスパッチを誤り、
  `SlowAPIMiddleware` が `X-RateLimit-*` を出さなくなりレート制限が静かに無効化）は
  `pyproject.toml` に明記。
- **#12** `pydantic-ai-slim 1.70.0 → 1.99.0` — 003 は 2.27.0 へ移行済みで無意味。

新設の `.github/dependabot.yml` は starlette の major と fastapi の minor/major を
ignore するため**将来の提案は抑止される**が、設定より前に開かれた既存 PR は残る。

**残作業**: マージに合わせて #17 / #12 を理由付きでクローズする。

---

## 5. Medium（記録のみ）

1. `/docs` `/redoc` `/openapi.json` が全環境で未認証公開（`docs_url` 未設定）。
   `SecurityHeadersMiddleware` が CSP を docs ルートに限定しているので公開自体は
   意図的と読めるが、production では無効化を推奨。
2. `enforce_llm_rate_limit` は `limits.parse(llm_rate_limit)` と `identifier` のみで
   キーを作るため、chat / stream / RAG query の 3 ルートが**単一の 30/min 予算を共有**する。
   LLM 予算としては妥当な挙動だが、docstring の "per-route" は誤読を招く。
3. `filterwarnings = ["error::DeprecationWarning"]` は依存更新のたびにスイート全体を
   落としうる。センサス手順は文書化されているが運用負荷は高い。
4. `derive_principal_id` は API キーの SHA-256 前 16 文字。単一共有 API キーである限り
   全呼び出し元の principal.id が同一になるため、IDOR 防御の実体は
   `secrets.token_urlsafe(16)`（128 bit）であって principal 束縛ではない。
   複数キー化に向けた前方互換設計としては妥当。
5. 実ツールは `tools_mock.py` のみ。`chat_agent.py` の実検索は TODO のまま。
6. ~~`pr.yml` に `push: branches: [main]` トリガがなく、マージ後の main で CI が走らない。~~
   **対応済み**。`push: branches: [main]` を追加し、`cancel-in-progress` を
   `${{ github.event_name == 'pull_request' }}` に変更した。無条件 `true` のままだと
   あるマージが 1 つ前のマージコミットを検証中の実行をキャンセルしてしまい、
   push トリガを足した意味が失われるため。ワークフロー名 `PR CI` は据え置き
   （branch protection の required status check が名前で照合されるため）。

---

## 6. 本レビューでの変更

### 新規

| ファイル | 対応 |
|----------|------|
| `app/config/_secret_placeholders.py` | BL3 |
| `tests/unit/test_config_env_example_placeholders.py` | BL3 |
| `tests/unit/test_middleware_rate_limit_cidr.py` | BL2 |
| `tests/unit/api/v1/test_rag_ingest_chunk_size.py` | H1 |
| `tests/unit/stores/test_in_memory_vector_store_doc_vector_cache.py` | H2 |
| `docs/003-merge-readiness-review.md` | 本書 |

### 修正

| ファイル | 対応 |
|----------|------|
| `app/middleware/rate_limit.py` | BL2: CIDR 包含判定、転送値の検証 |
| `app/config/security.py` | BL2: `trusted_proxies` 起動時検証 / BL3: 共通判定へ移行 |
| `app/config/llm.py`, `app/config/observability.py` | BL3: 共通判定へ移行（重複解消） |
| `.env.example` | BL3: 安全なプレースホルダと生成コマンド |
| `app/models/rag.py` | H1: per-chunk 制約 |
| `app/api/v1/rag.py` | H1: `ValueError` → 422 |
| `app/stores/vector_store/in_memory.py` | H2: 文書ベクトル + ノルムのキャッシュ |
| `tests/unit/test_middleware_rate_limit_{proxy,trusted_proxies}.py` | BL2: 実 IP へ更新 |

---

## 7. マージ前の残作業

| # | 内容 | 状態 |
|---|------|------|
| 1 | 003 → main の PR を開き、PR CI（Redis レーン含む）を green にする（BL1） | PR 作成済み・CI 結果待ち |
| 2 | Dependabot PR #17 / #12 を理由付きでクローズする（H3） | 対応済み |
| 3 | `pr.yml` に `push: branches: [main]` トリガを追加する（Medium 6） | 対応済み |

## 8. 関連ブランチ

`claude/main-branch-merge-review-9m5z3q` は同じ `956d9e2` から分岐し、
同一の欠陥群を独立に修正した先行レビューの成果物である。003 の方が範囲が広く
実装も優れているため、**003 を正とし同ブランチは破棄する**方針が確認されている。
同ブランチにしかなかった修正（CIDR 判定、文書ベクトルキャッシュ）は本レビューで
003 側へ移植済みであり、未回収の内容は残っていない。
