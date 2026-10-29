<p align="center">
  <img src="docs/images/banner.svg" alt="Open Game Studio — Your local game production workbench for Codex" width="100%" />
</p>

<p align="center">
<a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a> · <a href="README.zh-TW.md">繁體中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.ko.md">한국어</a>
</p>

<h1 align="center">Open Game Studio</h1>

<p align="center"><strong>Codex で使う、ローカルのゲーム制作ワークベンチ。</strong></p>

<p align="center">
  <img src="docs/images/badges.svg" alt="Alpha · Node.js 24+ · Codex MCP Apps · Local SQLite · MIT License" width="100%" />
</p>

参考画像、ドキュメント、メディア、3D アセットをひとつのキャンバスに。バージョン、制作上の判断、書き出し履歴を、手元のプロジェクトファイルと結び付けて管理できます。

> **Alpha · macOS 用インストーラー · MIT ライセンス。** ローカルでの整理は単独で利用できます。AI 制作には自分で設定したツールとアカウントが必要です。README は多言語ですが、ワークベンチの UI は現在中国語です。

[試す](#try-it) · [Monster](#monster) · [インストール](#install) · [機能](#features) · [画面](#gallery) · [ドキュメント](#documentation) · [制限](#limits)

---

<a id="try-it"></a>

## 会話から試す

インストール後、新しい Codex チャットに貼り付けてください：

```text
Open Game Studio ワークベンチを開いてください。
選択したフォルダに「森の遺跡」プロジェクトを作成してください。
キャラクターの企画書と参考画像をキャンバスに配置し、
キャラクター制作グループの入力として接続してください。
制作を開始する前に、固定した入力一覧を見せてください。
```

整理と確認はワークベンチで、制作の指示は会話で行います。接続は入力を指定するもので、制作の開始は別の操作です。

ファイル整理の出発点が必要なら、任意の[プロジェクトファイルガイド（中国語）](skills/studio-project/references/project-file-guide.md)を参照し、`ゲームプロジェクトに Open Game Studio ファイルガイドの標準構成を採用してください。` と依頼できます。一部のルールだけでも利用でき、移行を明示的に依頼するまでは既存の配置を維持します。

<a id="monster"></a>

## Monster モジュール：クラウド GPU リギング・姿勢学習・アニメーション生成

[![Open in Colab](docs/images/open-in-colab.svg)](https://colab.research.google.com/github/GrinZero/codex-game-studio/blob/main/packages/monster/notebooks/colab.ipynb)

[Monster ガイド](packages/monster/README.md) · [ワークフローと G1/G2/G3 監査スキル](packages/monster/skills/monster-workflow/SKILL.md) · [Colab ノートブック](packages/monster/notebooks/colab.ipynb)

Colab G4 GPU 上で、静的な 3D クリーチャー／キャラクターメッシュから候補スケルトンのリギング、専用姿勢プローブの学習、アニメーションのベイクまでを一貫して実行します。手元の Mac に CUDA、PyTorch、Blender を入れる必要はありません。明示的な **G1 / G2 / G3** ユーザー監査ゲートにより、骨格の選定・姿勢サンプリング・モーション品質を人間が確認しながら進められます。

| ステージ | パイプラインと主な機能 |
| --- | --- |
| **1. 候補リギング（`rig` · G1）** | **SkinTokens + Blender Cycles GPU**（`OPTIX`/`CUDA`、GPU `OpenImageDenoise`、永続 BVH）で 4 種類の候補リグを生成し、独立した装飾トポロジを保持したまま **G1** 選定用の多視点比較シートをレンダリングします。 |
| **2. 姿勢プローブ（`train` · G2/G3）** | **G2** の 50 ポーズ確認後に正・側面の平行投影ペア（標準 `1344×768`、または任意の `2560×1440` 2K モード）を合成し、専用 **DINOv2** ヒートマップ＋深度／可視性ヘッド（SDPA Attention、TF32、5×5 ピーク質量信頼度）を学習して **G3** 動画復元試験で検証します。 |
| **3. アクションベイク（`animate`）** | フル精度 **MiniMax H3**（`bf16`/`fp16`、2K 時は 2 段階 3D 潜在超解像＋タイル精修）で同期デュアルビュー動画を生成し、2D Hampel スパイク除去・5 点 Savitzky–Golay 四元数平滑化・制限付きローカルオイラー多子骨 Swing-Twist IK で `<action>_animated.glb` をベイクします。 |
| **4. ゼロ CUDA ローカルクライアント** | 上の **Open in Colab** バッジから G4 Worker を起動（標準で GitHub 最新版を取得し Google Drive キャッシュを再利用）。macOS では `npm run monster:call` 初回実行時に `packages/monster/.venv` を自動構築し、デスクトップ版 Google Drive の `endpoint_registry.json` 自動検出（または 1 行 `connect`）と `protocol_version` 検証を行って結果をダウンロードします。 |

Colab Worker の起動後、リポジトリのルートから直接呼び出せます：

```sh
npm run monster:call -- status
npm run monster:call -- rig creature_01 /absolute/path/input.glb
npm run monster:call -- train creature_01 --samples 3000 --epochs 15
npm run monster:call -- animate creature_01 'idle:loop walk:loop'
```

ダウンロードした成果物はワークベンチで不変のアセット版として取り込めます。オフライン配布用アーカイブは `npm run monster:package` で作成できます。

<a id="install"></a>

## インストールと起動

**macOS**、**Node.js 24+**（npm を含む）、`codex plugin` コマンドを備えた Codex CLI が必要です。埋め込みワークベンチには MCP Apps 対応の Codex ホストも必要です。

ワンラインでインストール（またはリポジトリをクローンして実行）：

```sh
curl -fsSL https://raw.githubusercontent.com/GrinZero/codex-game-studio/main/scripts/bootstrap.sh | bash
```

```sh
git clone https://github.com/GrinZero/codex-game-studio.git
cd codex-game-studio
npm ci
npm run install:local
```

インストーラーは両方のワークベンチをビルドし、独立したランタイムをコピーして `game-studio@game-studio-local` をインストールします。ブラウザ版はユーザーの LaunchAgent で起動します。現在はローカルのソースからのインストールのみで、npm パッケージやリモートのプラグインマーケット経由の手順は提供していません。

1. **新しい Codex チャット**を開き、`Open Game Studio ワークベンチを開いてください。` と入力します。
2. ホストが登録済みの入口を表示する場合は、サイドバーの **…** メニューから **Open Game Studio** を選べます。表示されない場合は Codex を再起動してください。
3. ブラウザ版は [http://127.0.0.1:4310](http://127.0.0.1:4310) です。チェックアウト先で `npm run demo` を実行すると、オリジナルの **Star Maker** サンプル（表示名 `星铸者 · 示例`）を追加できます。

デモはイラストと企画書を 2 つのキャンバスに読み込みます。生成、レビュー、エンジン検証は実行しません。

更新、削除、パスの変更、MCP の直接接続は [セットアップ](docs/SETUP.md) を参照してください。

<a id="features"></a>

## できること

| 機能 | 用途 |
| --- | --- |
| ローカルプロジェクト | フォルダの作成・関連付けとカバー設定。既存ファイルの配置を維持します。 |
| アセットのバージョン管理 | ファイルセットを SHA256 と変更不可の履歴付きで読み込みます。新しい版を追加してもキャンバスの参照は変わりません。 |
| ビジュアルキャンバス | アセットとメモの配置、入力の接続、別キャンバスでの版の再利用、レイアウトの取り消し・やり直し。 |
| ドキュメントとメディア | Markdown の編集を新しい版として保存。画像、音声、動画、自己完結型 GLB とアニメーションをプレビュー。 |
| アセット制作 | 画像、動画、モデル、骨格アニメーション、音声のタスクを準備。参考を固定してローカル Codex と設定済みツールで実行。 |
| キャラクター工程 | 企画 → 参考 → 候補 → 外観レビュー → 組立 → リギング → 変形レビュー → 書き出し。判断の保存と一部工程の再実行に対応。 |
| 追跡できる納品 | 明示した版をファイル一覧とハッシュ付きで書き出します。エンジンでの受け入れ確認は別途行います。 |

操作の詳細は [ワークベンチガイド](docs/WORKBENCH.md) を参照してください。

<a id="gallery"></a>

## ワークベンチを見る

リポジトリのオリジナル素材を使った、隔離されたローカルデモの実際の画面です。現在の中国語 UI を示しており、完成した制作アセットではありません。

**プロジェクトライブラリ · ダークテーマ**

![プロジェクトライブラリ · ダークテーマ](docs/images/projects.png)

**入力を接続したキャンバス · ダークテーマ**

![入力を接続したキャンバス · ダークテーマ](docs/images/canvas.png)

[Image provenance](docs/images/README.md)

<a id="documentation"></a>

## ツールとドキュメント

**26 個のドメイン MCP ツール + UI 専用ブリッジ 1 個 · 13 個のワークフロースキル。** HTTP 版と MCP Apps 埋め込み版は同じプロジェクトコアとアセット履歴を共有します。

| 入口 | 内容 |
| --- | --- |
| [セットアップ・データパス・MCP 設定](docs/SETUP.md) | インストール、更新、削除、直接接続。 |
| [ワークベンチガイド](docs/WORKBENCH.md) | キャンバス編集、版の選択、制作の引き継ぎ。 |
| [ツール契約](docs/TOOL_CONTRACT.md) | 全ツールの索引、入力 schema、状態のルール。 |
| [検証記録](docs/VERIFICATION.md) | 実行済みの検証と、実環境での確認が必要な項目。 |
| [実装状況](docs/IMPLEMENTATION.md) | 構成、未完了の段階、今後の作業。 |
| [Monster モジュール](packages/monster/README.md) | [![Open in Colab](docs/images/open-in-colab.svg)](https://colab.research.google.com/github/GrinZero/codex-game-studio/blob/main/packages/monster/notebooks/colab.ipynb) 起動、Worker ソース、自動検出ローカルクライアント。 |
| [ローカル GLB 圧縮スキル](skills/glb-compress/SKILL.md) | 単独配布できるオフライン CLI。Draco、テクスチャ、サイズ変更、比較レポートに対応。 |

<a id="limits"></a>

## 現在の制限

- **初期 Alpha：** ローカルプロジェクト、バージョン管理、ワークフローは実装済みです。組み込み Provider API、認証情報管理、費用パネル、ゲームフォルダの自動走査、DB バックアップ、Blender/Godot 検証 worker は未完成です。

- **制作ツールは各自で設定：** ローカル Codex 実行環境にモデルサービスは同梱されません。機能、ログイン、料金、送信の承認は個々のツールによります。ツール不足や承認待ちは失敗理由として記録し、仮のアセットで成功を装いません。ローカルタスクの停止は送信済みのリモート処理を取り消しません。

- **プレビューは品質保証ではありません：** GLB 構造やファイルハッシュの検証では、メッシュ品質、Blend の編集可能性、骨格の変形、エンジン互換性は確認できません。マニフェストのエンジン検証は `not_verified` のままです。

- **固定した版は維持：** 新しい版はキャンバスやレビューの採用版を置き換えません。SVG と外部ファイルに依存する glTF の読み込みは非対応です。取り消し・やり直しは現在レイアウトのみです。

- **ホストによる違い：** MCP App の入口は登録済みで、サンドボックス検証も済んでいます。ネイティブ Codex での表示、キャッシュ、再読み込みの挙動はホストに依存します。

<a id="contributing"></a>

## 開発に参加する

開発コマンド、検証、ビルド識別子、ランタイムへの同期は [CONTRIBUTING.md](CONTRIBUTING.md) を参照してください。不具合報告には表示中のビルド識別子、ホスト、再現手順、期待する動作を添え、認証情報や非公開ファイルは除いてください。

<a id="license"></a>

## ライセンスと出典

ワークベンチには [MIT ライセンス](LICENSE)、[Monster](packages/monster/LICENSING.md) とそのエクスポートされたワークフロースキルには [AGPL-3.0-only](packages/monster/LICENSE) が適用されます。依存ライブラリのライセンスとサンプルの出典は [第三者ライセンス情報](licenses/THIRD_PARTY.md) に記載しています。

README の見せ方は [Tripo Studio for Codex](https://github.com/GrinZero/tripo-studio-plugin) を参考にしました。ここで使うバナーとデモ画面は本プロジェクトのものです。
