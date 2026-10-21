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

[試す](#try-it) · [インストール](#install) · [機能](#features) · [画面](#gallery) · [ドキュメント](#documentation) · [制限](#limits)

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

<a id="install"></a>

## インストールと起動

**macOS**、**Node.js 24+**（npm を含む）、`codex plugin` コマンドを備えた Codex CLI が必要です。埋め込みワークベンチには MCP Apps 対応の Codex ホストも必要です。

リポジトリをクローンして、ローカルインストーラーを実行します：

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

**26 個のドメイン MCP ツール + UI 専用ブリッジ 1 個 · 11 個のワークフロースキル。** HTTP 版と MCP Apps 埋め込み版は同じプロジェクトコアとアセット履歴を共有します。

| 入口 | 内容 |
| --- | --- |
| [セットアップ・データパス・MCP 設定](docs/SETUP.md) | インストール、更新、削除、直接接続。 |
| [ワークベンチガイド](docs/WORKBENCH.md) | キャンバス編集、版の選択、制作の引き継ぎ。 |
| [ツール契約](docs/TOOL_CONTRACT.md) | 全ツールの索引、入力 schema、状態のルール。 |
| [検証記録](docs/VERIFICATION.md) | 実行済みの検証と、実環境での確認が必要な項目。 |
| [実装状況](docs/IMPLEMENTATION.md) | 構成、未完了の段階、今後の作業。 |
| [Monster モジュール](packages/monster/README.md) | [Open in Colab](https://colab.research.google.com/github/GrinZero/codex-game-studio/blob/main/packages/monster/notebooks/colab.ipynb) 起動、Worker ソース、自動検出ローカルクライアント。 |

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

このプロジェクトは [MIT ライセンス](LICENSE) の下で公開されています。依存ライブラリのライセンスとサンプルの出典は [第三者ライセンス情報](licenses/THIRD_PARTY.md) に記載しています。

README の見せ方は [Tripo Studio for Codex](https://github.com/GrinZero/tripo-studio-plugin) を参考にしました。ここで使うバナーとデモ画面は本プロジェクトのものです。
