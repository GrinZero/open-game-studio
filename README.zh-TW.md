<p align="center">
  <img src="docs/images/banner.svg" alt="Open Game Studio — Your local game production workbench for Codex" width="100%" />
</p>

<p align="center">
<a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a> · <a href="README.zh-TW.md">繁體中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.ko.md">한국어</a>
</p>

<h1 align="center">Open Game Studio</h1>

<p align="center"><strong>在 Codex 裡，把遊戲創作整理成一個本機工作台。</strong></p>

<p align="center">
  <img src="docs/images/badges.svg" alt="Alpha · Node.js 24+ · Codex MCP Apps · Local SQLite · MIT License" width="100%" />
</p>

將參考圖、文件、媒體與 3D 資產放在同一張畫布上，讓版本、製作決策與匯出紀錄都能追溯到本機專案檔案。

> **Alpha · macOS 安裝程式 · MIT 授權條款。** 本機整理可獨立使用；AI 製作需要自行設定工具與帳號。README 提供多語言版本，工作台介面目前仍使用簡體中文。

[試用](#try-it) · [安裝](#install) · [功能](#features) · [預覽](#gallery) · [文件](#documentation) · [限制](#limits)

---

<a id="try-it"></a>

## 先試一句話

安裝後，在新的 Codex 對話中貼上：

```text
開啟 Open Game Studio 工作台。
在我選擇的資料夾裡建立「林谷遺跡」專案。
把角色目標文件與參考圖放到畫布上，
連接為角色製作群組的輸入。
開始製作前，先顯示凍結的輸入清單。
```

工作台用來整理與檢視，對話用來表達製作意圖。連線提供輸入，啟動製作是另一個操作。

還沒有自己的檔案組織習慣？可以閱讀選用的[專案檔案指南（簡體中文）](skills/studio-project/references/project-file-guide.md)，然後說：`在我的遊戲專案中採用 Open Game Studio 檔案規範的標準方案。` 可以只選部分規則；既有專案保留原配置，搬移需另行明確提出。

<a id="install"></a>

## 安裝與開啟

需要 **macOS**、**Node.js 24+**（含 npm），以及提供 `codex plugin` 指令的 Codex CLI。內嵌工作台還需要宿主支援 MCP Apps。

複製儲存庫後執行本機安裝程式：

```sh
git clone https://github.com/GrinZero/codex-game-studio.git
cd codex-game-studio
npm ci
npm run install:local
```

安裝程式會建置兩套工作台、複製獨立執行環境、安裝 `game-studio@game-studio-local`，並透過使用者層級 LaunchAgent 啟動瀏覽器工作台。目前採用本機原始碼安裝，未提供 npm 套件或遠端外掛市集安裝方式。

1. 開啟 **新的 Codex 對話**，輸入：`開啟 Open Game Studio 工作台。`
2. 宿主顯示已註冊入口時，也可從側邊欄 **…** 選單選擇 **Open Game Studio**。入口未出現時重新啟動 Codex。
3. 備用瀏覽器入口為 [http://127.0.0.1:4310](http://127.0.0.1:4310)。在儲存庫目錄執行 `npm run demo`，即可匯入原創的 **星鑄者範例**（介面顯示 `星铸者 · 示例`）。

範例會將插畫與目標文件匯入兩張畫布，不執行生成、審核或引擎驗收。

更新、移除、自訂路徑與直接 MCP 連線見 [安裝設定](docs/SETUP.md)。

<a id="features"></a>

## 可以做什麼

| 功能 | 用途 |
| --- | --- |
| 本機專案 | 建立或連結資料夾、設定封面，保留既有檔案的組織方式。 |
| 資產版本 | 以檔案集匯入，記錄 SHA256 與不可變歷史；新版本不會取代畫布已引用的版本。 |
| 自由畫布 | 擺放資產與便箋、連接輸入、跨畫布重用版本，並復原或重做版面配置操作。 |
| 文件與媒體 | Markdown 編輯儲存為新版本；預覽圖片、影音與自包含 GLB，控制模型動畫。 |
| 資產製作 | 準備圖片、影片、模型、骨架動畫或音訊任務，凍結參考後呼叫本機 Codex 與已設定工具。 |
| 角色流程 | 依目標 → 參考 → 候選 → 外觀審核 → 組裝 → 綁定 → 變形審核 → 匯出推進，保存決策並局部重跑。 |
| 可追溯交付 | 依明確版本匯出，附檔案清單與雜湊值；引擎驗收另外驗證。 |

操作細節見 [工作台指南](docs/WORKBENCH.md)。

<a id="gallery"></a>

## 工作台預覽

截圖來自隔離的本機範例環境，使用儲存庫原創素材，展示目前的簡體中文介面，不代表已完成的正式製作資產。

**專案庫 · 深色主題**

![專案庫 · 深色主題](docs/images/projects.png)

**連線畫布 · 深色主題**

![連線畫布 · 深色主題](docs/images/canvas.png)

[Image provenance](docs/images/README.md)

<a id="documentation"></a>

## 工具與文件

**26 個領域 MCP 工具 + 1 個介面專用橋接工具 · 11 個工作流程技能。** HTTP 與 MCP 內嵌工作台共用專案核心及資產歷史。

| 入口 | 內容 |
| --- | --- |
| [安裝、資料路徑與 MCP 設定](docs/SETUP.md) | 安裝、更新、移除與直接連線。 |
| [工作台指南](docs/WORKBENCH.md) | 畫布編輯、版本選擇與製作交接。 |
| [工具契約](docs/TOOL_CONTRACT.md) | 完整工具索引、輸入 schema 與狀態規則。 |
| [驗證紀錄](docs/VERIFICATION.md) | 已執行的檢查與仍需實測的功能。 |
| [實作狀態](docs/IMPLEMENTATION.md) | 架構、未完成階段與後續計畫。 |
| [Monster 模組](packages/monster/README.md) | [Open in Colab](https://colab.research.google.com/github/GrinZero/codex-game-studio/blob/main/packages/monster/notebooks/colab.ipynb) 一鍵啟動、Worker 原始碼與自動配對本機用戶端。 |

<a id="limits"></a>

## 目前限制

- **仍處於 Alpha：** 本機專案、版本管理與製作流程已實作；內建 Provider API、憑證管理、完整費用面板、遊戲目錄自動掃描、資料庫備份和 Blender/Godot 驗證 worker 仍未完成。

- **製作工具自行設定：** 本機 Codex 執行器未內附模型服務；可用功能、登入、費用與提交確認取決於個別工具。缺少工具或需要確認時明確記錄失敗原因，不製造佔位資產。停止本機任務不會取消已提交的遠端工作。

- **預覽不等於驗收：** GLB 結構檢查與檔案雜湊不能證明網格品質、Blend 可編輯性、骨架變形或引擎相容性。匯出清單的引擎驗證維持 `not_verified`。

- **固定版本維持不變：** 新資產版本不會取代畫布或審核已採用的版本。暫不匯入 SVG 或依賴外部檔案的 glTF；復原/重做目前只涵蓋版面配置。

- **宿主行為有差異：** MCP App 入口已註冊並通過沙箱測試；原生 Codex 中的顯示、快取與重新整理行為取決於宿主。

<a id="contributing"></a>

## 參與開發

開發指令、檢查、建置標識與執行目錄同步見 [CONTRIBUTING.md](CONTRIBUTING.md)。回報問題時請附介面建置標識、宿主、重現步驟與預期結果，並移除憑證及私人專案檔案。

<a id="license"></a>

## 授權與來源

本專案採用 [MIT 授權條款](LICENSE)。相依套件授權與原創範例來源見 [第三方聲明](licenses/THIRD_PARTY.md)。

README 展示方式參考 [Tripo Studio for Codex](https://github.com/GrinZero/tripo-studio-plugin)；此處的橫幅與範例截圖來自本專案。
