<p align="center">
  <img src="docs/images/banner.svg" alt="Open Game Studio — Your local game production workbench for Codex" width="100%" />
</p>

<p align="center">
<a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a> · <a href="README.zh-TW.md">繁體中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.ko.md">한국어</a>
</p>

<h1 align="center">Open Game Studio</h1>

<p align="center"><strong>Codex 안에서 사용하는 로컬 게임 제작 워크벤치.</strong></p>

<p align="center">
  <img src="docs/images/badges.svg" alt="Alpha · Node.js 24+ · Codex MCP Apps · Local SQLite · MIT License" width="100%" />
</p>

참고 이미지, 문서, 미디어와 3D 에셋을 하나의 캔버스에 모으세요. 버전, 제작 결정과 내보내기 기록을 내 컴퓨터의 프로젝트 파일과 연결해 관리할 수 있습니다.

> **Alpha · macOS 설치 프로그램 · MIT 라이선스.** 로컬 정리는 독립적으로 사용할 수 있습니다. AI 제작에는 직접 설정한 도구와 계정이 필요합니다. README는 다국어로 제공되지만 워크벤치 UI는 현재 중국어입니다.

[시작](#try-it) · [설치](#install) · [기능](#features) · [화면](#gallery) · [문서](#documentation) · [제한](#limits)

---

<a id="try-it"></a>

## 대화로 시작하기

설치 후 새 Codex 채팅에 다음 내용을 붙여 넣으세요:

```text
Open Game Studio 워크벤치를 열어 주세요.
제가 선택한 폴더에 숲의 유적 프로젝트를 만들어 주세요.
캐릭터 기획 문서와 참고 이미지를 캔버스에 놓고,
캐릭터 제작 그룹의 입력으로 연결해 주세요.
제작을 시작하기 전에 고정된 입력 목록을 보여 주세요.
```

워크벤치에서는 정리와 확인을, 대화에서는 제작 지시를 합니다. 연결은 입력을 지정하며, 제작 시작은 별도 작업입니다.

파일 정리를 시작할 기준이 필요하다면 선택 사항인 [프로젝트 파일 가이드(중국어)](skills/studio-project/references/project-file-guide.md)를 읽고 `내 게임 프로젝트에 Open Game Studio 파일 가이드의 표준 구성을 적용해 주세요.`라고 요청하세요. 일부 규칙만 선택할 수 있으며, 이전을 명시적으로 요청하기 전까지 기존 배치를 유지합니다.

<a id="install"></a>

## 설치하고 열기

**macOS**, **Node.js 24+**(npm 포함), `codex plugin` 명령을 제공하는 Codex CLI가 필요합니다. 내장 워크벤치에는 MCP Apps를 지원하는 Codex 호스트도 필요합니다.

저장소를 복제한 뒤 로컬 설치 프로그램을 실행하세요:

```sh
git clone https://github.com/GrinZero/codex-game-studio.git
cd codex-game-studio
npm ci
npm run install:local
```

설치 프로그램은 두 워크벤치를 빌드하고 독립 런타임을 복사한 뒤 `game-studio@game-studio-local`을 설치합니다. 브라우저 버전은 사용자 LaunchAgent로 실행됩니다. 현재는 로컬 소스 설치를 제공하며, npm 패키지나 원격 플러그인 마켓 설치 방식은 제공하지 않습니다.

1. **새 Codex 채팅**을 열고 `Open Game Studio 워크벤치를 열어 주세요.`라고 입력하세요.
2. 호스트가 등록된 진입점을 표시하면 사이드바의 **…** 메뉴에서 **Open Game Studio**를 선택할 수도 있습니다. 보이지 않으면 Codex를 다시 시작하세요.
3. 브라우저 버전은 [http://127.0.0.1:4310](http://127.0.0.1:4310)에서 열립니다. 저장소 디렉터리에서 `npm run demo`를 실행하면 오리지널 **Star Maker** 예제(표시 이름 `星铸者 · 示例`)를 추가할 수 있습니다.

예제는 일러스트와 기획 문서를 두 캔버스에 가져옵니다. 생성, 검토 또는 엔진 검증은 실행하지 않습니다.

업데이트, 제거, 경로 변경과 직접 MCP 연결은 [설정 가이드](docs/SETUP.md)를 참고하세요.

<a id="features"></a>

## 주요 기능

| 기능 | 용도 |
| --- | --- |
| 로컬 프로젝트 | 폴더를 만들거나 연결하고 표지를 설정합니다. 기존 파일 배치를 유지합니다. |
| 에셋 버전 관리 | 파일 묶음을 SHA256 및 변경 불가 이력과 함께 가져옵니다. 새 버전이 생겨도 캔버스 참조는 유지됩니다. |
| 시각적 캔버스 | 에셋과 메모 배치, 입력 연결, 캔버스 간 버전 재사용, 레이아웃 실행 취소와 다시 실행. |
| 문서와 미디어 | Markdown 편집 내용을 새 버전으로 저장합니다. 이미지, 오디오, 영상, 자체 포함 GLB 및 애니메이션을 미리 볼 수 있습니다. |
| 에셋 제작 | 이미지, 영상, 모델, 골격 애니메이션 또는 오디오 작업을 준비합니다. 참고 자료를 고정한 뒤 로컬 Codex와 설정된 도구로 실행합니다. |
| 캐릭터 워크플로 | 기획 → 참고 → 후보 → 외형 검토 → 조립 → 리깅 → 변형 검토 → 내보내기. 결정을 저장하고 일부 단계만 다시 실행합니다. |
| 추적 가능한 결과물 | 명시한 버전을 파일 목록 및 해시와 함께 내보냅니다. 엔진에서의 검증은 별도로 진행합니다. |

자세한 조작 방법은 [워크벤치 가이드](docs/WORKBENCH.md)를 참고하세요.

<a id="gallery"></a>

## 워크벤치 미리 보기

저장소의 오리지널 소재를 사용한 격리된 로컬 예제의 실제 화면입니다. 현재 중국어 UI를 보여 주며, 완성된 제작 에셋을 의미하지 않습니다.

**프로젝트 라이브러리 · 다크 테마**

![프로젝트 라이브러리 · 다크 테마](docs/images/projects.png)

**입력이 연결된 캔버스 · 다크 테마**

![입력이 연결된 캔버스 · 다크 테마](docs/images/canvas.png)

[Image provenance](docs/images/README.md)

<a id="documentation"></a>

## 도구와 문서

**도메인 MCP 도구 26개 + 앱 전용 브리지 1개 · 워크플로 스킬 11개.** HTTP 버전과 MCP Apps 내장 버전은 같은 프로젝트 코어와 에셋 이력을 공유합니다.

| 문서 | 내용 |
| --- | --- |
| [설치·데이터 경로·MCP 설정](docs/SETUP.md) | 설치, 업데이트, 제거와 직접 연결. |
| [워크벤치 가이드](docs/WORKBENCH.md) | 캔버스 편집, 버전 선택과 제작 인계. |
| [도구 계약](docs/TOOL_CONTRACT.md) | 전체 도구 목록, 입력 schema와 상태 규칙. |
| [검증 기록](docs/VERIFICATION.md) | 수행한 검증과 실제 환경에서 확인해야 할 항목. |
| [구현 현황](docs/IMPLEMENTATION.md) | 구조, 미완료 단계와 향후 작업. |
| [Monster 모듈](packages/monster/README.md) | [Open in Colab](https://colab.research.google.com/github/GrinZero/codex-game-studio/blob/main/packages/monster/notebooks/colab.ipynb) 실행, Worker 소스 및 자동 탐색 로컬 클라이언트. |

<a id="limits"></a>

## 현재 제한 사항

- **초기 Alpha:** 로컬 프로젝트, 버전 관리와 워크플로는 구현되었습니다. 내장 Provider API, 인증 정보 관리, 전체 비용 패널, 게임 폴더 자동 탐색, DB 백업, Blender/Godot 검증 worker는 아직 미완성입니다.

- **제작 도구는 직접 설정:** 로컬 Codex 실행기에 모델 서비스가 포함되지는 않습니다. 지원 기능, 로그인, 요금과 제출 승인은 각 도구에 따라 다릅니다. 도구가 없거나 승인이 필요하면 구체적인 실패 이유를 남기며 임시 에셋으로 성공을 꾸미지 않습니다. 로컬 작업을 중지해도 이미 제출된 원격 작업은 취소되지 않습니다.

- **미리 보기는 검증이 아닙니다:** GLB 구조와 파일 해시 검사만으로 메시 품질, Blend 편집 가능 여부, 골격 변형 또는 엔진 호환성을 확인할 수 없습니다. 내보내기 매니페스트의 엔진 검증은 `not_verified`로 유지됩니다.

- **고정된 버전은 유지:** 새 버전이 캔버스나 검토에서 채택한 버전을 대체하지 않습니다. SVG와 외부 파일에 의존하는 glTF 가져오기는 지원하지 않습니다. 실행 취소와 다시 실행은 현재 레이아웃에만 적용됩니다.

- **호스트별 차이:** MCP App 진입점은 등록되었으며 샌드박스 검증을 통과했습니다. 네이티브 Codex에서 표시, 캐시와 새로 고침 동작은 호스트에 따라 달라집니다.

<a id="contributing"></a>

## 개발 참여

개발 명령, 검사, 빌드 식별자와 런타임 동기화는 [CONTRIBUTING.md](CONTRIBUTING.md)를 참고하세요. 문제를 보고할 때 표시된 빌드 식별자, 호스트, 재현 단계와 기대 동작을 포함하고 인증 정보와 비공개 프로젝트 파일은 제외하세요.

<a id="license"></a>

## 라이선스와 출처

이 프로젝트는 [MIT 라이선스](LICENSE)를 따릅니다. 의존성 라이선스와 오리지널 예제 출처는 [서드파티 고지](licenses/THIRD_PARTY.md)에 기록되어 있습니다.

README 구성은 [Tripo Studio for Codex](https://github.com/GrinZero/tripo-studio-plugin)를 참고했습니다. 이곳의 배너와 예제 화면은 본 프로젝트의 자료입니다.
