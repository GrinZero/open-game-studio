"""Backend/CLI client for the existing Monster Gradio API. No automatic resubmission."""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
DEFAULT_REGISTRY = REPO_ROOT / "var/monster/endpoint_registry.json"
PROTOCOL = json.loads((ROOT / "contracts/protocol.json").read_text())


def monster_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", value):
        raise ValueError("Monster ID must contain only letters, digits, _ or - (1–100 characters)")
    return value


def discover_registry_path(explicit=None, *, home=None):
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"Registry file not found: {path}")
        return path
    env_path = os.environ.get("MONSTER_REGISTRY")
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"Registry file not found: {path}")
        return path
    base_home = Path(home).expanduser() if home else Path.home()
    candidates = [
        DEFAULT_REGISTRY,
        ROOT / "var/endpoint_registry.json",
        *base_home.glob("Library/CloudStorage/GoogleDrive-*/MyDrive/MonsterStudio/runtime/endpoint_registry.json"),
        base_home / "Google Drive/My Drive/MonsterStudio/runtime/endpoint_registry.json",
        Path("/content/drive/MyDrive/MonsterStudio/runtime/endpoint_registry.json"),
    ]
    existing = [p.resolve() for p in candidates if p.is_file()]
    if not existing:
        raise ValueError(
            "未找到 endpoint_registry.json；请先在 Colab 启动 Worker（挂载 Google Drive 桌面版可自动发现），"
            "或运行 npm run monster:call -- connect '<配对JSON>'。"
        )
    return max(existing, key=lambda p: p.stat().st_mtime)


def parse_pairing_payload(raw):
    text = raw.strip()
    path = Path(text).expanduser()
    if not text.startswith("{") and path.is_file():
        text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Pairing payload must be a JSON object")
    return {
        **data,
        "username": data.get("username") or "monster",
    }


def save_registry(payload, destination=DEFAULT_REGISTRY, *, output_dir=None, **client_kwargs):
    data = parse_pairing_payload(payload) if isinstance(payload, str) else {**payload, "username": payload.get("username") or "monster"}
    client = MonsterClient(data, output_dir or (REPO_ROOT / "var/monster/downloads"), **client_kwargs)
    target = Path(destination).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        target.chmod(0o600)
    except OSError:
        pass
    return {
        "status": "CONNECTED",
        "url": data["url"],
        "username": data["username"],
        "protocol_version": PROTOCOL["protocol_version"],
        "registry": str(target),
        "output_dir": str(client.output_dir),
    }


class MonsterClient:
    def __init__(self, registry, output_dir, *, client_factory=None, file_handler=None):
        url = urlsplit(registry.get("url", ""))
        if (url.scheme not in {"http", "https"} or not url.hostname
                or url.username or url.password or url.query or url.fragment):
            raise ValueError("Invalid Worker URL")
        if url.scheme == "http" and url.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Remote Worker connections require HTTPS")
        if not registry.get("username") or not registry.get("token"):
            raise ValueError("Worker username and token are required")
        if client_factory is None:
            from gradio_client import Client, handle_file
            client_factory, file_handler = Client, handle_file
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._file = file_handler
        self._client = client_factory(
            registry["url"], auth=(registry["username"], registry["token"]),
            download_files=str(self.output_dir), verbose=False,
        )
        # Live handshake, not the potentially stale registry, establishes compatibility.
        status = self.heartbeat()
        if status.get("protocol_version") != PROTOCOL["protocol_version"]:
            raise ValueError("Incompatible Monster Worker; deploy the migrated release and reconnect")

    @classmethod
    def from_registry(cls, registry_path=None, output_dir=None, **kwargs):
        resolved = discover_registry_path(registry_path)
        return cls(
            json.loads(resolved.read_text(encoding="utf-8")),
            output_dir or (REPO_ROOT / "var/monster/downloads"),
            **kwargs,
        )

    def heartbeat(self):
        return json.loads(self._client.predict(api_name="/heartbeat"))

    def _upload(self, file, suffix):
        path = Path(file).resolve()
        if not path.is_file() or path.suffix.lower() != suffix:
            raise ValueError(f"Expected an existing {suffix} file")
        return self._file(str(path))

    def rig_candidates(self, identity, input_glb=None, run_auto_audit=True):
        identity = monster_id(identity)
        uploaded = self._upload(input_glb, ".glb") if input_glb else None
        return self._client.predict(
            identity, uploaded, run_auto_audit,
            api_name="/rig_candidates",
        )

    def train_monster_probe(self, identity, *, custom_blend=None, custom_meta_json=None,
                            num_samples=5000, epochs=15, auto_shutdown_after=False):
        identity = monster_id(identity)
        if type(num_samples) is not int or num_samples <= 0 or type(epochs) is not int or epochs <= 0:
            raise ValueError("num_samples and epochs must be positive integers")
        return self._client.predict(
            identity, self._upload(custom_blend, ".blend") if custom_blend else None,
            custom_meta_json, num_samples, epochs, auto_shutdown_after,
            api_name="/train_monster_probe",
        )

    def reexam_only(self, identity, *, rerender_exam=False, auto_shutdown_after=False):
        identity = monster_id(identity)
        return self._client.predict(
            identity, bool(rerender_exam), bool(auto_shutdown_after),
            api_name="/reexam_only",
        )

    def generate_actions_batch(self, identity, actions, *, bake_glb_on_cloud=True, auto_shutdown_after=False):
        identity = monster_id(identity)
        if not isinstance(actions, str) or not actions.strip():
            raise ValueError("At least one action is required")
        # Keep the runtime alive until callers have downloaded and adopted the bundle.
        return self._client.predict(identity, actions, bool(bake_glb_on_cloud), bool(auto_shutdown_after),
                                    api_name="/generate_actions_batch")

    def shutdown(self, reason="local_scheduler_explicit_shutdown"):
        return self._client.predict(reason, api_name="/shutdown")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=None, help="Optional endpoint_registry.json path (auto-discovered by default)")
    parser.add_argument("--output", default=str(REPO_ROOT / "var/monster/downloads"))
    sub = parser.add_subparsers(dest="operation", required=True)
    connect = sub.add_parser("connect", help="Save and verify Colab Worker connection JSON or file")
    connect.add_argument("payload", help="Pairing JSON string or path to endpoint_registry.json")
    sub.add_parser("status")
    rig = sub.add_parser("rig")
    rig.add_argument("monster_id")
    rig.add_argument("input_glb", nargs="?", default=None, help="Optional local GLB (omit to use U1 Drive-synced GLB)")
    train = sub.add_parser("train")
    train.add_argument("monster_id")
    train.add_argument("--samples", type=int, default=5000)
    train.add_argument("--epochs", type=int, default=15)
    reexam = sub.add_parser("reexam")
    reexam.add_argument("monster_id")
    reexam.add_argument("--rerender", action="store_true")
    animate = sub.add_parser("animate")
    animate.add_argument("monster_id")
    animate.add_argument("actions")
    animate.add_argument("--no-cloud-bake", action="store_true")
    sub.add_parser("shutdown")
    args = parser.parse_args()
    if args.operation == "connect":
        result = save_registry(
            args.payload,
            destination=args.registry or DEFAULT_REGISTRY,
            output_dir=args.output,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    client = MonsterClient.from_registry(args.registry, args.output)
    if args.operation == "status":
        result = client.heartbeat()
    elif args.operation == "rig":
        result = client.rig_candidates(args.monster_id, args.input_glb)
    elif args.operation == "train":
        result = client.train_monster_probe(args.monster_id, num_samples=args.samples, epochs=args.epochs)
    elif args.operation == "reexam":
        result = client.reexam_only(args.monster_id, rerender_exam=args.rerender)
    elif args.operation == "animate":
        result = client.generate_actions_batch(
            args.monster_id, args.actions, bake_glb_on_cloud=not args.no_cloud_bake
        )
    else:
        result = client.shutdown()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
