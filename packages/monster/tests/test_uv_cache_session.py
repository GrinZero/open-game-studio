import importlib.util
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("uv_cache_session", ROOT / "scripts/uv_cache_session.py")
cache_session = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache_session)


class UVCacheSessionTests(unittest.TestCase):
    def test_roundtrip_preserves_portable_links_and_skips_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, store = root / "local/uv", root / "drive/uv_snapshots/test"
            wheel = cache / "archive-v0/package/module.py"
            wheel.parent.mkdir(parents=True)
            wheel.write_text("cached dependency\n")
            (cache / "wheel").symlink_to(wheel.parent)  # absolute uv cache link becomes portable
            (cache / ".tmp-incomplete").mkdir()
            (cache / ".tmp-incomplete/part").write_text("partial download")
            (cache / ".lock").touch()
            first = cache_session.save(cache, store)
            self.assertEqual(first["status"], "saved")
            self.assertEqual(cache_session.save(cache, store)["status"], "unchanged")
            restored = root / "second_vm/uv"
            restored.parent.mkdir()
            self.assertEqual(cache_session.restore(restored, store)["status"], "restored")
            self.assertEqual((restored / "wheel/module.py").read_text(), "cached dependency\n")
            self.assertFalse((restored / ".tmp-incomplete").exists())
            self.assertFalse((restored / ".lock").exists())
            self.assertEqual(cache_session.save(restored, store)["status"], "unchanged")

    def test_corrupt_new_snapshot_falls_back_without_replacing_live_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, store = root / "local/uv", root / "drive/uv_snapshots/test"
            cache.mkdir(parents=True)
            (cache / "dependency").write_text("first good version")
            cache_session.save(cache, store)
            (cache / "dependency").write_text("second version with changed content")
            latest = cache_session.save(cache, store)
            (store / latest["archive"]).write_bytes(b"interrupted upload")
            restored = root / "new_vm/uv"
            restored.parent.mkdir()
            self.assertEqual(cache_session.restore(restored, store)["status"], "restored")
            self.assertEqual((restored / "dependency").read_text(), "first good version")
            (restored / "dependency").write_text("active local changes")
            self.assertEqual(cache_session.restore(restored, store)["status"], "local_cache_reused")
            self.assertEqual((restored / "dependency").read_text(), "active local changes")

    def test_archive_escape_is_rejected_even_with_matching_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = root / "drive/uv_snapshots/test"
            store.mkdir(parents=True)
            archive = store / "malicious.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                info = tarfile.TarInfo("uv/../../escape")
                info.size = 1
                tar.addfile(info, io.BytesIO(b"x"))
            (store / "latest.json").write_text(json.dumps({"archive": archive.name,
                "bytes": archive.stat().st_size, "sha256": cache_session.digest(archive),
                "fingerprint": "unused"}))
            local = root / "local/uv"
            local.parent.mkdir()
            self.assertEqual(cache_session.restore(local, store)["status"], "cache_miss")
            self.assertFalse((root / "escape").exists())


if __name__ == "__main__":
    unittest.main()
