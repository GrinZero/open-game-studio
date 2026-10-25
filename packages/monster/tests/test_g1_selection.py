"""Reviewed local repairs must propagate into probes, pose rendering and U2 payloads."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'worker'))
sys.path.insert(0, str(ROOT / 'skills/monster-workflow/scripts'))
from sop_gate import resolve_g1_candidate_files
import sop_g3_exam as exam


class G1SelectionTests(unittest.TestCase):
    def fixture(self, folder):
        work = Path(folder) / 'work'
        old = work / 'g1_candidates'
        old.mkdir(parents=True)
        (old / 'candidate_0_full.blend').write_bytes(b'old-model')
        (old / 'candidate_0_rig_meta.json').write_text('{}')
        repaired = work / 'local_repair.blend'
        repaired.write_bytes(b'repaired-model')
        meta = work / 'local_repair_meta.json'
        meta.write_text(json.dumps({'revision': 'repaired'}))
        info = {'selected_candidate_id': '0', 'materials': {
            'selected_candidate_id': '0', 'selected_blend_path': str(repaired),
            'rig_meta_path': str(meta)}}
        return work, repaired, meta, info

    def test_explicit_repair_wins_over_historical_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            work, repaired, meta, info = self.fixture(folder)
            self.assertEqual(resolve_g1_candidate_files(work, info), ('0', meta, repaired))

    def test_deleted_explicit_revision_cannot_silently_reuse_old_model(self):
        with tempfile.TemporaryDirectory() as folder:
            work, repaired, meta, info = self.fixture(folder)
            for target in (repaired, meta):
                original = target.read_bytes()
                target.unlink()
                with self.assertRaises(FileNotFoundError):
                    resolve_g1_candidate_files(work, info)
                target.write_bytes(original)

    def test_candidate_array_uses_selected_index_and_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as folder:
            work, repaired, meta, info = self.fixture(folder)
            del info['materials']['selected_blend_path']
            info['materials']['candidate_blends'] = [str(repaired)]
            self.assertEqual(resolve_g1_candidate_files(work, info)[2], repaired)
            info['selected_candidate_id'] = '1'
            with self.assertRaises(ValueError):
                resolve_g1_candidate_files(work, info)
            del info['materials']['selected_candidate_id']
            with self.assertRaises(ValueError):
                resolve_g1_candidate_files(work, info)

    def test_legacy_ledger_without_material_paths_still_resolves(self):
        with tempfile.TemporaryDirectory() as folder:
            work, _, _, _ = self.fixture(folder)
            cid, meta, blend = resolve_g1_candidate_files(work, {'selected_candidate_id': '0'})
            self.assertEqual(blend.read_bytes(), b'old-model')
            self.assertEqual(meta.parent, work / 'g1_candidates')

    def test_u2_payload_copies_reviewed_revision(self):
        with tempfile.TemporaryDirectory() as folder:
            work, repaired, meta, info = self.fixture(folder)
            drive = Path(folder) / 'drive'
            ledger = {'gates': {}, 'fingerprints': {}}
            with patch.object(exam, 'load_ledger', return_value=ledger), \
                 patch.object(exam, 'assert_gate', return_value=info), \
                 patch.object(exam, 'resolve_scratch_dirs', return_value={'work_dir': work}), \
                 patch.object(exam, 'save_ledger'):
                result = exam.sync_u2_to_drive('test', project_root=folder, drive_root=drive)
            self.assertEqual(Path(result['drive_blend']).read_bytes(), repaired.read_bytes())
            self.assertEqual(Path(result['drive_meta']).read_bytes(), meta.read_bytes())


if __name__ == '__main__':
    unittest.main()
