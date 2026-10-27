"""Renderer ABI isolation and Cell 2 version migration without starting a GPU."""
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'worker'))
import blender_runtime as runtime


class BlenderRuntimeTests(unittest.TestCase):
    def test_pinned_renderer_and_python_match_manifest(self):
        requirements=(ROOT/'worker/blender-requirements.txt').read_text()
        self.assertIn('bpy=='+runtime.BPY_VERSION,requirements)
        self.assertEqual(runtime.PYTHON_VERSION,'3.13')
        self.assertNotIn('/content/skintokens-service/venv/bin/python',runtime.bpy_python_candidates())

    def test_explicit_interpreter_takes_precedence_and_missing_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            binary=Path(temporary)/'python'; binary.touch()
            with patch.dict(os.environ,{'MONSTER_BPY_PYTHON':str(binary)}):
                self.assertEqual(runtime.resolve_bpy_python(),str(binary))
            with patch.dict(os.environ,{'MONSTER_BPY_PYTHON':str(binary)+'missing'}):
                with self.assertRaises(FileNotFoundError):runtime.resolve_bpy_python()

    def test_ready_marker_version_is_checked_and_renderer_always_prepared(self):
        spec=importlib.util.spec_from_file_location('colab_runtime_version_test',ROOT/'scripts/colab_runtime.py')
        colab=importlib.util.module_from_spec(spec);spec.loader.exec_module(colab)
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ,{'SKINTOKENS_ROOT':temporary}), \
             patch.object(colab,'stop_previous_worker'), patch.object(colab.subprocess,'run') as run:
            ready=Path(temporary)/'ready.json'; ready.write_text('{}')
            colab.setup_skintokens()
            self.assertEqual([Path(c.args[0][1]).name for c in run.call_args_list],
                             ['bootstrap-skintokens.sh','bootstrap-blender.sh'])
            run.reset_mock()
            ready.write_text(json.dumps({'bpy':runtime.CONTRACT['skintokens_bpy_version']}))
            colab.setup_skintokens()
            self.assertEqual(Path(run.call_args.args[0][1]).name,'bootstrap-blender.sh')
            self.assertEqual(run.call_count,1)


if __name__=='__main__':unittest.main()
