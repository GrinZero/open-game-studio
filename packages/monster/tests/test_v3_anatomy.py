"""Anatomical meaning through anonymous classification, not just valid name syntax."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "worker"))
from bone_taxonomy import (
    classify_rig_bones,
    parse_canonical_bone_name,
    default_limits_and_corr_for_kind,
)


def paired_case(wing=False, common_digit_heads=False, legacy=False, distal_only=False):
    meta = {}
    logical = {}

    def add(name, parent, head, tail, kind="limb", side="C", **fields):
        meta[name] = dict(
            parent=parent,
            rest_head_world=list(head),
            rest_tail_world=list(tail),
            rest_length=float(np.linalg.norm(np.array(tail) - head)),
            kind=kind,
            side=side,
            **fields,
        )
        return name

    add("Root", None, [0, 0, 0.8], [0, 0, 1], "root")
    add("Spine1", "Root", [0, 0, 1], [0, 0, 1.4], "spine")
    add("Head", "Spine1", [0, 0, 1.4], [0, -0.1, 2], "head")
    for side, sign in (("L", 1), ("R", -1)):
        for family in ("leg", "wing" if wing else "arm"):
            if family == "leg":
                points = [
                    [sign * 0.22, 0, 0.8],
                    [sign * 0.3, 0.08, 0.45],
                    [sign * 0.3, 0, 0.1],
                    [sign * 0.3, -0.12, 0.03],
                    [sign * 0.3, -0.27, 0.02],
                ]
                parent = "Root"
            else:
                points = [
                    [sign * 0.1, 0, 1.4],
                    [sign * 0.28, 0, 1.4],
                    [sign * 0.55, 0, 1.35],
                    [sign * 0.85, 0, 1.25],
                    [sign * 1.0, 0, 1.2],
                ]
                parent = "Spine1"
            for j, (head, tail) in enumerate(zip(points, points[1:]), 1):
                name = (
                    f"LIMB_{side}_{1 if family == 'leg' else 2:02d}_{j:02d}"
                    if legacy
                    else f"bone_{len(meta):03d}"
                )
                marked = family == "wing" and (not distal_only or j == 4)
                parent = add(
                    name,
                    parent,
                    head,
                    tail,
                    "wing" if marked else "limb",
                    side,
                    **(
                        {"flatness": 6.0, "winged": True, "role": "wing"}
                        if marked
                        else {}
                    ),
                )
                logical[(family, side, j)] = parent
            hub = parent
            # Deliberately create outer before inner to prevent ID order standing in for geometry.
            for finger in (3, 1, 2):
                base = np.array(points[-1], float)
                base[0] += sign * 0.02 * (finger - 2)
                head = np.array(points[-1], float) if common_digit_heads else base
                for j in (1, 2):
                    tail = base + [0, -0.04 * j, 0]
                    name = f"branch_{len(meta):03d}"
                    parent = add(
                        name, hub if j == 1 else parent, head, tail, "digit", side
                    )
                    logical[(family + "_digit", side, finger, j)] = parent
                    head = tail
    return meta, logical


def classify(meta):
    return classify_rig_bones(meta, [-1.2, -0.5, 0], [1.2, 0.5, 2])


class AnatomicalMeaningTests(unittest.TestCase):
    def test_finger_and_toe_labels_match_mirrored_geometry_and_increase_outward(self):
        for common_heads in (False, True):
            with self.subTest(common_digit_heads=common_heads):
                meta, logical = paired_case(common_digit_heads=common_heads)
                before = copy.deepcopy(meta)
                actual = classify(meta)
                self.assertEqual(meta, before)
                for family, prefix in (("arm", "Finger"), ("leg", "Toe")):
                    indices = []
                    for finger in (1, 2, 3):
                        for joint in (1, 2):
                            left = actual[
                                logical[(family + "_digit", "L", finger, joint)]
                            ]["canonical_name"]
                            right = actual[
                                logical[(family + "_digit", "R", finger, joint)]
                            ]["canonical_name"]
                            self.assertTrue(left.startswith(prefix))
                            self.assertEqual(left[:-2] + ".R", right)
                        indices.append(int(left.split("_")[0][len(prefix) :]))
                    self.assertLess(indices[0], indices[1])
                    self.assertLess(indices[1], indices[2])
                reversed_meta = dict(reversed(list(meta.items())))
                self.assertEqual(
                    {n: b["canonical_name"] for n, b in actual.items()},
                    {
                        n: b["canonical_name"]
                        for n, b in classify(reversed_meta).items()
                    },
                )

    def test_marked_wing_survives_digit_hub_and_propagates_to_proximal_chain(self):
        for legacy in (False, True):
            for distal_only in (False, True):
                with self.subTest(legacy=legacy, distal_only=distal_only):
                    meta, logical = paired_case(
                        wing=True, legacy=legacy, distal_only=distal_only
                    )
                    actual = classify(meta)
                    for side in ("L", "R"):
                        for j, role in enumerate(
                            ("WingShoulder", "WingUpper", "WingForearm", "WingHand"), 1
                        ):
                            b = actual[logical[("wing", side, j)]]
                            self.assertEqual(b["kind"], "wing")
                            self.assertEqual(b["canonical_name"], role + "." + side)
                        for finger in (1, 2, 3):
                            for joint in (1, 2):
                                self.assertEqual(
                                    actual[
                                        logical[("wing_digit", side, finger, joint)]
                                    ]["kind"],
                                    "digit",
                                )
                    self.assertEqual(actual["Spine1"]["kind"], "spine")
                    self.assertEqual(actual["Root"]["kind"], "root")
                    self.assertTrue(
                        all(
                            b["kind"] == "limb"
                            for n, b in actual.items()
                            if n
                            in [
                                logical[("leg", side, j)]
                                for side in ("L", "R")
                                for j in (1, 2, 3, 4)
                            ]
                        )
                    )

    def test_known_v3_names_and_legacy_readonly_parsing_remain_compatible(self):
        meta, _ = paired_case(wing=True)
        initial = classify(meta)
        renames = {old: b["canonical_name"] for old, b in initial.items()}
        canonical = {
            renames[old]: {**b, "parent": renames.get(b.get("parent"))}
            for old, b in initial.items()
        }
        self.assertTrue(
            all(b["canonical_name"] == name for name, b in classify(canonical).items())
        )
        for name, kind, side in [
            ("LIMB_L_02_03", "limb", "L"),
            ("DIGIT_R_21_02", "digit", "R"),
            ("WING_R_02_01", "wing", "R"),
        ]:
            parsed = parse_canonical_bone_name(name)
            self.assertEqual((parsed["kind"], parsed["side"]), (kind, side))

    def test_flatness_is_strict_and_v3_identity_has_priority_over_stale_hints(self):
        for value, expected in ((5.0, "limb"), (5.0001, "wing"), (None, "limb")):
            with self.subTest(flatness=value):
                meta, logical = paired_case(wing=True, distal_only=True)
                tip = logical[("wing", "L", 4)]
                for side in ("L", "R"):
                    for j in (1, 2, 3, 4):
                        b = meta[logical[("wing", side, j)]]
                        b["kind"] = "limb"
                        b.pop("role", None)
                        b.pop("winged", None)
                        b.pop("flatness", None)
                meta[tip]["flatness"] = value
                actual = classify(meta)
                self.assertEqual(actual[tip]["kind"], expected)
        meta, logical = paired_case()
        initial = classify(meta)
        renames = {old: b["canonical_name"] for old, b in initial.items()}
        canonical = {
            renames[old]: {**b, "parent": renames.get(b.get("parent"))}
            for old, b in initial.items()
        }
        for side in ("L", "R"):
            canonical["Hand." + side].update(kind="wing", flatness=9.0, role="wing")
        actual = classify(canonical)
        for side in ("L", "R"):
            self.assertEqual(actual["Hand." + side]["kind"], "limb")
            self.assertEqual(actual["Hand." + side]["canonical_name"], "Hand." + side)

    def test_wing_reclassification_refreshes_defaults_but_preserves_explicit_limits(
        self,
    ):
        meta, logical = paired_case()
        initial = classify(meta)
        for side in ("L", "R"):
            tip = logical[("arm", side, 4)]
            initial[tip].update(kind="wing", flatness=6.0)
        custom = logical[("arm", "R", 4)]
        initial[custom].update(
            limits_rad=[[-0.1, 0.2], [-0.2, 0.3], [-0.3, 0.4]],
            limits_source="user_config",
            chain_corr=0.123,
        )
        actual = classify(initial)
        expected, correlation, _ = default_limits_and_corr_for_kind("wing", "hand_hub")
        left = logical[("arm", "L", 4)]
        np.testing.assert_allclose(
            actual[left]["limits_rad"], np.round(expected, 4), atol=1e-12
        )
        self.assertEqual(actual[left]["chain_corr"], correlation)
        self.assertEqual(actual[custom]["limits_rad"], initial[custom]["limits_rad"])
        self.assertEqual(actual[custom]["chain_corr"], 0.123)

    def test_v2_wing_group_and_joint_indices_stay_readable(self):
        meta, logical = paired_case(wing=True, legacy=True)
        rename = {
            old: old.replace("LIMB_", "WING_") if key[0] == "wing" else old
            for key, old in logical.items()
        }
        remapped = {
            rename.get(old, old): {
                **info,
                "parent": rename.get(info.get("parent"), info.get("parent")),
            }
            for old, info in meta.items()
        }
        actual = classify(remapped)
        for side in ("L", "R"):
            for j in (1, 2, 3, 4):
                value = actual[f"WING_{side}_02_{j:02d}"]
                self.assertEqual(value["kind"], "wing")
                self.assertEqual(value["group_id"], "02")
                self.assertEqual(value["chain_index"], j)


class NativeAnatomyTests(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("MONSTER_TEST_BLENDER"),
        "Set MONSTER_TEST_BLENDER for actual candidate naming",
    )
    def test_candidate_handoff_preserves_wings_weights_pose_and_digit_pairs(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                [
                    os.environ["MONSTER_TEST_BLENDER"],
                    "--background",
                    "--factory-startup",
                    "--python-exit-code",
                    "1",
                    "--python",
                    str(Path(__file__).with_name("fixture_v3_anatomy.py")),
                    "--",
                    temporary,
                ],
                capture_output=True,
                text=True,
                timeout=180,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("ANATOMY_NATIVE_PASSED", result.stdout)
            data = json.loads((Path(temporary) / "native-anatomy.json").read_text())
            self.assertEqual(data["wing_bones"], 8)
            self.assertLess(data["surface_error"], 2e-6)


if __name__ == "__main__":
    unittest.main()
