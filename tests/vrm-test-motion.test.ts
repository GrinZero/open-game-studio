import assert from "node:assert/strict";
import test from "node:test";
import {
  sampleTestPose,
  TEST_DURATION,
  TEST_MOTIONS,
} from "../src/ui/vrm-test-motion";

test("standard walk drives alternating limbs and loops at 90 frames / 30 fps", () => {
  const pose = sampleTestPose("walk_cycle_90f.vrma", TEST_DURATION / 8);
  assert.ok(Math.abs(pose.rotations.leftUpperLeg[0]) > 0.3);
  assert.equal(
    pose.rotations.leftUpperLeg[0],
    -pose.rotations.rightUpperLeg[0],
  );
  assert.ok(pose.rotations.leftLowerLeg[0] > 0.6);
  assert.equal(pose.rotations.rightLowerLeg[0], 0);
  const opposite = sampleTestPose(
    "walk_cycle_90f.vrma",
    (TEST_DURATION * 3) / 8,
  );
  assert.ok(opposite.rotations.rightLowerLeg[0] > 0.6);
  for (const motion of Object.keys(TEST_MOTIONS)) {
    const start = sampleTestPose(motion, 0);
    const end = sampleTestPose(motion, TEST_DURATION);
    for (const name of Object.keys(start.rotations)) {
      start.rotations[name].forEach((v, i) =>
        assert.ok(Math.abs(v - end.rotations[name][i]) < 1e-10),
      );
    }
    assert.ok(Math.abs(start.turn - end.turn) < 1e-10);
    assert.ok(Math.abs(start.lift - end.lift) < 1e-10);
  }
});

test("idle leaves legs at rest; turn reaches 180 degrees and returns", () => {
  const idle = sampleTestPose("idle_breathe.vrma", TEST_DURATION / 4);
  assert.equal(Math.abs(idle.rotations.leftUpperLeg[0]), 0);
  assert.equal(Math.abs(idle.rotations.rightUpperLeg[0]), 0);
  assert.ok(idle.lift > 0);
  assert.equal(
    sampleTestPose("turn_180_skirt_test.vrma", TEST_DURATION / 2).turn,
    Math.PI,
  );
});
