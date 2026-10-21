/** Built-in diagnostic poses, sampled on the normalized VRM humanoid (radians). */
export type TestMotion =
  | "walk_cycle_90f.vrma"
  | "idle_breathe.vrma"
  | "turn_180_skirt_test.vrma";
export const TEST_MOTIONS: Record<TestMotion, string> = {
  "walk_cycle_90f.vrma": "标准步行测试",
  "idle_breathe.vrma": "待机呼吸测试",
  "turn_180_skirt_test.vrma": "转身裙摆测试",
};
export const TEST_DURATION = 3;
export type TestPose = {
  rotations: Record<string, [number, number, number]>;
  lift: number;
  turn: number;
};

export function sampleTestPose(motion: string, time: number): TestPose {
  const phase = (time / TEST_DURATION) * Math.PI * 2;
  const stride = Math.sin(phase * 2);
  const walking = motion !== "idle_breathe.vrma";
  const swing = walking ? stride : 0;
  const rotations: TestPose["rotations"] = {
    leftUpperArm: [0.32 * swing, 0, -1.25],
    rightUpperArm: [-0.32 * swing, 0, 1.25],
    leftLowerArm: [-0.15 - Math.max(0, swing) * 0.2, 0, 0],
    rightLowerArm: [-0.15 - Math.max(0, -swing) * 0.2, 0, 0],
    leftUpperLeg: [-0.42 * swing, 0, 0],
    rightUpperLeg: [0.42 * swing, 0, 0],
    leftLowerLeg: [Math.max(0, swing) * 0.7, 0, 0],
    rightLowerLeg: [Math.max(0, -swing) * 0.7, 0, 0],
    leftFoot: [-Math.max(0, swing) * 0.25, 0, 0],
    rightFoot: [-Math.max(0, -swing) * 0.25, 0, 0],
    spine: [0.02 * Math.sin(phase), 0.035 * swing, 0],
    chest: [0.015 * Math.sin(phase), -0.04 * swing, 0],
    hips: [0, 0.04 * swing, walking ? 0.025 * stride : 0],
  };
  return {
    rotations,
    lift: walking ? 0.015 * (1 - Math.cos(phase * 4)) : 0.004 * Math.sin(phase),
    // Turn out and back so the test loops without an orientation jump.
    turn:
      motion === "turn_180_skirt_test.vrma"
        ? (Math.PI * (1 - Math.cos(phase))) / 2
        : 0,
  };
}
