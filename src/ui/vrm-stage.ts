import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import {
  VRMLoaderPlugin,
  VRMUtils,
  type VRM,
  type VRMHumanBoneName,
} from "@pixiv/three-vrm";
import {
  createVRMAnimationClip,
  VRMAnimationLoaderPlugin,
  type VRMAnimation,
} from "@pixiv/three-vrm-animation";
import { sampleTestPose, TEST_DURATION } from "./vrm-test-motion";

export type StageSettings = {
  motion: string;
  physics: boolean;
  bones: boolean;
  wireframe: boolean;
};
export type StageStatus = {
  phase: "loading" | "ready" | "error";
  message: string;
  springJoints: number;
  duration: number;
};
export function createVrmStage(
  host: HTMLDivElement,
  src: string,
  onStatus: (s: StageStatus) => void,
  onTime: (t: number) => void,
) {
  const scene = new THREE.Scene();
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  host.append(renderer.domElement);
  renderer.domElement.setAttribute("aria-label", "VRM 实时动作画面");
  const camera = new THREE.PerspectiveCamera(32, 1, 0.01, 1000);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.maxPolarAngle = Math.PI * 0.92;
  scene.add(new THREE.HemisphereLight(0xffffff, 0x8792a5, 2.4));
  const light = new THREE.DirectionalLight(0xffffff, 2.8);
  light.position.set(2, 4, 3);
  scene.add(light);
  const grid = new THREE.GridHelper(10, 40, 0x657080, 0x394250);
  scene.add(grid);
  let disposed = false;
  let vrm: VRM | undefined;
  let skeleton: THREE.SkeletonHelper | undefined;
  let mixer: THREE.AnimationMixer | undefined;
  let action: THREE.AnimationAction | undefined;
  let customClip: THREE.AnimationClip | undefined;
  let duration = TEST_DURATION;
  let time = 0;
  let playing = true;
  let settings: StageSettings = {
    motion: "walk_cycle_90f.vrma",
    physics: true,
    bones: false,
    wireframe: false,
  };
  let height = 1.7;
  let center = new THREE.Vector3(0, 0.85, 0);
  let rootY = 0;
  let rootRotation = 0;
  let loadId = 0;
  const wireMaterials: (THREE.Material & { wireframe: boolean })[] = [];
  const resetView = () => {
    controls.target.copy(center);
    camera.position.set(
      center.x,
      center.y + height * 0.05,
      center.z + height * 2.1,
    );
    controls.minDistance = height * 0.35;
    controls.maxDistance = height * 8;
    camera.near = height / 100;
    camera.far = height * 100;
    camera.updateProjectionMatrix();
    controls.update();
  };
  const status = (phase: StageStatus["phase"], message = "") => {
    if (disposed) return;
    host.dataset.phase = phase;
    onStatus({
      phase,
      message,
      springJoints: vrm?.springBoneManager?.joints.size || 0,
      duration,
    });
  };
  const resize = new ResizeObserver(() => {
    const { width, height } = host.getBoundingClientRect();
    if (!width || !height) return;
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  });
  resize.observe(host);
  status("loading");
  const loader = new GLTFLoader();
  loader.register((parser) => new VRMLoaderPlugin(parser));
  loader
    .loadAsync(src)
    .then((gltf) => {
      if (disposed) {
        VRMUtils.deepDispose(gltf.scene);
        return;
      }
      vrm = gltf.userData.vrm as VRM | undefined;
      if (!vrm) {
        VRMUtils.deepDispose(gltf.scene);
        throw new Error("此文件没有可用的 VRM 人形骨架，无法执行动作测试");
      }
      VRMUtils.rotateVRM0(vrm);
      scene.add(vrm.scene);
      vrm.scene.updateMatrixWorld(true);
      const bounds = new THREE.Box3().setFromObject(vrm.scene);
      height = Math.max(0.1, bounds.max.y - bounds.min.y);
      center = bounds.getCenter(new THREE.Vector3());
      grid.position.y = bounds.min.y;
      grid.scale.setScalar(height / 1.7);
      rootY = vrm.scene.position.y;
      rootRotation = vrm.scene.rotation.y;
      skeleton = new THREE.SkeletonHelper(vrm.scene);
      skeleton.visible = settings.bones;
      (skeleton.material as THREE.LineBasicMaterial).depthTest = false;
      skeleton.renderOrder = 10;
      scene.add(skeleton);
      vrm.scene.traverse((object) => {
        if (!(object instanceof THREE.Mesh)) return;
        const materials = Array.isArray(object.material)
          ? object.material
          : [object.material];
        for (const material of materials) {
          if ("wireframe" in material)
            wireMaterials.push(
              material as THREE.Material & { wireframe: boolean },
            );
        }
      });
      for (const material of wireMaterials)
        material.wireframe = settings.wireframe;
      // Gate the instance's SpringBone solver while preserving all other VRM updates.
      const spring = vrm.springBoneManager;
      if (spring) {
        const update = spring.update.bind(spring);
        spring.update = (delta) => {
          if (settings.physics) update(delta);
        };
      }
      resetView();
      status("ready");
    })
    .catch((error) =>
      status(
        "error",
        error instanceof Error ? error.message : "VRM 模型加载失败",
      ),
    );

  let previous = performance.now();
  let lastReport = 0;
  function render(now: number) {
    if (disposed) return;
    const delta = Math.min((now - previous) / 1000, 0.1);
    previous = now;
    if (vrm) {
      if (playing) time = (time + delta) % duration;
      if (mixer && action) mixer.setTime(time);
      else {
        vrm.humanoid.resetNormalizedPose();
        const pose = sampleTestPose(settings.motion, time);
        for (const [bone, rotation] of Object.entries(pose.rotations)) {
          vrm.humanoid
            .getNormalizedBoneNode(bone as VRMHumanBoneName)
            ?.rotation.set(...rotation);
        }
        vrm.scene.position.y = rootY + pose.lift * height;
        vrm.scene.rotation.y = rootRotation + pose.turn;
      }
      // Keep playback in wall-clock time while giving the spring solver bounded steps.
      if (playing) {
        const steps = Math.max(1, Math.ceil(delta * 60));
        for (let step = 0; step < steps; step++) vrm.update(delta / steps);
      } else vrm.update(0);
      if (now - lastReport > 100) {
        onTime(time);
        host.dataset.time = time.toFixed(3);
        lastReport = now;
      }
    }
    controls.update();
    renderer.render(scene, camera);
  }
  renderer.setAnimationLoop(render);

  return {
    setSettings(next: StageSettings) {
      if (settings.physics !== next.physics) vrm?.springBoneManager?.reset();
      if (
        next.motion !== "__custom" &&
        (settings.motion !== next.motion || customClip)
      ) {
        ++loadId;
        mixer?.stopAllAction();
        if (vrm && customClip) mixer?.uncacheClip(customClip);
        mixer = undefined;
        action = undefined;
        customClip = undefined;
        time = 0;
        duration = TEST_DURATION;
        if (vrm) {
          vrm.scene.position.y = rootY;
          vrm.scene.rotation.y = rootRotation;
          vrm.humanoid.resetNormalizedPose();
          vrm.springBoneManager?.reset();
          status("ready");
        }
      }
      settings = next;
      if (skeleton) skeleton.visible = next.bones;
      for (const material of wireMaterials) material.wireframe = next.wireframe;
    },
    setPlaying(value: boolean) {
      playing = value;
    },
    seek(value: number) {
      time = Math.max(0, Math.min(duration - 0.001, value));
      onTime(time);
      vrm?.springBoneManager?.reset();
    },
    resetView,
    async loadVrma(file: File) {
      if (!vrm) throw new Error("请等待 VRM 模型加载完成");
      const id = ++loadId;
      const animationLoader = new GLTFLoader();
      animationLoader.register(
        (parser) => new VRMAnimationLoaderPlugin(parser),
      );
      const gltf = await animationLoader.parseAsync(
        await file.arrayBuffer(),
        "",
      );
      if (disposed || id !== loadId) return false;
      const animation = (
        gltf.userData.vrmAnimations as VRMAnimation[] | undefined
      )?.[0];
      if (!animation || animation.duration <= 0)
        throw new Error("此文件不包含可播放的 VRMA 人形动作");
      mixer?.stopAllAction();
      if (customClip) mixer?.uncacheClip(customClip);
      customClip = createVRMAnimationClip(animation, vrm);
      vrm.humanoid.resetNormalizedPose();
      vrm.scene.position.y = rootY;
      vrm.scene.rotation.y = rootRotation;
      mixer = new THREE.AnimationMixer(vrm.scene);
      action = mixer.clipAction(customClip);
      action.play();
      duration = customClip.duration;
      time = 0;
      vrm.springBoneManager?.reset();
      status("ready");
      return true;
    },
    dispose() {
      disposed = true;
      ++loadId;
      renderer.setAnimationLoop(null);
      resize.disconnect();
      controls.dispose();
      mixer?.stopAllAction();
      if (vrm) {
        mixer?.uncacheRoot(vrm.scene);
        VRMUtils.deepDispose(vrm.scene);
      }
      skeleton?.dispose();
      grid.geometry.dispose();
      for (const material of Array.isArray(grid.material)
        ? grid.material
        : [grid.material])
        material.dispose();
      renderer.dispose();
      renderer.forceContextLoss();
      renderer.domElement.remove();
    },
  };
}
export type VrmStage = ReturnType<typeof createVrmStage>;
