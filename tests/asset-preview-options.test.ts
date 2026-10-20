import assert from "node:assert/strict";
import { test } from "node:test";
import type { AssetRevision } from "../src/shared/types";
import { assetPreviewOptions } from "../src/ui/asset-preview-options";

const revision = (mimes: string[]) =>
  ({ files: mimes.map((mime) => ({ mime })) }) as AssetRevision;

test("image bundles expose MP4 playback ahead of their static cover", () => {
  assert.deepEqual(
    assetPreviewOptions(
      "image",
      revision(["image/png", "video/mp4", "application/octet-stream"]),
    ),
    [
      { kind: "video", label: "视频", index: 1 },
      { kind: "image", label: "图片", index: 0 },
    ],
  );
});
test("model bundles retain their GLB preview and expose attached video", () => {
  assert.deepEqual(
    assetPreviewOptions(
      "model",
      revision(["model/gltf-binary", "image/png", "video/mp4"]),
    ).map(({ kind, index }) => ({ kind, index })),
    [
      { kind: "model", index: 0 },
      { kind: "video", index: 2 },
      { kind: "image", index: 1 },
    ],
  );
});
test("media uses its matching MIME regardless of file order", () => {
  assert.equal(
    assetPreviewOptions(
      "video",
      revision(["text/plain", "image/png", "video/webm"]),
    )[0].index,
    2,
  );
  assert.equal(
    assetPreviewOptions("audio", revision(["text/plain", "audio/mpeg"]))[0]
      .index,
    1,
  );
  assert.deepEqual(assetPreviewOptions("video", revision(["text/plain"])), []);
});
