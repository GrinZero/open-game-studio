import type { AssetKind, AssetRevision } from "../shared/types";

export interface PreviewOption {
  kind: AssetKind;
  index: number;
  label: string;
}

// Resource bundles can contain playable media even when classified as images
// or models. Select the actual file rather than assuming file zero is playable.
export function assetPreviewOptions(
  kind: AssetKind,
  revision: AssetRevision,
): PreviewOption[] {
  const options: PreviewOption[] = [];
  const add = (
    kind: AssetKind,
    label: string,
    matches: (mime: string) => boolean,
  ) => {
    const index = revision.files.findIndex((file) => matches(file.mime));
    if (index >= 0) options.push({ kind, label, index });
  };
  if (kind === "model")
    add("model", "3D 模型", (mime) => mime === "model/gltf-binary");
  if (kind === "document")
    add("document", "文档", (mime) => mime === "text/plain");
  add("video", "视频", (mime) => mime.startsWith("video/"));
  if (kind === "audio")
    add("audio", "音频", (mime) => mime.startsWith("audio/"));
  add("image", "图片", (mime) => mime.startsWith("image/"));
  return options;
}
