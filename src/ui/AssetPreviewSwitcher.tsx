import { useState, type ReactNode } from "react";
import type { Asset, AssetRevision } from "../shared/types";
import {
  assetPreviewOptions,
  type PreviewOption,
} from "./asset-preview-options";

export function AssetPreviewSwitcher({
  asset,
  revision,
  children,
}: {
  asset: Asset;
  revision: AssetRevision;
  children: (option: PreviewOption | undefined) => ReactNode;
}) {
  const options = assetPreviewOptions(asset.kind, revision);
  const [choice, setChoice] = useState<{ revisionId: string; index: number }>();
  const selected =
    options.find(
      (option) =>
        choice?.revisionId === revision.id && option.index === choice.index,
    ) || options[0];
  return (
    <>
      {options.length > 1 && (
        <div
          className="asset-preview-switcher nodrag nowheel nopan"
          role="group"
          aria-label={`${asset.name}预览方式`}
          onPointerDown={(event) => event.stopPropagation()}
          onClick={(event) => event.stopPropagation()}
          onDoubleClick={(event) => event.stopPropagation()}
        >
          {options.map((option) => (
            <button
              key={option.index}
              type="button"
              aria-pressed={selected?.index === option.index}
              onClick={() =>
                setChoice({ revisionId: revision.id, index: option.index })
              }
            >
              {option.label}
            </button>
          ))}
        </div>
      )}
      {children(selected)}
    </>
  );
}
