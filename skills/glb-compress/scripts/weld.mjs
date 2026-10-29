import { compactPrimitive } from "@gltf-transform/functions";

// Position tolerance in mesh-local units; every other attribute (including
// morph POSITION deltas, joints, weights, normals and UVs) must match exactly.
export function weldWithTolerance(document, tolerance) {
  for (const mesh of document.getRoot().listMeshes())
    for (const prim of mesh.listPrimitives()) {
      const position = prim.getAttribute("POSITION");
      if (!position || !position.getCount()) continue;
      const other = prim
        .listSemantics()
        .filter((s) => s !== "POSITION")
        .map((s) => prim.getAttribute(s));
      for (const target of prim.listTargets())
        other.push(...target.listAttributes());
      const count = position.getCount();
      const remap = new Uint32Array(count).fill(0xffffffff);
      const indices =
        prim.getIndices()?.getArray() ??
        Array.from({ length: count }, (_, i) => i);
      const grid = new Map();
      let next = 0;
      const a = [],
        b = [];
      function matches(i, j) {
        position.getElement(i, a);
        position.getElement(j, b);
        if (Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) > tolerance)
          return false;
        for (const attr of other) {
          const arr = attr.getArray(),
            size = attr.getElementSize();
          for (let k = 0; k < size; k++)
            if (arr[i * size + k] !== arr[j * size + k]) return false;
        }
        return true;
      }
      for (const i of indices) {
        if (remap[i] !== 0xffffffff) continue;
        const p = position.getElement(i, []);
        if (!p.every(Number.isFinite))
          throw Error("Non-finite vertex position");
        const cell = p.map((v) => Math.floor(v / tolerance));
        let representative;
        search: for (let x = -1; x <= 1; x++)
          for (let y = -1; y <= 1; y++)
            for (let z = -1; z <= 1; z++) {
              for (const j of grid.get(
                `${cell[0] + x},${cell[1] + y},${cell[2] + z}`,
              ) ?? []) {
                if (matches(i, j)) {
                  representative = j;
                  break search;
                }
              }
            }
        if (representative === undefined) {
          remap[i] = next++;
          const key = cell.join(",");
          if (!grid.has(key)) grid.set(key, []);
          grid.get(key).push(i);
        } else remap[i] = remap[representative];
      }
      if (next) compactPrimitive(prim, remap, next);
    }
}
