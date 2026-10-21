# fantasy_creature_2k_v2 — 2026-10-21 G2 rerun

Frozen record of run `fantasy_creature_g2_20261021_145300`: approved G2 seed 2026102101, 5,000 native 2K training poses, seed 2026102201, 15 epochs. Final UV loss: 0.012767705673607774. This is now the default test dataset. The obsolete v1 fixture and baseline were removed at the user’s request.

G1/G2 are approved; **G3 is pending, quality is not accepted**. Cloud exam direction error is 14.438° over all 77 bones (8.469° over 47 primary bones). Prayer convergence and digit rotations remain failure cases. Acceptance thresholds are preserved unchanged from the previous version.

The fixture contains the trained head, original rig, post-training rig metadata/effective limits, independent 288-frame video and GT, all ten perception cache arrays, original predictions, GLB, four videos, per-frame errors, validation receipts, approved pose JSON, and frozen worker sources. The 590 MB synthesis cache remains on the authorized Google Drive account, with its ID and verified SHA-256 in `manifest.json`.

`baseline_latest.json` is an **observed regression reference**, not a passing-quality designation. Tier 1 uses the current eval adapter; the cloud exam uses the frozen trainer's preprocessing. Both reports are preserved separately. All nine coarse golden probes pass in Tier 1, but they do not certify hand convergence or overall rotation targets. The three added probes preserve the cloud exam's worst primary endpoint frames (125–127).

```sh
npm run monster:eval -- dataset verify fantasy_creature_2k_v2
npm run monster:eval -- run fantasy_creature_2k_v2 --tier fast
npm run monster:eval -- diagnose fantasy_creature_2k_v2 --frame 126
```

Private Drive archives require the `bugyalu.ai@gmail.com` account. Download the archive through the authenticated Drive connector, then use `dataset pull fantasy_creature_2k_v2 --archive-path /absolute/path/to/archive.tar.gz`; the CLI's public URL downloader does not authenticate private Drive files.
