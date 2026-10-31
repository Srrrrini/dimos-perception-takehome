# Review response and decision record

This file records external-review findings, their validation status, and the resulting decisions.
It complements [`EXPERIMENTS.md`](EXPERIMENTS.md), which remains the chronological metrics ledger.

## 2026-10-31 review

### 1. Pose-graph correction geometry — valid blocker

The replay exposes a `PoseArray` containing the current ordered keyframe poses, but a semantic
detection contains neither its source keyframe ID nor a timestamp-indexed keyframe association.
The 3D centroid is also lifted from the accumulated `global_map`, not from a keyframe-local scan.
Attaching every detection to the newest keyframe can therefore select the wrong anchor and can
reapply a correction already present in the accumulated cloud.

Decision: pose-graph reprojection is disabled by default and is not claimed by the final blueprint.
The existing implementation remains behind `enable_experimental_pose_graph_correction` for future
work, but must not be enabled until observations carry a stable keyframe ID and a genuinely local
measurement. E07 and E09 are retained in the ledger as historical experiments and explicitly
invalidated for geometric-correctness claims.

Correct future design:

1. Lift against the timestamp-matched raw scan or retain the sensor-frame object measurement.
2. Associate it with a stable PGO keyframe ID, not a mutable array index.
3. Store `T_keyframe_object` and re-evaluate `T_world_keyframe * T_keyframe_object` after PGO.
4. Test pre-map insertion, keyframe reordering, delayed frames, and repeated optimization.

### 2. Association metric — valid

Previously, `observations - newly_created_tracks` was labeled as matched. A pending relocation can
reuse an object in the assignment result without actually updating that object, inflating the rate.

Decision: `accepted_fusions` now counts only increases in existing objects' observation counts.
`unfused_observations` separately counts pending/unaccepted observations. `association_rate` is now
strictly `accepted_fusions / valid_observations`. Historical E01–E09 rates are labeled legacy
proxies and are not directly comparable until replayed.

### 3. Frame-level failure isolation — valid

Decision: detector, camera-LiDAR lifting, appearance extraction, and transform lookup now isolate
ordinary per-frame exceptions, log them, and increment explicit failure counters. A descriptor
failure preserves the valid 3D detection without a descriptor. Process-fatal errors are not
silenced.

### 4. Frame and numeric contracts — valid

Decision: the semantic map rejects non-world frames, non-finite positions/sizes/confidences/times,
non-positive box dimensions, confidence outside `[0, 1]`, and excessively late frames. The
experimental PGO path additionally rejects non-world or non-unit/invalid poses.

### 5. Evidence wording — valid

Decision: use “zero reported move events on a recording believed to be static,” never “zero false
moves.” Pose-array callbacks are not loop corrections; the final submission makes no semantic-loop
correction claim.

### 6. Reproducibility and descriptor transport — valid with scoped mitigation

The final replay used:

- `ultralytics==8.4.14`, `yolo11n-seg.pt`, SHA-256
  `55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152`, 6,182,636 bytes.
- `torchvision==0.22.1+cu128`, `resnet18-f37072fd.pth`, SHA-256
  `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`, 46,830,571 bytes.

Both must be cached for a network-free replay. The installed Ultralytics distribution declares
AGPL-3.0 with a separate commercial-license option; torchvision declares BSD, while pretrained
weight use remains subject to dataset/model terms. This take-home does not make a commercial-use
license determination.

The float16/base64 descriptor in `Detection3D.id` is a compatibility bridge, not an ideal public
schema: it adds about 1.4 KiB for a 512-D vector and makes IDs opaque. Plain external IDs still
decode unchanged and the appearance stage is injectable/no-op. Production follow-up should add a
versioned semantic-observation message or a parallel descriptor topic keyed by a compact detection
ID; changing DimOS's public message schema is intentionally outside this take-home.
