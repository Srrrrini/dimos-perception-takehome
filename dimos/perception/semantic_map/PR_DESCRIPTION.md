## Design

This adds a persistent 3D semantic object map for the Unitree Go2 replay. It fits into DimOS as
swappable modules joined by existing message contracts rather than a monolithic perception node:

- YOLO11 segmentation supplies 2D instances and masks.
- The hybrid lifter projects the calibrated accumulated LiDAR cloud into each instance mask and
  falls back to the 2D box when mask support is insufficient.
- Global one-to-one assignment prevents two observations in one frame from claiming the same map
  object.
- The temporal backend tracks active/dormant/moved lifecycle, uses conservative ResNet-18
  appearance ReID for revisits, and exposes `where_is_object()` plus structured metrics.
- Pose-graph reprojection is deliberately disabled: this replay does not expose a stable,
  timestamp-matched source keyframe and detections use an accumulated global cloud. Enabling the
  earlier newest-keyframe approximation would make an unsupported geometry claim.

Detector, appearance encoder, 3D lifter, and map backend are independently factory-selected. Their
standard boundaries remain `Detection2DArray` and `Detection3DArray`, making later model/fusion
comparisons blueprint changes rather than rewrites.

## Replay proof

![Final replay](evidence/final_replay.png)

![Temporal blueprint](evidence/temporal_blueprint.svg)

The screenshot shows the camera, odometry, telemetry, accumulated LiDAR map, and persistent labeled
3D objects from `unitree-go2-semantic-map-temporal`. A complete replay artifact was also generated
and verified:

```text
recordings/20261030-204447-6317-unitree-go2-semantic-map-temporal/rerun.rrd
SHA-256: 00fd881da11eeee07395db895637616e30aaa584fba18182064df5d316ed7791
Size: 112 MiB; 9,231 chunks; 53 entities; verification passed
```

## Results

Final corrected E10 replay:

- 224/224 surviving 3D detections received appearance descriptors.
- 181/224 valid observations were genuinely fused into existing objects: 80.80% strict accepted-
  fusion rate. This excludes new tracks and any pending/unaccepted relocation.
- 0.62 ms mean / 0.77 ms p95 association latency.
- 4.16 ms mean / 8.23 ms p95 appearance extraction per detection.
- Zero reported move events on a replay believed to be static; this is not ground-truth proof.
- Two geometric loop closures were reported by the mapper; no semantic correction claim is made.
- Zero invalid observations, rejected frames, or detector/lifter/descriptor failures in the final
  corrected replay.
- 29 focused semantic-map, appearance, and lifting tests pass; lint and type checking pass.

The replay has no ground-truth object annotations, so these are runtime, consistency, duplicate
proxy, and safety measurements—not invented precision/recall claims. A moved-object recall claim
requires an annotated move-and-revisit recording.

Detailed experiment history, including rejected trials, is in
`dimos/perception/semantic_map/EXPERIMENTS.md`. Review-driven corrections and invalidated claims are
recorded in `dimos/perception/semantic_map/REVIEW_RESPONSE.md`.
