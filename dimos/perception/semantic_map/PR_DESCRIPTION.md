## Design

This adds a persistent 3D semantic object map for the Unitree Go2 replay. It fits into DimOS as
swappable modules joined by existing message contracts rather than a monolithic perception node:

- YOLO11 segmentation supplies 2D instances and masks.
- The hybrid lifter projects the calibrated accumulated LiDAR cloud into each instance mask and
  falls back to the 2D box when mask support is insufficient.
- Global one-to-one assignment prevents two observations in one frame from claiming the same map
  object.
- The temporal backend tracks active/dormant/moved lifecycle, uses conservative ResNet-18
  appearance ReID for revisits, and exposes `where_is_object()`, `navigate_to_object()`, and
  structured metrics. Navigation publishes a 1 m standoff goal to the existing Go2 A* planner.
- Pose-graph reprojection is deliberately disabled: this replay does not expose a stable,
  timestamp-matched source keyframe and detections use an accumulated global cloud. Enabling the
  earlier newest-keyframe approximation would make an unsupported geometry claim.

Detector, appearance encoder, 3D lifter, and map backend are independently factory-selected. Their
standard boundaries remain `Detection2DArray` and `Detection3DArray`, making later model/fusion
comparisons blueprint changes rather than rewrites.

Pretrained weights may download automatically on first use, per the take-home clarification. The
default YOLO11-seg and ResNet-18 paths already do this; the optional Grounding DINO path uses the
Hugging Face cache, and YOLO-E falls back from DimOS Git-LFS data to Ultralytics' first-use download.
The default CUDA replay is measured below. A fixed 13-frame CPU benchmark is also included and
records the database hash, frame IDs, model/package versions, cold latency, and warm latency.

## Replay proof

![Final replay](evidence/final_replay.png)

![Temporal blueprint](evidence/temporal_blueprint.svg)

The screenshot shows the camera, odometry, telemetry, accumulated LiDAR map, and persistent labeled
3D objects from `unitree-go2-semantic-map-temporal`. A complete replay artifact was also generated
and verified:

```text
recordings/20261101-125124-ea4c-unitree-go2-semantic-map-temporal/rerun.rrd
SHA-256: 8e70e16ac003a0dbbab34154c204e7002e061505fd90524d8ef966751d5a7e9f
Size: 112 MiB; `rerun rrd verify` passed
```

## Results

Final headless E17 replay:

- 222/222 surviving 3D detections received appearance descriptors.
- 175/222 valid observations were genuinely fused into existing objects: 78.83% strict accepted-
  fusion rate. This excludes new tracks and any pending/unaccepted relocation.
- 0.64 ms mean / 0.87 ms p95 association latency.
- 4.25 ms mean / 8.27 ms p95 appearance extraction per detection.
- 571 detector calls at 28.11 ms mean / 42.21 ms p95 inference latency; 222/321 aligned 2D
  detections survived 3D lifting (69.16%).
- Zero reported move events on a replay believed to be static; this is not ground-truth proof.
- Two geometric loop closures were reported by the mapper; no semantic correction claim is made.
- Zero invalid observations, rejected frames, or detector/lifter/descriptor failures in the final
  headless replay.
- The fixed CPU sample ran YOLO11n-seg at 81.11 ms mean / 95.05 ms p95 warm latency (12.33 Hz).
- Controlled tracker simulation measured 100% track recall, 0 duplicate tracks, 0 identity
  switches, 0 false merges, and 1/1 moved-object re-identification across five known objects. This
  is component-level association evidence, not end-to-end detector recall.
- 48 focused semantic-map, detector-adapter, blueprint, appearance, and lifting tests pass; lint and
  type checking pass.

The replay has no ground-truth object annotations, so these are runtime, consistency, duplicate
proxy, and safety measurements—not invented precision/recall claims. A moved-object recall claim
requires an annotated move-and-revisit recording.

The team confirmed that no annotated evaluation dataset is available and suggested simulation for
accuracy evaluation. Accordingly, replay results remain evidence of runtime and internal
consistency; deterministic tracker tests cover known stationary, duplicate, revisit, and moved-object
scenarios without presenting them as end-to-end detector accuracy.

Detailed experiment history, including rejected trials, is in
`dimos/perception/semantic_map/EXPERIMENTS.md`. Review-driven corrections and invalidated claims are
recorded in `dimos/perception/semantic_map/REVIEW_RESPONSE.md`.
