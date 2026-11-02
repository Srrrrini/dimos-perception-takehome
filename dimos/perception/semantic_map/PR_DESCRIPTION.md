## Design

This adds a persistent 3D semantic object map for the Unitree Go2 replay. It fits into DimOS as
swappable modules joined by existing message contracts rather than a monolithic perception node:

- Prompt-free YOLO-E supplies open-vocabulary 2D instances and masks. YOLO11 remains an explicit
  faster fixed-vocabulary baseline.
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

The blueprint consumes DimOS's existing `color_image`, accumulated world-frame `global_map`,
camera calibration, and TF streams. It publishes the standard `Detection3DArray`, so the existing
Rerun bridge and downstream modules work without a new message type. Query, navigation, and metrics
are exposed through normal DimOS RPCs; navigation reuses the existing cost map and replanning A*
module rather than introducing another planner.

## Alternatives evaluated

- **Detection:** YOLO11 segmentation, prompt-free YOLO-E, prompted Grounding DINO, and Grounding
  DINO followed by SAM 2. YOLO-E is the default because it broadened observed vocabulary while
  maintaining the replay's online pipeline rate. YOLO11 remains the faster conservative baseline;
  DINO variants operated below 1 Hz.
- **Camera-LiDAR fusion:** box frustum, mask frustum, and hybrid mask/box lifting. The selected
  projection-based sequential fusion path transforms world LiDAR into the calibrated camera,
  selects points inside the instance mask, and retries with the 2D box when fewer than 20 points
  support the mask. Box lifting survives more often but admits background; mask lifting is tighter
  but loses sparse observations; hybrid recovered 42% of fallback attempts.
- **Association:** greedy nearest-neighbour, frame-global one-to-one assignment, and temporal
  assignment with appearance and active/dormant/moved lifecycle. The temporal backend is selected
  because it enforces same-frame exclusivity and supports conservative revisit and move handling.

Pretrained weights may download automatically on first use, per the take-home clarification. The
default YOLO-E and ResNet-18 paths already do this; the optional Grounding DINO path uses the
Hugging Face cache, and YOLO-E falls back from DimOS Git-LFS data to Ultralytics' first-use download.
The default CUDA replay is measured below. A fixed 13-frame CPU benchmark is also included and
records the database hash, frame IDs, model/package versions, cold latency, and warm latency.

## Replay proof

![Final replay](https://github.com/Srrrrini/dimos-perception-takehome/blob/a4729a3fd7/dimos/perception/semantic_map/evidence/final_replay.png?raw=true)

![Temporal blueprint](https://github.com/Srrrrini/dimos-perception-takehome/blob/a4729a3fd7/dimos/perception/semantic_map/evidence/temporal_blueprint.svg?raw=true)

The screenshot shows the camera, odometry, telemetry, accumulated LiDAR map, and persistent labeled
3D objects from the temporal pipeline using the YOLO11 baseline. The final YOLO-E default was also
recorded over the complete replay and is available through Git LFS at
[`evidence/yoloe_default_replay.rrd`](https://github.com/Srrrrini/dimos-perception-takehome/blob/srinivas/feat/semantic-object-map/dimos/perception/semantic_map/evidence/yoloe_default_replay.rrd).

```text
dimos/perception/semantic_map/evidence/yoloe_default_replay.rrd
SHA-256: b96b00c157ffbdb0dd1deccf43f5f4afced0ea78e9629a6390eabd44e6422f4c
Size: 120,732,545 bytes; `rerun rrd verify` passed
```

## Results

Final full-replay YOLO-E measurement (E19):

- 291/291 surviving 3D detections received appearance descriptors.
- 203/291 valid observations were genuinely fused into existing objects: 69.76% strict accepted-
  fusion rate. This excludes new tracks and any pending/unaccepted relocation.
- 0.90 ms mean / 1.89 ms p95 association latency.
- 5.10 ms mean / 9.96 ms p95 appearance extraction per detection.
- 568 detector calls at 34.69 ms mean / 51.60 ms p95 inference latency; 291/470 aligned 2D
  detections survived 3D lifting (61.91%).
- Zero reported move events on a replay believed to be static; this is not ground-truth proof.
- Two geometric loop closures were reported by the mapper; no semantic correction claim is made.
- Zero rejected frames or detector/lifter/descriptor failures in the final headless replay.
- The fixed GPU sample ran YOLO-E at 28.39 ms mean / 36.41 ms p95 warm detector latency (35.23 Hz).
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

## Run and inspect

```bash
uv run dimos --replay --replay-db data/go2_short.db --viewer rerun \
  run unitree-go2-semantic-map-temporal
```

From `dimos shell`:

```python
app.SemanticObjectMap.where_is_object("chair")
app.SemanticObjectMap.navigate_to_object("chair")
app.SemanticObjectMap.get_metrics()
app.Detection3DModule.get_perception_metrics()
```
