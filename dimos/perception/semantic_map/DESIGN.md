# Semantic object map: design and evaluation

## Goal

Build a persistent world-frame object map from the supplied Go2 camera and LiDAR replay, then make
the map useful through object queries and navigation goals. The design keeps every perception stage
replaceable so that alternatives can be evaluated without changing downstream DimOS interfaces.

## Default pipeline

1. **YOLO-E detection.** Prompt-free YOLO-E produces open-vocabulary labels, 2D boxes, and instance
   masks. YOLO11 remains the faster fixed-vocabulary baseline.
2. **Projection-based sequential fusion.** The accumulated world-frame LiDAR map is transformed
   into the camera optical frame and projected through the recorded fisheye camera model. Points
   inside an instance mask form the object observation. If fewer than 20 projected points support
   the mask, the lifter retries with the detection box.
3. **Appearance description.** ResNet-18 converts each crop into a compact descriptor. Appearance
   is supporting evidence for conservative revisit/move handling, not the primary identity signal.
4. **Temporal association.** Frame-global one-to-one assignment matches same-class observations to
   nearby map objects. The backend maintains active, dormant, and moved states and requires repeated,
   unambiguous evidence before relocating a dormant identity.
5. **Useful output.** Confirmed objects are published as a standard `Detection3DArray`.
   `where_is_object()` returns map matches; `navigate_to_object()` selects the nearest match and
   publishes a configurable standoff goal to the existing replanning A* module.

## DimOS integration

The Go2 blueprint composes existing and new modules rather than creating a monolithic application:

- Inputs are the existing `color_image`, accumulated `global_map`, camera calibration, and TF
  streams.
- `Detection2DArray` and `Detection3DArray` remain the boundaries between stages.
- The existing Rerun bridge visualizes the LiDAR map and labeled 3D detections.
- Query, navigation, and metrics are exposed through normal DimOS RPCs.
- Detector, `Detection3DLifter`, `AppearanceDescriptor`, and `SemanticMapBackend` are independently
  selected in the blueprint.
- Navigation reuses the existing Go2 cost map and replanning A* implementation; the semantic map
  does not implement a second planner.

## Alternatives evaluated

Detector variants:

- YOLO11 segmentation: fastest, fixed COCO vocabulary.
- YOLO-E: broader prompt-free vocabulary and masks; selected as the default.
- Grounding DINO: configurable prompted vocabulary, boxes only, substantially slower.
- Grounding DINO + SAM 2: prompted detections refined into masks, with additional latency.

Fusion variants:

- Box frustum: highest lift survival, but admits background points.
- Mask frustum: tighter support, but sparse LiDAR causes many observations to disappear.
- Hybrid mask/box: uses the mask when sufficiently supported and otherwise falls back to the box;
  selected as the default robustness/precision trade-off.

Association variants:

- Greedy nearest-neighbour association.
- Frame-global one-to-one assignment.
- Temporal assignment with appearance evidence and active/dormant/moved lifecycle; selected as the
  default.

## Full-replay evidence

Every registered pipeline variant was run in a fresh process over the same 855-frame,
59.80-sensor-second replay. The normalized matrix is in
[`evidence/e19_pipeline_matrix.csv`](evidence/e19_pipeline_matrix.csv), and exact experiment notes
are in [`EXPERIMENTS.md`](EXPERIMENTS.md).

For the default YOLO-E pipeline:

- 625 2D detections and 291 surviving 3D observations.
- 47 confirmed map objects.
- 5.44 effective inference calls per sensor second.
- 34.69 ms mean / 51.60 ms p95 detector latency.
- 291/470 aligned detections survived lifting: 61.91%.
- 203/291 observations fused into existing objects: 69.76%.
- 0.90 ms mean / 1.89 ms p95 association latency.
- 5.10 ms mean / 9.96 ms p95 appearance extraction latency.

The controlled tracker scenario contains five known objects, identical-looking chairs, occlusion,
a revisit, and a moved object. It produced 100% track recall, zero duplicate tracks, zero identity
switches, zero false merges, and 1/1 moved-object re-identification. This isolates tracker behavior;
it is not end-to-end detector accuracy.

## Interpretation and limitations

The supplied replay has no ground-truth object annotations. Detection count, confirmed-object count,
3D survival, association rate, centroid dispersion, and nearby same-label pairs are runtime and
consistency diagnostics. They do not establish precision, recall, 3D IoU, localization error, or
IDF1/HOTA.

The camera-LiDAR method is calibration-based instance-mask frustum lifting. It is related to
projection-level sequential fusion methods such as PointPainting and frustum-based 3D detection,
but it does not train a learned LiDAR detector. Establishing the most accurate fusion strategy
requires annotated 3D boxes or object point clouds and controlled comparisons of center error,
3D IoU, point precision/recall, calibration sensitivity, and task success.

Pose-graph object reprojection is disabled. The replay does not expose a stable, timestamp-matched
source keyframe for each observation, and the detections use an accumulated global cloud. Applying
the newest-keyframe correction would therefore be geometrically unsupported.

## Reproduction

Run the default pipeline:

```bash
uv run dimos --replay --replay-db data/go2_short.db --viewer rerun \
  run unitree-go2-semantic-map-temporal
```

Inspect the map from `dimos shell`:

```python
app.SemanticObjectMap.where_is_object("chair")
app.SemanticObjectMap.navigate_to_object("chair")
app.SemanticObjectMap.get_metrics()
app.Detection3DModule.get_perception_metrics()
```

