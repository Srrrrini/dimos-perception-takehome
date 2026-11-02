# Go2 semantic object map

`SemanticObjectMap` turns frame-local 3D detections into a persistent world-frame map. Run the
Go2 replay blueprint with:

```bash
dimos --replay run unitree-go2-semantic-map
```

The blueprint reuses the existing Go2 stack:

1. `Detection3DModule` runs the shipped YOLO detector on `color_image`.
2. It projects each 2D box through the calibrated camera model into the accumulated
   world-frame `global_map`.
3. `SemanticObjectMap` associates observations with nearby objects of the same semantic class.
4. Confirmed objects remain in `semantic_map` after leaving the camera view. The output is a
   standard `Detection3DArray`, so the existing Rerun bridge renders it as labeled 3D boxes.

From `dimos shell`, query the useful capability directly:

```python
app.SemanticObjectMap.where_is_object("chair")
app.SemanticObjectMap.navigate_to_object("chair")
app.SemanticObjectMap.get_metrics()
```

## Association contract

- All input positions must share the message's world frame.
- One observation can update at most one object and one object can be claimed at most once per
  frame.
- Only same-label observations within `association_distance_m` are fused.
- An object becomes queryable after `min_observations` independent frames.
- Position and size are incremental means; confidence is the strongest observation.
- Objects are retained for the lifetime of the module. This is runtime persistence as the robot
  moves, not persistence across process restarts.

## Decision log

- **Shipped YOLO instead of a new open-vocabulary model:** it is already integrated in DimOS and
  starts quickly once its exact weight is cached. A first run may require network access. The
  mapper consumes standard 3D messages, so another detector can replace it without changing
  association.
- **Accumulated LiDAR map instead of a single scan:** the existing `global_map` is denser and is
  already corrected into the `world` frame by the Go2 mapping stack.
- **World-space same-class proximity instead of raw tracker IDs:** image tracker IDs are
  short-lived and can reset after occlusion; world position remains meaningful across robot
  motion. Class gating prevents overlapping detections such as a seated person and chair from
  being merged.
- **Query plus planner goal:** `where_is_object()` reports all confirmed matches. The explicit
  `navigate_to_object()` skill selects the nearest match, publishes a configurable 1 m standoff
  goal, and lets the existing Go2 replanning A* module generate the collision-aware path. It never
  treats the object centroid itself as a traversable goal.
- **No disk database in the baseline:** the task requires persistence as the robot moves. Runtime
  state is sufficient and avoids an unspecified stale-map/versioning contract.

## Evidence and limitations

`get_metrics()` reports confirmed object/observation counts, observation span, and centroid
standard deviation. These are consistency measurements, not ground-truth accuracy. The replay has
no object annotations, so claims of precision or recall would be invented.

`Detection3DModule.get_perception_metrics()` reports effective inference throughput, 2D-to-3D
survival, camera/cloud alignment, and stage latency. The reproducible experiment history and
interpretation rules live in [`EXPERIMENTS.md`](EXPERIMENTS.md).

The detector is selected through `Detection2DModule`'s existing detector factory. Camera-LiDAR
fusion is selected through the `Detection3DLifter` factory, and association/map lifecycle through
the `SemanticMapBackend` factory. Their standard message boundary remains `Detection3DArray`, so
variants can be benchmarked without changing the blueprint's downstream contract.

The default uses prompt-free YOLO-E because it preserves instance masks and expanded the useful
vocabulary while maintaining the replay's online processing rate. Optional blueprints select the
faster fixed-vocabulary YOLO11 baseline, prompted Grounding DINO, or Grounding DINO followed by
SAM 2 instance masks. Grounding DINO prompt lists are blueprint configuration, not hard-coded
runtime policy.

The semantic vocabulary and false positives inherit the detector's limitations. Prompt-free
YOLO-E can emit broad or ambiguous scene labels. LiDAR is sparse on glass, thin furniture,
and object tops, so box dimensions are visualization aids rather than collision geometry. Dynamic
objects need evidence beyond a single tracker-ID jump before their persistent identity is moved.

The `unitree-go2-semantic-map-temporal` blueprint combines YOLO-E segmentation, hybrid mask/box
lifting, batched ResNet-18 appearance embeddings, global one-to-one assignment, and explicit
active/dormant/moved lifecycle. Static objects seen after a long gap retain their identity by class
and world position. A moved object retains its identity only when a dormant same-class candidate
passes both an appearance threshold and a best-versus-second-best margin; ambiguous revisits create
a new object instead of silently corrupting the map. Source-ID-only relocation is available as an
opt-in backend policy, but is disabled because replay showed stable IDs can accompany bad 3D jumps.

Appearance extraction is selected through the `AppearanceDescriptor` factory. The final blueprint
uses public torchvision ResNet-18 weights; no-op, HSV-histogram, and DimOS DINOv2 implementations
remain swappable. Descriptors are float16-encoded into the opaque standard `Detection3D.id` field,
so the downstream message type remains `Detection3DArray` and plain third-party IDs still decode.

Pose-graph correction is disabled in the final blueprint. The available replay messages do not
provide a stable, timestamp-matched source keyframe for each object, and detections are lifted from
an accumulated global cloud; attaching them to the newest keyframe is not geometrically valid. An
experimental implementation remains behind an explicit opt-in for future timestamped/local
measurements. See [`REVIEW_RESPONSE.md`](REVIEW_RESPONSE.md) for the required correction design.

The temporal backend exposes descriptor-bearing observations so a ReID/shape encoder can be added
without changing the map API. The current ID-field transport is documented as a temporary
interoperability compromise, not the preferred production schema.

## Submission artifacts

- [Final replay screenshot](evidence/final_replay.png)
- [Generated temporal blueprint graph](evidence/temporal_blueprint.svg)
- [Replay artifact details and regeneration command](evidence/README.md)
- [Ready-to-paste PR description](PR_DESCRIPTION.md)
- [Experiment and metrics ledger](EXPERIMENTS.md)
- [Review response and decision record](REVIEW_RESPONSE.md)
