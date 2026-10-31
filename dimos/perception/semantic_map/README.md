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

- **Shipped YOLO instead of a new open-vocabulary model:** it is already an offline dependency,
  starts quickly, and makes replay reproducible. The mapper consumes standard 3D messages, so a
  different detector can replace it without changing association.
- **Accumulated LiDAR map instead of a single scan:** the existing `global_map` is denser and is
  already corrected into the `world` frame by the Go2 mapping stack.
- **World-space same-class proximity instead of raw tracker IDs:** image tracker IDs are
  short-lived and can reset after occlusion; world position remains meaningful across robot
  motion. Class gating prevents overlapping detections such as a seated person and chair from
  being merged.
- **“Where is X?” instead of navigation:** querying directly demonstrates semantic persistence
  without inventing navigation stand-off distances or safety behavior that the task does not
  specify.
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

The semantic vocabulary and false positives inherit the detector's limitations. The default COCO
model can, for example, mislabel large indoor fixtures. LiDAR is sparse on glass, thin furniture,
and object tops, so box dimensions are visualization aids rather than collision geometry. Dynamic
objects need evidence beyond a single tracker-ID jump before their persistent identity is moved.

The `unitree-go2-semantic-map-temporal` blueprint combines YOLO11 segmentation, hybrid mask/box
lifting, global one-to-one assignment, and explicit active/dormant/moved lifecycle. Static objects
seen after a long gap retain their identity by class and world position. A moved object retains its
identity only after an injected appearance-descriptor match; ambiguous revisits create a new
object instead of silently corrupting the map. Source-ID-only relocation is available as an
opt-in backend policy, but is disabled because replay showed stable IDs can accompany bad 3D jumps.

Semantic observations retain their keyframe-local coordinates and are reprojected whenever the Go2
pose graph updates, including after loop closure. Tracks are merged after correction only when they
share a label, become spatially close, and have non-overlapping lifetimes. This propagates PGO
corrections into the semantic layer; it is not a joint object/pose factor graph like SlideSLAM.
The temporal backend also exposes descriptor-bearing observations so a ReID/shape encoder can be
added without changing the map API.
