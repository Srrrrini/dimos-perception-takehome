# Semantic map experiment ledger

Every entry records one reproducible Go2 replay configuration. Metrics are comparable only when
the dataset, replay command, hardware, and configuration are unchanged. Nearby same-label pairs
are duplicate diagnostics, not accuracy measurements; accuracy requires annotations.

## Run template

- Commit:
- Date:
- Dataset:
- Hardware:
- Detector/config:
- 3D lifting/config:
- Association/config:
- Command:
- Visual artifact:
- Perception metrics:
- Map metrics:
- Qualitative observations:
- Conclusion / next experiment:

## E00 — uninstrumented nearest-centroid baseline

- Commit: `9cadb7fab017f1fb47a43b2a5234ad8b508cf38d`
- Date: 2026-10-30
- Dataset: `data/go2_short.db` (approximately 60 seconds)
- Detector/config: YOLO11n, confidence 0.5, IoU 0.6, input cap 10 Hz
- 3D lifting/config: accumulated world cloud projected through 2D boxes; default point filters
- Association/config: greedy nearest same-label centroid, 0.75 m gate, 2 observations to confirm
- Command: `dimos --replay --replay-db data/go2_short.db run unitree-go2-semantic-map`
- Visual artifact: `recordings/20261030-151036-ea66-unitree-go2-semantic-map/rerun.rrd`
- Map metrics: 43 candidates, 28 confirmed, 179 confirmed observations, 6.39 observations/object,
  0.189 m mean centroid standard deviation, 0.353 m maximum standard deviation, 57.35 s maximum
  observed duration
- Qualitative observations: repeated same-class map objects are visible; many small shelf objects
  are missed. This run cannot distinguish detector misses from 2D-to-3D fusion losses.
- Conclusion: add stage-level throughput, survival, association, and duplicate diagnostics before
  changing algorithms.

## E01 — instrumented nearest-centroid baseline

- Commit: this instrumentation commit (resolve with `git log --oneline`)
- Date: 2026-10-30
- Dataset: `data/go2_short.db` (59.80 seconds between first and last camera timestamps)
- Hardware: local workstation; detector selected CUDA when available
- Detector/config: YOLO11n, confidence 0.5, IoU 0.6, input cap 10 Hz
- 3D lifting/config: accumulated world cloud projected through 2D boxes; default point filters
- Association/config: greedy nearest same-label centroid, 0.75 m gate, 2 observations to confirm
- Command: `dimos --replay --replay-db data/go2_short.db --viewer none run
  unitree-go2-semantic-map`
- Visual artifact: none for this metrics-only run; E00 Rerun remains the visual baseline
- Perception metrics: 855 input frames at 14.30 source FPS; 566 unique detector invocations at
  6.67 calls/s; inference latency 31.08 ms mean / 64.01 ms p95; 238 2D detections entered
  fusion and 173 survived into 3D (72.69%); image/cloud alignment delta 124.23 ms mean / 236.99
  ms p95; fusion latency 9.51 ms mean / 51.85 ms p95
- Map metrics: 44 candidates, 25 confirmed, 173 observations, 74.57% associated to an existing
  track, 25.43% created a new track, 6.16 confirmed observations/object, 0.216 m mean centroid
  standard deviation; same-label nearby-pair diagnostics were 0 at 0.5 m, 2 at 0.75 m, and 7 at
  1.0 m
- Qualitative observations: 27.31% of detections entering fusion were lost before 3D output. The
  236.99 ms p95 image/cloud alignment delta is close to the configured 250 ms tolerance. The
  detector result is already multicast by DimOS `backpressure()`; its consumers have independent
  latest-frame queues. Fusion processed fewer frames because its heavier downstream work dropped
  stale frames, not because YOLO ran once per subscriber.
- Conclusion: retain the existing shared detector stream. Next introduce strategy interfaces so
  detector, lifter, and association variants can be benchmarked against this ledger; separately
  investigate the high image/cloud alignment delta and fusion attrition.

## E02 — injectable lifter and map backend

- Commit: this strategy-interface commit (resolve with `git log --oneline`)
- Date: 2026-10-30
- Dataset/hardware/detector/association: same as E01
- 3D lifting/config: `ProjectedPointCloudLifter`; same projection and filters as E01, but the
  world cloud is projected once per detection frame and shared by every 2D instance
- Command: same as E01
- Visual artifact: none for this metrics-only run
- Perception metrics: 855 input frames at 14.30 source FPS; 564 detector calls at 6.72 calls/s;
  inference 31.11 ms mean / 60.69 ms p95; 244 2D detections entered fusion and 185 survived
  (75.82%); alignment delta 121.23 ms mean / 233.20 ms p95; fusion latency 10.76 ms mean / 36.35
  ms p95
- Map metrics: 40 candidates, 25 confirmed, 185 observations, 78.38% association rate, 6.8
  confirmed observations/object, 0.172 m mean centroid standard deviation; nearby same-label pair
  counts were 0 at 0.5 m, 4 at 0.75 m, and 7 at 1.0 m
- Comparison to E01: p95 fusion latency fell from 51.85 to 36.35 ms and measured 3D survival rose
  from 72.69% to 75.82%. Mean fusion latency rose from 9.51 to 10.76 ms. Because replay sampling
  and inference timing vary between runs, these single-run differences are directional rather than
  a statistical performance claim.
- Conclusion: the unchanged command works with explicit `Detection3DLifter` and
  `SemanticMapBackend` boundaries. Detector, fusion, and association variants can now be injected
  independently while preserving the standard `Detection3DArray` contract.

## E03 — YOLO11n instance masks

- Commit: this segmentation-variant commit (resolve with `git log --oneline`)
- Date: 2026-10-30
- Dataset/hardware/lifter/association: same as E02
- Detector/config: YOLO11n-seg, confidence 0.5, IoU 0.6, input cap 10 Hz; official Ultralytics
  weight is downloaded on first use when it is not in the DimOS model directory
- Command: `dimos --replay --replay-db data/go2_short.db --viewer none run
  unitree-go2-semantic-map-seg`
- Visual artifact: none for this metrics-only run
- Perception metrics: 855 input frames at 14.30 source FPS; 564 detector calls at 7.07 calls/s;
  394 total 2D detections; inference 34.40 ms mean / 60.77 ms p95 (1.89 s cold-start maximum);
  301 2D detections entered fusion and 121 survived (40.20%); alignment 122.76 ms mean / 233.23
  ms p95; fusion 10.95 ms mean / 37.79 ms p95
- Map metrics: 38 candidates, 23 confirmed, 106 confirmed observations, 68.60% association rate,
  4.61 confirmed observations/object, 0.128 m mean centroid standard deviation; nearby same-label
  pair counts were 0 at 0.5 m, 2 at 0.75 m, and 2 at 1.0 m
- Comparison to E02: the mask model produced substantially more 2D detections (394 total; 301
  reached fusion versus 244 in E02), but 3D survival fell from 75.82% to 40.20% and confirmed
  observations fell from 185 to 106. Inference p95 was effectively unchanged; the first model
  invocation caused the large cold-start maximum.
- Conclusion: instance masks expose more shelf candidates and reject background LiDAR points, but
  the accumulated cloud is too sparse or temporally misaligned inside many small masks. Keep this
  as an optional blueprint, not the default. Next evaluate a hybrid mask-plus-box fallback and
  explicitly measure how many masks lack enough projected points.

## E04 — hybrid mask-to-box lifting

- Commit: this hybrid-lifter commit (resolve with `git log --oneline`)
- Date: 2026-10-30
- Dataset/hardware/detector/association: same as E03
- 3D lifting/config: use the segmentation mask when at least 20 projected cloud points support it
  and filtering succeeds; otherwise retry the same detection using its 2D bounding box
- Command: `dimos --replay --replay-db data/go2_short.db --viewer none run
  unitree-go2-semantic-map-hybrid`
- Visual artifact: none for this metrics-only run
- Perception metrics: 855 input frames at 14.30 source FPS; 567 detector calls at 7.33 calls/s;
  402 total 2D detections; inference 31.50 ms mean / 55.48 ms p95 (1.46 s cold-start maximum);
  293 detections entered fusion and 202 survived (68.94%); alignment 121.01 ms mean / 231.72 ms
  p95; fusion 12.76 ms mean / 53.04 ms p95
- Lifter diagnostics: 288 masks attempted; 143 mask results retained; 20 masks (6.94%) had fewer
  than 20 projected points; 145 box fallbacks attempted and 59 recovered a valid 3D observation
  (40.69% fallback recovery); mean raw projected support was 1,979.91 points/mask
- Map metrics: 46 candidates, 29 confirmed, 185 confirmed observations, 77.23% association rate,
  6.38 confirmed observations/object, 0.174 m mean centroid standard deviation; nearby same-label
  pair counts were 2 at 0.5 m, 4 at 0.75 m, and 8 at 1.0 m
- Comparison to E03: survival rose from 40.20% to 68.94%, confirmed observations from 106 to 185,
  and confirmed objects from 23 to 29. Fusion p95 rose from 37.79 to 53.04 ms because failed masks
  run the point filters again for the box fallback. Duplicate proxies also rose, so the additional
  objects are not automatically additional true positives.
- Conclusion: the fallback recovers most of the mask-only map loss while preserving masks for
  supported objects. Raw mask sparsity explains only 20 of 145 fallbacks; filter rejection is the
  larger loss source. Keep the hybrid as an experimental option until visual inspection and an
  association improvement distinguish recovered shelf objects from duplicate landmarks.

## E05 — global one-to-one association

- Commit: this global-assignment commit (resolve with `git log --oneline`)
- Date: 2026-10-30
- Dataset/hardware/detector/lifter: same as E04
- Association/config: Hungarian minimum-cost assignment over all observations and existing objects
  per frame; same-label 0.75 m gate with a bounded source-track-ID preference
- Command: `dimos --replay --replay-db data/go2_short.db --viewer none run
  unitree-go2-semantic-map-assignment`
- Visual artifact: the immediately preceding hybrid Rerun was inspected live; the shelf scene
  confirms broad COCO labels but not individual grocery SKU coverage
- Perception metrics: 855 input frames at 14.30 source FPS; 569 detector calls; 409 total 2D
  detections; inference 32.40 ms mean / 67.07 ms p95; 305 detections entered fusion and 192
  survived (62.95%); fusion 11.21 ms mean / 37.85 ms p95
- Map metrics: 44 candidates, 29 confirmed, 177 confirmed observations, 77.08% association rate,
  6.10 confirmed observations/object, 0.181 m mean centroid standard deviation; nearby same-label
  pair counts were 1 at 0.5 m, 3 at 0.75 m, and 7 at 1.0 m; association latency was 2.91 ms mean /
  0.13 ms p95 (mean includes the one-time lazy SciPy import)
- Comparison to E04: candidate objects fell from 46 to 44 and duplicate proxies from 2/4/8 to
  1/3/7, while confirmed-object count remained 29. Perception counts varied between replays, so
  this is directional evidence; deterministic unit tests cover the order-dependent failure case.
- Conclusion: global assignment removes the greedy matcher’s order dependence at negligible
  steady-state cost. Use it in the final blueprint, while keeping the backend injectable. Temporal
  lifecycle and moved-object re-identification remain separate from per-frame assignment.

## E06 — temporal lifecycle and conservative moved-object handling

- Commit: this temporal-tracker commit (resolve with `git log --oneline`)
- Date: 2026-10-30
- Dataset/hardware/detector/lifter: same as E05
- Association/config: E05 global assignment plus active/dormant/moved lifecycle; a same-source
  relocation requires three spatially coherent observations within two seconds. A matching
  appearance descriptor can re-identify immediately, but the current replay stream does not
  provide descriptors.
- Command: `dimos --replay --replay-db data/go2_short.db --viewer none run
  unitree-go2-semantic-map-temporal`
- Perception metrics: 855 input frames at 14.30 source FPS; 567 detector calls at 6.99 Hz;
  inference 38.33 ms mean / 81.19 ms p95; 284 detections entered fusion and 192 survived
  (67.61%); fusion 13.88 ms mean / 63.19 ms p95
- Map metrics: 43 candidates, 29 confirmed, 176 confirmed observations, 77.60% association rate,
  0.148 m mean centroid standard deviation; nearby same-label pairs were 0/3/8 at
  0.5/0.75/1.0 m; association latency was 1.49 ms mean / 0.19 ms p95
- Lifecycle metrics at shutdown: 5 active, 38 dormant, 0 moved, 0 move events. Earlier trial
  variants that trusted a single source-track jump reported 2 and 3 move events on this static
  replay; requiring three coherent observations removed those false move claims.
- Loop closure observation: the Go2 pose-graph backend detected two geometric loop closures.
  Semantic observations currently store world coordinates rather than keyframe-relative
  measurements, so they are not retroactively corrected when the pose graph changes.
- Conclusion: lifecycle and moved-object identity are now explicit and unit-tested without
  overstating what this unannotated, static replay proves. Full semantic loop closure requires
  keyframe-anchored observations and propagation of pose-graph corrections; long-term moved-object
  re-identification requires detector descriptors or crops at the observation boundary.

## E07 — keyframe-anchored semantic correction

- Commit: this semantic-loop-correction commit (resolve with `git log --oneline`)
- Date: 2026-10-30
- Dataset/hardware/detector/lifter/association: same final stack as E06
- Method: retain each 3D observation in the latest PGO keyframe's local coordinates; on every
  optimized keyframe update, reproject all historical samples, recompute object centroids and
  variance, and republish the semantic map. Merge only same-label tracks that become spatially
  close and whose observed lifetimes do not overlap.
- Command: `dimos --replay --replay-db data/go2_short.db --viewer none run
  unitree-go2-semantic-map-temporal`
- Replay event: 66 pose-graph updates and two geometric loop closures; maximum semantic centroid
  correction was 0.110 m. No pair met the conservative post-correction merge rule.
- Perception metrics: 855 input frames at 14.30 source FPS; 570 detector calls at 6.67 Hz; 423
  2D detections; inference 18.32 ms mean / 23.94 ms p95; 337 detections entered fusion and 233
  survived (69.14%); fusion 7.82 ms mean / 27.21 ms p95
- Map metrics: 51 candidates, 34 confirmed, 216 confirmed observations, 78.11% association rate,
  0.140 m mean centroid standard deviation; nearby same-label pairs were 2/3/10 at
  0.5/0.75/1.0 m; association latency was 0.48 ms mean / 0.40 ms p95
- Lifecycle result: 0 moved objects and 0 move events after disabling source-ID-only relocation.
  Two preceding E07 trials produced one false move each even with three- and five-observation
  confirmation, showing that stable upstream IDs are not sufficient evidence for relocation.
- Conclusion: semantic landmarks now follow the existing DimOS pose graph through loop closure
  instead of remaining frozen in stale world coordinates. This is correction propagation rather
  than a joint object/pose factor graph; the latter would require adding semantic factors to PGO.
