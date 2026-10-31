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
