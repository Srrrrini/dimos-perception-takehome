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

> **Invalidated 2026-10-31:** this experiment attached detections to the newest keyframe without a
> stable timestamp-matched keyframe ID, while centroids came from an accumulated global cloud. Its
> correction magnitudes and merge behavior do not establish geometrically valid semantic loop
> correction. The final blueprint disables this path. See `REVIEW_RESPONSE.md`.

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

## E08 — learned appearance descriptors and dormant-object ReID

- Commit: this appearance-ReID commit (resolve with `git log --oneline`)
- Date: 2026-10-30
- Dataset/hardware/detector/lifter/association: same final stack as E07; NVIDIA RTX 5070 Ti
- Method: batch 2D instance crops through an ImageNet ResNet-18 feature extractor, L2-normalize
  512-D embeddings, carry float16 descriptors through the standard `Detection3D.id` string, and
  allow a far 3D association only for dormant same-class objects when cosine similarity is at
  least 0.90 and exceeds the second-best candidate by at least 0.05.
- Command: `dimos --replay --replay-db data/go2_short.db --viewer none run
  unitree-go2-semantic-map-temporal`
- Descriptor metrics: 207/207 surviving 3D detections described (100% coverage); 5.19 ms mean /
  10.93 ms p95 per detection. Weights download once from torchvision's public model cache.
- Perception metrics: 855 input frames at 14.30 source FPS; 572 detector calls at 6.91 Hz; 418
  2D detections; inference 29.96 ms mean / 53.75 ms p95; 312 detections entered fusion and 207
  survived (66.35%); fusion 12.70 ms mean / 45.27 ms p95
- Map metrics: 40 candidates, 27 confirmed, 194 confirmed observations, 80.19% legacy association
  proxy (withdrawn as an accepted-fusion rate after review),
  0.190 m mean centroid standard deviation; nearby same-label pairs were 1/2/5 at
  0.5/0.75/1.0 m; cached descriptor matching reduced association latency from a rejected trial's
  9.95 ms mean / 54.97 ms p95 to 0.81 ms mean / 0.95 ms p95
- Safety result: 0 reported moved objects and 0 reported move events on a recording believed to be
  static; this is not ground-truth false-move accuracy. A histogram-only trial
  falsely moved 7 objects; an unconstrained ResNet trial falsely moved 4. Both were rejected.
  Requiring the candidate to be dormant and unambiguous removed those false relocations.
- Dependency trial: DimOS's DINOv2 wrapper was evaluated first, but Hugging Face returned HTTP 401
  for `facebook/dinov2-base` on this host. The injectable interface retains DINO as an option;
  ResNet-18 is the replayed final default.
- Conclusion: the pipeline has end-to-end general-object appearance ReID with measured cost and
  conservative temporal gating. This recording shows zero reported moves, but neither false-move
  rate nor moved-object recall is measurable without an annotated move/revisit sequence.

## E09 — final visual-evidence replay

> **Qualification 2026-10-31:** `graph_corrections` counted pose-array callbacks, not verified loop
> corrections. The semantic-correction path and its claims are disabled in the final blueprint.

- Commit: documentation/evidence commit following E08
- Date: 2026-10-30
- Stack: unchanged from the final E08 temporal blueprint
- Command: `dimos --replay --replay-db data/go2_short.db --viewer rerun run
  unitree-go2-semantic-map-temporal` with Rerun file saving enabled
- Visual artifact: [`evidence/final_replay.png`](evidence/final_replay.png)
- Full artifact: `recordings/20261030-204447-6317-unitree-go2-semantic-map-temporal/rerun.rrd`
  (112 MiB, SHA-256 `00fd881da11eeee07395db895637616e30aaa584fba18182064df5d316ed7791`)
- Artifact verification: 1 file verified without error; 9,231 chunks, 53 entity paths, 9,268 rows
- Submission verification: 24 focused tests passed; Ruff and mypy passed; all five semantic-map
  blueprints, including `unitree-go2-semantic-map-temporal`, were discoverable through `dimos list`
- Descriptor result: 191/191 surviving 3D detections described; 5.04 ms mean / 9.63 ms p95
- Map result: 46 candidates, 29 confirmed, 174 confirmed observations, 75.92% association rate;
  nearby same-label duplicate proxies were 0/2/8 at 0.5/0.75/1.0 m
- Temporal/PGO result as originally logged: 0 reported move events; 66 pose-array callbacks, two
  mapper-reported loop closures, 0.108 m experimental centroid shift, and 0 experimental merges.
  These do not validate semantic loop correction.
- Note: small differences from E08 are expected because replay scheduling changes which source
  frames pass the detector's frequency cap. E08 remains the primary performance run; E09 proves
  that the final visual artifact was generated by the same committed blueprint.

## E10 — external-review remediation

- Date: 2026-10-31
- Scope: validate and remediate the six findings recorded in `REVIEW_RESPONSE.md`
- Geometry: pose-graph reprojection disabled by default; the final blueprint no longer claims
  semantic loop correction
- Metric correction: accepted fusion now requires an actual increase in an existing object's
  observation count; pending moves are reported separately as `unfused_observations`
- Runtime hardening: detector, lifting, descriptor, and transform failures are isolated per frame
  and counted
- Contract hardening: reject wrong-frame, non-finite, non-positive-size, invalid-confidence, and
  excessively late observations; validate experimental pose updates before use
- Verification: 29 focused tests passed; Ruff and mypy passed
- Corrected replay command: `dimos --replay --replay-db data/go2_short.db --viewer none run
  unitree-go2-semantic-map-temporal`
- Corrected map metrics: 224 valid observations, 43 new tracks, 181 accepted fusions, 0 unfused
  observations, 80.80% strict accepted-fusion rate; 32 confirmed objects and 213 confirmed
  observations; association latency 0.62 ms mean / 0.77 ms p95
- Contract metrics: 0 invalid observations and 0 rejected frames
- Runtime metrics: 569 detector calls, 410 2D detections, 224/312 3D survival (71.79%);
  0 detector, lifting, or descriptor failures; one missing transform; descriptor extraction
  4.16 ms mean / 8.23 ms p95
- Temporal result: 0 reported move events on a recording believed to be static
- Geometry result: pose-graph correction disabled, 0 semantic graph corrections; the mapper
  independently reported two geometric loop closures

## E11 — follow-up review consistency and test isolation

- Date: 2026-10-31
- Documentation correction: `PR_DESCRIPTION.md` now uses only E10 values—224 descriptors, 80.80%
  strict accepted fusion, 0.62/0.77 ms association mean/p95, and 4.16/8.23 ms descriptor mean/p95
- Test isolation: moved the Unitree connection import out of detection `conftest.py` collection and
  replaced the focused lifter tests' Go2 connection dependency with a `CameraInfo` test double
- Verification: all 29 focused tests collected and passed with `LD_LIBRARY_PATH` unset; PortAudio
  is no longer required for this focused suite
- Static verification: Ruff passed; production semantic-map and detection source mypy checks remain
  clean. The legacy generic detection `conftest.py` is not itself part of the typed source target.

## E12 — closed- versus open-vocabulary detector comparison

- Date: 2026-10-31
- Code revision: `24c4048876` for the compared implementations; the reproducibility runner was
  added immediately afterward in the E12 documentation commit.
- Method: add interchangeable YOLO-E prompt-free and Grounding DINO detector blueprints, plus
  per-label 2D, 3D, candidate, and confirmed-object counters. All variants retain the same hybrid
  LiDAR lifter, appearance descriptor, temporal tracker, query interface, and replay.
- Dataset: `data/go2_short.db`, SHA-256
  `4f056a042d58ad1dbc8248e408bd61d98bb90e5989cf0520c5400a1088dfe1b9`.
- Fixed sample: `color_image_blob` row IDs
  `50,100,150,200,250,300,350,400,450,500,600,700,800`. Run each detector in a fresh process:
  `uv run python scripts/compare_open_vocab_detectors.py --detector coco-yolo`, then repeat with
  `--detector yoloe` and `--detector grounding-dino`. This host additionally required its extracted
  TurboJPEG directory in `LD_LIBRARY_PATH`; a normal system installation needs no prefix.
- Versions: Python 3.12.12, PyTorch 2.7.1+cu128, Ultralytics 8.4.14, Transformers 4.53.3.
- Models/config: YOLO11n-seg COCO, confidence 0.50 and IoU 0.60, weight SHA-256
  `55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152`; YOLO-E-11s-seg-pf
  prompt-free LRPC, confidence 0.60, IoU 0.60 and maximum area ratio 0.30, weight SHA-256
  `292bdf157a9ec7315f34b567cb93467c5043cd1889a1cc18abbfdeb88d7a948d`; Grounding DINO Tiny
  revision `a2bb814dd30d776dcf7e30523b00659f4f141c71`, confidence 0.25 and class-wise NMS IoU 0.50,
  weight SHA-256 `1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3`, using the 14 prompts
  defined by `GroundingDino2DDetector.DEFAULT_QUERIES`.
- Identical-frame result: on those 13 frames, COCO YOLO produced 7 detections across 5
  labels; prompt-free YOLO-E produced 10 across 9 labels; prompted Grounding DINO produced 118
  across 13 labels. DINO included shelf/carton/display-case/cup/box/bottle/can concepts absent or
  sparse in the COCO result. YOLO-E broadened vocabulary but emitted ambiguous scene labels such as
  `bookstore`, `newsstand`, and `potty`.
- Corrected DINO partial replay: 65 inference calls, 616 2D detections, 189/463 aligned detections
  lifted to 3D (40.82%), 128 candidate tracks, and 42 confirmed objects. Confirmed labels were
  carton 4, chair 9, display case 12, person 1, potted plant 1, shelf 9, and table 6.
- DINO runtime/consistency: 931.35 ms mean / 991.69 ms p95 inference, 0.70 Hz effective processing,
  32.28% strict association, and nearby same-label pairs 5/7/10 at 0.5/0.75/1.0 m. This run was
  stopped after the sensor replay completed while DINO was still draining work, so counts are not a
  full-run recall comparison.
- Download incident: the first Hugging Face/Xet download stalled after producing a full-size but
  checksum-invalid model file. It loaded but returned identical low scores. The rerun above used a
  clean file whose SHA-256 matched the repository ETag
  `1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3`.
- Conclusion: retain YOLO11-seg as the default real-time baseline. Grounding DINO is useful as a
  slower prompted recall/oracle pass but needs frame dropping plus mask generation (for example,
  SAM) before it is a practical online 3D mapper. YOLO-E is the strongest single-stage open-world
  candidate because it preserves masks and near-real-time throughput, but its broad prompt-free
  labels need vocabulary normalization or explicit prompts.

## E13 — take-home clarification follow-up

- Date: 2026-11-01
- Team clarification: automatic first-use pretrained-weight downloads are preferred. Additional
  basic recordings may be available, but no annotated evaluation dataset exists; simulation is an
  acceptable source of controlled evaluation scenarios.
- Runtime change: YOLO-E now first tries the existing DimOS Git-LFS cache and falls back to the
  standard Ultralytics model-name download when the archive, Git LFS, or extraction is unavailable.
  YOLO11-seg, ResNet-18, and Grounding DINO already use first-use download/cache behavior.
- Evaluation policy: retain replay throughput, fusion-survival, association, and duplicate-proxy
  evidence, while reserving recall, identity-switch, and moved-object accuracy claims for controlled
  ground-truth simulation. Existing deterministic tracker tests exercise stationary revisits,
  same-frame exclusivity, global assignment, and moved-object identity as component-level simulated
  scenarios.

## E14 — configurable open vocabulary and DINO + SAM

- Date: 2026-11-01
- Grounding DINO queries are supplied through `Detection2DModule.detector_kwargs` in the blueprint;
  the 14-item grocery vocabulary remains a deterministic default, not a fixed adapter constraint.
- Added an optional Grounding DINO + SAM 2 blueprint. DINO supplies prompted boxes and labels; SAM
  2.1 Tiny refines each box into an instance mask before the unchanged hybrid LiDAR lifter.
- Real replay-frame smoke test (row 300, prompts shelf/bottle/box/bag): 7 detections, all seven were
  `Detection2DSeg` with non-empty masks (3 box, 4 shelf).
- The first SAM run downloads `sam2.1_t.pt`; subsequent runs use the local Ultralytics cache/path.

## E15 — controlled temporal-association evaluation

- Date: 2026-11-01
- Command: `uv run python scripts/evaluate_semantic_tracker.py`.
- Scenario: four same-label, identical-descriptor chairs; small position noise; one occluded chair
  revisited after dormancy; and one uniquely described box moved 2 m after dormancy.
- Ground truth: 5 objects. Result: 100% tracker recall, 0 duplicate tracks, 0 identity switches,
  0 false merges, and 1/1 moved-object re-identification.
- Scope: tracker component only. This deliberately excludes detector and depth errors and therefore
  is not presented as end-to-end perception recall.

## E16 — fixed-sample CPU benchmark

- Date: 2026-11-01
- Command: `uv run python scripts/compare_open_vocab_detectors.py --detector coco-yolo --device cpu`.
- Same 13 frame IDs and database SHA-256 as E12; YOLO11n-seg, PyTorch 2.7.1+cu128,
  Ultralytics 8.4.14, Linux x86-64.
- Result: 257.25 ms cold first frame; 81.11 ms mean, 79.03 ms median, and 95.05 ms p95 warm
  latency; 12.33 Hz warm throughput. The sample produced 7 detections across 5 labels.
- The runner now emits exact frame IDs, database hash, model name, package versions, device, and
  timing statistics in one JSON record.

## E17 — final default replay and submission verification

- Date: 2026-11-01
- Revision: final implementation after configurable prompts, DINO+SAM, simulation evaluation, CPU
  benchmark, and semantic navigation were added.
- Headless command: `dimos --replay --replay-db data/go2_short.db --viewer none run
  unitree-go2-semantic-map-temporal`.
- Perception: 855 input frames over 59.80 s; 571 inference calls; 408 2D detections; 222/321 aligned
  detections lifted to 3D (69.16%); 28.11 ms mean / 42.21 ms p95 inference; no stage failures and
  one startup transform miss.
- Map: 47 candidates, 31 confirmed objects, 222 valid observations, 175 accepted fusions, 47 new
  tracks, 0 unfused observations, 0 invalid observations, and 0 rejected frames. Strict accepted-
  fusion rate was 78.83%; association latency was 0.64 ms mean / 0.87 ms p95.
- Temporal: 0 reported move events on the static replay; pose-graph object correction remained
  disabled. The mapper independently reported two geometric loop closures.
- Navigation: 0 requests during replay, as expected; unit tests verify nearest-object selection,
  1 m standoff goal geometry, missing-pose rejection, and connection to the existing A* planner.
- Fresh Rerun artifact:
  `recordings/20261101-125124-ea4c-unitree-go2-semantic-map-temporal/rerun.rrd`, SHA-256
  `8e70e16ac003a0dbbab34154c204e7002e061505fd90524d8ef966751d5a7e9f`; 112 MiB and verified.
- Verification: 48 focused tests passed. Ruff and production-source mypy checks passed.

## E18 — fair four-detector GPU speed comparison

- Date: 2026-11-01
- Method: four fresh processes, identical E12 database hash and 13 frame IDs, CUDA, stateless
  prediction for every detector. This corrects the earlier comparison in which YOLO11 used its
  stateful tracking path while YOLO-E used stateless prediction.
- Host: NVIDIA GeForce RTX 5070 Ti Laptop GPU, 12,227 MiB, driver 580.178.04; PyTorch 2.7.1+cu128;
  Ultralytics 8.4.14.
- YOLO11n-seg: 7 detections / 5 labels; 717.04 ms cached cold start; 7.41 ms warm mean,
  7.00 ms median, 10.06 ms p95; 134.96 Hz.
- YOLO-E-11s-seg-pf: 10 detections / 9 labels; 1,200.02 ms cached cold start; 28.39 ms warm mean,
  30.21 ms median, 36.41 ms p95; 35.23 Hz.
- Grounding DINO Tiny: 118 detections / 13 labels; 4,209.38 ms cached cold start; 748.44 ms warm
  mean, 750.28 ms median, 782.85 ms p95; 1.34 Hz.
- Grounding DINO Tiny + SAM 2.1 Tiny: the same 118 DINO detections refined into masks; 6,257.01 ms
  cached cold start; 976.06 ms warm mean, 975.94 ms median, 1,082.43 ms p95; 1.02 Hz. The initial
  uncached run additionally spent about 24 seconds downloading the 74.5 MiB SAM weight.
- Interpretation: YOLO11 is the fastest fixed-vocabulary baseline. YOLO-E is 3.83x slower than
  YOLO11 but remains comfortably online and broadens labels. DINO is 100.99x slower than YOLO11;
  adding SAM costs another 30.4% over DINO. Detection totals measure output volume, not accuracy.

## E19 — full supplied-replay pipeline matrix

- Date: 2026-11-01
- Protocol: eight fresh processes at the same revision, each replaying all 855 inputs from
  `data/go2_short.db` (59.80 sensor seconds) with no viewer. Every process received the same 110 s
  wall-time budget and graceful interrupt. Raw logs are represented by the normalized rows in
  `evidence/e19_pipeline_matrix.csv`.
- Box/greedy baseline (`unitree-go2-semantic-map`): YOLO boxes, box LiDAR lifting, no appearance,
  greedy tracker. It produced 325 2D detections, lifted 187/251 aligned detections (74.50%), ran
  inference at 24.26 ms mean / 34.88 ms p95, and created 40 candidates / 26 confirmed objects.
  Its accepted-fusion rate was 78.61%, with nearby same-label pairs 1/6/10.
- Mask/greedy (`...-seg`): YOLO instance masks, mask-only LiDAR lifting, no appearance, greedy
  tracker. It produced 422 detections but lifted only 135/325 (41.54%), confirming 22 objects. The
  mask-only path reduced nearby-pair counts to 0/1/1 but discarded sparse-mask observations.
- Hybrid/greedy (`...-hybrid`): YOLO masks, mask-first lifting with box fallback, no appearance,
  greedy tracker. It lifted 209/301 (69.44%); fallback recovered 63/150 attempts (42.00%). It
  created 46 candidates / 34 confirmed objects, with 77.78% accepted fusion and pairs 1/4/11.
- Hybrid/global assignment (`...-assignment`): the same front end with frame-global one-to-one
  matching. It lifted 196/289 (67.82%), created 45 candidates / 30 confirmed objects, accepted
  76.80% of observations, and had pairs 2/3/10. Association cost was 0.64 ms mean.
- Preferred temporal baseline (`...-temporal`): YOLO masks, hybrid lifting, ResNet-18 appearance,
  and global temporal tracking. It lifted 209/326 (64.11%), created 49 candidates / 33 confirmed,
  accepted 76.21% of observations, and had pairs 2/6/14. Inference was 28.28 ms mean / 42.39 ms
  p95, descriptor extraction 4.48 ms mean / 8.69 ms p95, and association 0.81 ms mean. Unlike the
  simpler trackers, it exposes active/dormant/moved lifecycle and conservative re-identification.
- YOLO-E temporal (`...-temporal-yoloe`): prompt-free open-vocabulary masks with the preferred
  downstream stack. It produced 625 2D detections, lifted 291/470 (61.91%), and created 88
  candidates / 47 confirmed. Inference was 34.69 ms mean / 51.60 ms p95; accepted fusion was
  69.76%; pairs were 1/2/5. Broad but noisy labels included scene/category errors, so the larger
  map is not evidence of higher accuracy.
- Grounding DINO temporal (`...-temporal-grounding-dino`): prompted open-vocabulary boxes with
  hybrid lifting, ResNet-18, and temporal tracking. At 0.89 effective Hz it produced 871 detections,
  lifted 273/674 (40.50%), and created 159 candidates / 72 confirmed. Inference was 660.94 ms mean
  / 665.68 ms p95; accepted fusion was 41.54%; pairs were 5/16/21.
- Grounding DINO + SAM temporal (`...-temporal-grounding-dino-sam`): DINO boxes refined to SAM
  masks. At 0.70 effective Hz it produced 706 detections, lifted 205/534 (38.39%), and created 132
  candidates / 47 confirmed. Inference was 825.93 ms mean / 839.09 ms p95; accepted fusion was
  35.61%; pairs were 1/7/9. SAM masks reduced close duplicate proxies relative to DINO boxes, but
  sparse masks and latency reduced usable observations.
- Selection (superseded by the user-requested E20 default): YOLO11 was initially retained as the
  conservative submission baseline, while YOLO-E was the preferred open-vocabulary alternative.
  The unannotated replay
  cannot establish precision, recall, IDF1/HOTA, or 3D localization accuracy; confirmed-object
  count, association rate, and nearby-pair count are engineering diagnostics, not ground truth.

## E20 — promote YOLO-E to the submission default

- Date: 2026-11-01
- Decision: `unitree-go2-semantic-map-temporal` now selects YOLO-E. The measured E19 YOLO-E row is
  therefore the default's full-replay result. `unitree-go2-semantic-map-temporal-yolo11` preserves
  the faster fixed-vocabulary baseline, and the older `...-temporal-yoloe` name remains an alias.
- Rationale: compared with YOLO11 on the same supplied replay, YOLO-E produced 625 versus 417 2D
  detections, 291 versus 209 3D observations, and 47 versus 33 confirmed tracks while maintaining
  essentially the same effective pipeline rate (5.44 versus 5.43 Hz). It trades higher inference
  latency (34.69 versus 28.28 ms) and lower accepted-fusion rate (69.76% versus 76.21%) for broader
  coverage. These counts do not prove accuracy; ambiguous labels remain a documented limitation.

## E21 — final YOLO-E Rerun artifact

- Date: 2026-11-01
- Command: `RERUNBRIDGEMODULE__RERUN_OPEN=none RERUNBRIDGEMODULE__RERUN_SAVE=1 dimos
  --replay --replay-db data/go2_short.db --viewer rerun run
  unitree-go2-semantic-map-temporal`.
- Artifact: `evidence/yoloe_default_replay.rrd`, stored through Git LFS; 120,732,545 bytes; SHA-256
  `b96b00c157ffbdb0dd1deccf43f5f4afced0ea78e9629a6390eabd44e6422f4c`; `rerun rrd verify`
  passed.
- Replay: 855 inputs, 571 inference calls, 652 2D detections, 312/505 aligned detections lifted to
  3D (61.78%), 90 candidates, 47 confirmed objects, and 71.06% accepted fusion. Inference was
  32.14 ms mean / 44.40 ms p95. Counts vary slightly with asynchronous replay scheduling.
