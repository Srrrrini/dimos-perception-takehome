# Replay evidence

![Final temporal semantic-map replay](final_replay.png)

![Temporal blueprint graph](temporal_blueprint.svg)

The screenshot shows the temporal pipeline replaying the Go2 recording with its YOLO11 baseline
detector. The left side contains the camera stream, odometry, and mapper/pose-graph timing; the
right side shows the accumulated LiDAR map and persistent labeled 3D semantic objects. The default
was subsequently promoted to YOLO-E in E20; its complete headless replay metrics are recorded in
the matrix below.

The graph is generated directly by `dimos graph unitree-go2-semantic-map-temporal --rpc`. It
verifies the blueprint's module, channel, and RPC wiring using the same registry entry used by
`dimos run`.

[`e19_pipeline_matrix.csv`](e19_pipeline_matrix.csv) contains the normalized full-replay metrics
for all eight registered semantic-map pipeline variants. The protocol and interpretation are
documented under E19 in [`../EXPERIMENTS.md`](../EXPERIMENTS.md).

The final YOLO-E default recording is committed through Git LFS at:

```text
dimos/perception/semantic_map/evidence/yoloe_default_replay.rrd
```

Integrity:

```text
SHA-256: b96b00c157ffbdb0dd1deccf43f5f4afced0ea78e9629a6390eabd44e6422f4c
Rerun verification: 1 file verified without error
Size: 120,732,545 bytes
```

Reviewers who clone with Git LFS receive the recording automatically. From an existing clone, run
`git lfs pull`, then open the file with Rerun. Reviewers can also use the recording link in the PR
while signed into an account with access to the private repository.

Regenerate it from the repository root with:

```bash
RERUNBRIDGEMODULE__RERUN_OPEN=none \
RERUNBRIDGEMODULE__RERUN_SAVE=1 \
uv run dimos --replay --replay-db data/go2_short.db --viewer rerun \
  run unitree-go2-semantic-map-temporal
```
