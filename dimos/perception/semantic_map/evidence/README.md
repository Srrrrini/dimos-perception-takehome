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

The full recording is intentionally not committed because it is 112 MiB. It was generated at:

```text
recordings/20261101-125124-ea4c-unitree-go2-semantic-map-temporal/rerun.rrd
```

Integrity:

```text
SHA-256: 8e70e16ac003a0dbbab34154c204e7002e061505fd90524d8ef966751d5a7e9f
Rerun verification: 1 file verified without error
Size: 112 MiB
```

Regenerate it from the repository root with:

```bash
RERUNBRIDGEMODULE__RERUN_OPEN=none \
RERUNBRIDGEMODULE__RERUN_SAVE=1 \
uv run dimos --replay --replay-db data/go2_short.db --viewer rerun \
  run unitree-go2-semantic-map-temporal
```
