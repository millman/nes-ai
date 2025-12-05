# nes-ai

## Experiment Viewer

Run the JEPA experiment viewer to inspect runs saved under `out.jepa_world_model_trainer`:

```bash
python -m web_viewer.cli --host 127.0.0.1 --port 5001
```

Visit `http://127.0.0.1:5001/` for the experiment grid or `http://127.0.0.1:5001/comparison` for the A/B comparison workspace.
