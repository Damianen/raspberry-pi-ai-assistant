# Desk Assistant

A local AI desk assistant on a Raspberry Pi 5 (Pironman 5 Pro Max). Pixel eyes on
the touchscreen, tap to talk. Local commands (alarm/reminder/timer) work offline;
open questions go to a cheap cloud LLM via OpenRouter.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp config.example.toml config.toml      # then edit it
export OPENROUTER_API_KEY=sk-...         # for the LLM fallback
python run.py
```

## Develop / test the face without hardware
```bash
SDL_VIDEODRIVER=dummy python -m assistant.ui   # headless smoke test
python -m pytest -q                            # intent parser tests
```

See **CLAUDE.md** for the architecture, the state contract, and the slice-by-slice
build plan. Build and test each slice on the Pi before moving to the next.
