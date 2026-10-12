.PHONY: all fast dashboard demo test api sweep docker
all:       ; python run_all.py
fast:      ; python run_all.py --fast
test:      ; python -m pytest -q tests
demo:      ; python pipeline.py --hub "Bhiwandi" --severity 0.8 --out result.json
dashboard: ; streamlit run app/dashboard.py
api:       ; uvicorn api.main:app --reload
sweep:     ; python -m evaluation.sweep
docker:    ; docker build -t pravah .
