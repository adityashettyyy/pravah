.PHONY: all fast dashboard demo test
all:       ; python run_all.py
fast:      ; python run_all.py --fast
test:      ; python -m pytest -q tests
demo:      ; python pipeline.py --hub "Bhiwandi" --severity 0.8 --out result.json
dashboard: ; streamlit run app/dashboard.py
