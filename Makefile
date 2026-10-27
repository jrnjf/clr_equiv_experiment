# Uses the venv interpreter directly. No `activate` script is required (this
# machine's venv was created without activation scripts because
# python3.14-venv is not installed; .venv/bin/python works regardless).
PY ?= .venv/bin/python

.PHONY: help test dry check tiny cifar clean

help:
	@echo "Interpreter: $(PY)"
	@echo "make test    - unit tests (torch-free)"
	@echo "make dry     - validate both configs, torch-free"
	@echo "make check   - PREREQUISITE: internal-vs-external parameter equivalence (tiny, CPU)"
	@echo "make tiny    - smoke-run all 4 arms on tiny backend (CPU, seconds)"
	@echo "make cifar   - real run on CIFAR-10/ResNet-18 (GPU)"

test:
	$(PY) -m unittest discover -s tests -v

dry:
	$(PY) experiments/run_equivalence.py --config configs/tiny.json --dry-run
	$(PY) experiments/run_equivalence.py --config configs/cifar.json --dry-run

check:
	$(PY) experiments/run_equivalence.py --config configs/tiny.json --check-equivalence

tiny:
	$(PY) experiments/run_equivalence.py --config configs/tiny.json

cifar:
	$(PY) experiments/run_equivalence.py --config configs/cifar.json

clean:
	rm -rf results/*.csv results/*.json __pycache__ src/__pycache__ tests/__pycache__
