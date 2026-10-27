.PHONY: help test dry tiny check cifar clean

help:
	@echo "make test    - unit tests (torch-free)"
	@echo "make dry     - validate both configs, no torch needed"
	@echo "make check   - PREREQUISITE: internal-vs-external parameter equivalence (tiny, CPU)"
	@echo "make tiny    - smoke-run all 4 arms on tiny backend (CPU, seconds)"
	@echo "make cifar   - real run on CIFAR-10/ResNet-18 (GPU)"

test:
	python3 -m unittest discover -s tests -v

dry:
	python3 experiments/run_equivalence.py --config configs/tiny.json --dry-run
	python3 experiments/run_equivalence.py --config configs/cifar.json --dry-run

check:
	python3 experiments/run_equivalence.py --config configs/tiny.json --check-equivalence

tiny:
	python3 experiments/run_equivalence.py --config configs/tiny.json

cifar:
	python3 experiments/run_equivalence.py --config configs/cifar.json

clean:
	rm -rf results/*.csv results/*.json __pycache__ src/__pycache__ tests/__pycache__
