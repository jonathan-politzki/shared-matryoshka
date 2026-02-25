PYTHON ?= python3

.PHONY: install data train eval all clean

install:
	pip install -e ".[dev]"

data:
	$(PYTHON) scripts/generate_data.py --config configs/base.yaml

train:
	@for cfg in configs/v3_contrastive.yaml configs/v3_mse.yaml configs/v3_no_prefix.yaml \
		configs/single_dating.yaml configs/single_hiring.yaml \
		configs/projection_heads.yaml configs/adversarial.yaml; do \
		echo "Training $$cfg..."; \
		$(PYTHON) scripts/train.py --config $$cfg; \
	done

eval:
	@for cfg in configs/v3_contrastive.yaml configs/v3_mse.yaml configs/v3_no_prefix.yaml \
		configs/single_dating.yaml configs/single_hiring.yaml \
		configs/projection_heads.yaml configs/adversarial.yaml; do \
		echo "Evaluating $$cfg..."; \
		$(PYTHON) scripts/evaluate.py --config $$cfg; \
	done

all:
	$(PYTHON) scripts/run_all.py

test:
	pytest tests/ -v

clean:
	rm -rf data/ checkpoints/ results/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
