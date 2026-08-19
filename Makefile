.PHONY: install lint test smoke deploy bench clean

install:
	pip install -r requirements.txt

lint:
	ruff check src tests

test:
	pytest tests

# Offline end-to-end smoke: boots the router in-process against a synthetic
# predictor and simulates the autoscaler control loop. No cluster/GPU/network.
smoke:
	python scripts/smoke.py

deploy:
	bash scripts/deploy.sh

bench:
	locust -f src/load_test.py --headless -u 50 -r 10 -t 2m --host http://localhost:8080

clean:
	rm -rf __pycache__ .pytest_cache *.egg-info
