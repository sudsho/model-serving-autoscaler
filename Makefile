.PHONY: install lint test deploy bench clean

install:
	pip install -r requirements.txt

lint:
	ruff check src tests

test:
	pytest tests

deploy:
	bash scripts/deploy.sh

bench:
	locust -f src/load_test.py --headless -u 50 -r 10 -t 2m --host http://localhost:8080

clean:
	rm -rf __pycache__ .pytest_cache *.egg-info
