venv:
	python3 -m venv venv
	./venv/bin/python3 -m pip install --upgrade pip

.PHONY: setup
setup: venv
	./venv/bin/pip3 install -r requirements.txt

.PHONY: setup-conda-env
setup-conda-env:
	conda env create -f environment.yml

.PHONY: format
format:
	./venv/bin/black ./ibat
	./venv/bin/black ./examples

.PHONY: lint
lint:
	./venv/bin/flake8 ./ibat

.PHONY: build-wheel
build-wheel:
	rm -rf dist
	rm -rf ibat.egg-info
	rm -rf build
	python setup.py sdist bdist_wheel

.PHONY: clean
clean:
	rm -rf venv
