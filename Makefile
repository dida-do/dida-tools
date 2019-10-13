SHELL:=/bin/bash
TEST=pytest
LINT=pylint
TESTFLAGS= -v

.PHONY: help env env-dev env-remove doc clean-doc clean test lint docker-build docker-run

help:
	@echo "env 		- create the conda environment 'dl-repo' based on environment.yml"
	@echo "env-dev		- runs 'make env' and installs additional development dependencies"
	@echo "env-remove 	- remove 'dl-repo'"
	@echo "doc 		- build sphinx docs, requires 'dl-repo'"
	@echo "clean-doc 	- remove binaries created by sphinx"
	@echo "clean	- remove all compiled bytecode and doc binaries"
	@echo "test		- run unit tests"
	@echo "lint		- run linter for all modules"
	@echo "docker-build	- build docker image"
	@echo "docker-run	- run docker container"

setup:
	$(SHELL) setup.sh

env:
	conda env create -f environment.yml

env-dev:
	conda env create -f environment.yml || true
	conda env update -f dev.yml 

env-remove:
	conda remove --name dl-repo-env --all

doc:
	mkdir -p docs/source/_static || true
	$(MAKE) -C docs/ html

clean-doc:
	$(MAKE) -C docs/ clean || true

clean:
	$(MAKE) clean-doc
	py3clean . && rm -rf .pytest_cache

test:
	$(TEST) $(TESTFLAGS)

lint:
	find . -iname "*.py" | xargs $(LINT)
    
docker-build:
	docker build . -t dl-repo
    
docker-run:
	docker run -it dl-repo
