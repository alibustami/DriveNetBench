SHELL := /bin/bash

.ONE_SHELL:
create-venv:
	@echo "START: Creating drivenetbench-env virtual environment" && \
	conda create -n drivenetbench-env python=3.8 -y && \
	conda run -n drivenetbench-env python -m pip install --upgrade pip && \
	conda run -n drivenetbench-env python -m pip install --upgrade setuptools wheel && \
	conda run -n drivenetbench-env pip install -e . && \
	conda run -n drivenetbench-env pre-commit install


remove-venv:
	@echo "START: Removing drivenetbench-env virtual environment" && \
	source $$(conda info --base)/etc/profile.d/conda.sh && \
	conda activate base && \
	conda env remove -n drivenetbench-env -y
