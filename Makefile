.ONE_SHELL:
create-venv:
	@echo "START: Creating drivenetbench-env virtual environment" && \
	conda env create -f environment.yml && \
	source $$(conda info --base)/etc/profile.d/conda.sh && \
	conda activate drivenetbench-env && \
	python -m pip install --upgrade pip && \
	python -m pip install --upgrade setuptools wheel && \
	pre-commit install && \
	pip install -e .

remove-venv:
	@echo "START: Removing drivenetbench-env virtual environment" && \
	conda activate base && \
	conda env remove -n drivenetbench-env -y
