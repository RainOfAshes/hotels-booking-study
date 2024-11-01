
.PHONY: install shell jupyter

# Install dependencies from Poetry
install:
	poetry install

# Start Poetry shell
shell:
	poetry shell

# Run Jupyter Notebook
jupyter:
	poetry run jupyter lab
