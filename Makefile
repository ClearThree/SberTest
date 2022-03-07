# Makefile
build:
	docker build -t automl .

run:
	docker run automl

format:
	isort automl
	black automl
	pflake8 automl
