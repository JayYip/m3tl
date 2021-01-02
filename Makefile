.PHONY: test


check_dirs := tests

test:
	coverage run -m pytest --rootdir /data/bert-multitask-learning tests
	coverage xml -o cov.xml -i