.PHONY: test


check_dirs := tests

test:
	coverage run -m pytest --rootdir /data/bert-multitask-learning tests
	coverage xml -o cov.xml -i

commit:
	nbdev_clean_nbs
	nbdev_build_lib

release:
	rm -rf dist/
	python setup.py sdist bdist_wheel
	twine upload dist/*
