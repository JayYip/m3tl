.PHONY: test


check_dirs := tests

test:
	pytest --cov-report xml:cov.xml --cov=bert_multitask_learning

commit:
	nbdev_clean_nbs
	nbdev_build_lib

release:
	rm -rf dist/
	python setup.py sdist bdist_wheel
	twine upload dist/*
