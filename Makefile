.PHONY: test


check_dirs := tests

test:
	pytest --cov-report xml:cov.xml --cov=m3tl

nbbuild:
	nbdev_build_lib
	nbdev_build_docs

commit:
	nbdev_read_nbs
	nbdev_clean_nbs
	nbdev_diff_nbs
	nbdev_test_nbs

check:
	nbdev_read_nbs
	nbdev_clean_nbs
	nbdev_diff_nbs

release:
	rm -rf dist/
	python setup.py sdist bdist_wheel
	twine upload dist/*
