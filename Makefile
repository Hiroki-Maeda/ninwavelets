all:
	python setup_cy.py build_ext --inplace
run:
	python setup_cy.py build_ext --inplace
	python -m nin_wavelets.test

