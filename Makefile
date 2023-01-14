all:

install-depends:
	grep -q "^deb:" DEPENDS && \
		sudo apt install $$(grep "^deb: " DEPENDS | cut -d" " -f2) || true
	grep -q "^pip:" DEPENDS && \
		pip3 install $$(grep "^pip: " DEPENDS | cut -d" " -f2) || true

# to run a single test, do someting like:
#  make tests T=tests:DatasetProjectTests.test_roman_to_number
tests: T = tests.py
tests:
	nosetests3 $(T)
