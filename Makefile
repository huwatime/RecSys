TEST_CMD = python3 -m doctest
CHECKSTYLE_CMD = flake8

all:  compile test checkstyle

call: clearscreen compile test checkstyle

clearscreen:
			clear

compile:
	    @echo "Nothing to compile for Python"

test:
	    $(TEST_CMD) *.py

checkstyle:
	    $(CHECKSTYLE_CMD) *.py

clean:
	    rm -f *.pyc
			    rm -rf __pycache__
