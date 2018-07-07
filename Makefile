TEST_CMD = python3 -m doctest
CHECKSTYLE_CMD = flake8
TEST_FILES = $(filter-out main.py, $(wildcard *.py))

all:  compile test checkstyle

call: clearscreen compile test checkstyle

clearscreen:
			clear

compile:
	    @echo "Nothing to compile for Python"

test:
	    $(TEST_CMD) $(TEST_FILES)

checkstyle:
	    $(CHECKSTYLE_CMD) *.py

clean:
	    rm -f *.pyc
			rm -rf __pycache__
			rm -f total_rating.csv
			rm -f training_file_*.csv
			rm -f ratings_split_*.csv
