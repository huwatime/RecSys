TEST_CMD = python3 -m doctest
TEST_FILES = $(filter-out evaluate.py recommend.py, $(wildcard *.py))

all:  compile test

call: clearscreen compile test

clearscreen:
			clear

compile:
	    @echo "Nothing to compile for Python"

test:
	    $(TEST_CMD) $(TEST_FILES)

clean:
	    rm -f *.pyc
			rm -rf __pycache__
			rm -f total_rating.csv
			rm -f training_file_*.csv
			rm -f ratings_split_*.csv
