scripts := $(shell find . -maxdepth 1 -type f -name '*.py')
notebooks := $(patsubst %.py, %.ipynb, $(scripts))
html := $(patsubst %.py, %.html, $(scripts))

.PHONY: all notebooks html clean

all: notebooks

notebooks: $(notebooks)

html: $(html)

%.ipynb: %.py
	jupytext $< -o $@

%.html: %.ipynb
	jupyter nbconvert --execute --to html $<

clean:
	rm -rf *.ipynb .ipynb_checkpoints
