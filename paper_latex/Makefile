PAPER := paper.pdf

all: $(PAPER)

$(PAPER): paper.tex figures/*.png
	latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex

clean:
	latexmk -c

.PHONY: all clean
