# name the dir and stamp file
VENV        := .venv
STAMP       := $(VENV)/.touchfile
REQ         := ../requirements_gpu.txt

.PHONY: venv test clean

# “venv” target now depends on the .venv/.touchfile
venv: $(STAMP)

# rebuild whenever requirements change
$(STAMP): $(REQ)
	# create venv if missing
	test -d $(VENV) || python -m venv $(VENV)
	# install into it
	. $(VENV)/bin/activate && pip install -Ur $(REQ)
	# update timestamp marker
	touch $(STAMP)

example_driver: venv
	# run example driver
	. $(VENV)/bin/activate && python ../demos/probe_potential_surface.py

# clean working directory
clean:
	rm -rf $(VENV)
	find . -iname "*.pyc" -delete
