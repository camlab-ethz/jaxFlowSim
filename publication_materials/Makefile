# name the dir and stamp file
VENV        := .venv
VENV_GPU    := .venv_gpu
STAMP       := $(VENV)/.touchfile
STAMP_GPU   := $(VENV_GPU)/.touchfile
REQ         := ../requirements.txt
REQ_GPU     := ../requirements_gpu.txt

.PHONY: docs venv venv_gpu test clean

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

# “venv_gpu” target now depends on the .venv_gpu/.touchfile
venv_gpu: $(STAMP_GPU)

# rebuild whenever requirements change
$(STAMP_GPU): $(REQ_GPU)
	# create venv if missing
	test -d $(VENV_GPU) || python -m venv $(VENV_GPU)
	# install into it
	. $(VENV_GPU)/bin/activate && pip install -Ur $(REQ_GPU)
	# update timestamp marker
	touch $(STAMP_GPU)

# setup and run example driver
example_driver: venv_gpu
	# run example driver
	. $(VENV_GPU)/bin/activate && python ../demos/probe_potential_surface.py

# setup and run tests
test: venv
	# run tests
	. $(VENV)/bin/activate && python ../test/test_models.py publication_materials/test_data

# generate demos
docs: venv
	. $(VENV)/bin/activate && cd ../docs/ && make latexpdf; make latexpdf
	echo "copy generated docs pdf to working directory"
	cp ../docs/_build/latex/jaxflowsim.pdf .

# clean working directory
clean:
	rm -rf $(VENV)
	rm -rf jaxflowsim.pdf
	find . -iname "*.pyc" -delete
	cd ../docs/ && make clean
