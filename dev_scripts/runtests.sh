#!/bin/bash
set -e  # exit on first error, print each command

abinit --version
abinit --build
abicheck.py --with-flow

#nosetests -v --with-coverage --cover-package=abiflows --logging-level=INFO --doctest-tests
#pytest -n 2 --cov-config=.coveragerc --cov=abiflows -v --doctest-modules
pytest --cov-config=.coveragerc --cov=abiflows -v --doctest-modules abiflows --ignore=abiflows/fireworks/integration_tests 

# This is to run the integration tests (append results)
# integration_tests are excluded in setup.cfg
if [[ "${ABIPY_COVERALLS}" == "yes" ]]; then 
    #pytest -n 2 --cov-config=.coveragerc --cov=abiflows --cov-append -v abiflows/fireworks/integration_tests 
    pytest --cov-config=.coveragerc --cov=abiflows --cov-append -v abiflows/fireworks/integration_tests 
fi

# Generate documentation
if [[ "${ABIPY_SPHINX}" == "yes" ]]; then
    pip install -r ./docs/requirements.txt
    cd ./docs && make && cd ..
fi
