#!/bin/bash
# Remove pipenv environment
pipenv --rm

# Remove project
cd ../
rm -rf MIL-COVIDNET/

echo -e '[CORRECT] Project removed.'
