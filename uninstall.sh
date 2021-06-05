#!/bin/bash
# Remove pipenv environment
pipenv --rm

# Remove project
cd ../
rm -rf COVIDNET/

echo -e '[CORRECT] Project removed.'
