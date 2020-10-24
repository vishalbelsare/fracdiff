#!/bin/sh

python3 -m pytest --doctest-modules fracdiff
python3 -m pytest --doctest-modules tests

python3 -m flake8 fracdiff
python3 -m black --check fracdiff || read -p "Run formatter? (y/N): " yn; [[ $yn = [yY] ]] && python3 -m black fracdiff
python3 -m isort --check --force-single-line-imports fracdiff || read -p "Run formatter? (y/N): " yn; [[ $yn = [yY] ]] && python3 -m isort --force-single-line-imports fracdiff
