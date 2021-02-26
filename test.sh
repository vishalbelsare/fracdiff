#!/bin/sh -eu

python3 -m pytest --doctest-modules fracdiff
python3 -m pytest --doctest-modules tests

python3 -m flake8 fracdiff
python3 -m black --check --quiet fracdiff || read -p "Run black? (y/n): " yn; [[ $yn = [yY] ]] && python3 -m black --quiet fracdiff
python3 -m isort --check --force-single-line-imports fracdiff || read -p "Run isort? (y/n): " yn; [[ $yn = [yY] ]] && python3 -m isort --force-single-line-imports --quiet fracdiff
python3 -m black --quiet tests
python3 -m isort --force-single-line-imports --quiet tests
