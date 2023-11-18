import logging
import os
import sys

import pytest

cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(f"{cwd}/.."))
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def pytest_addoption(parser):
    parser.addoption("--fromfile", action="store", default="tests/parts.txt", help="load test data from file")


@pytest.fixture(scope="session")
def fromfile(request):
    fromfile = request.config.option.fromfile
    if fromfile is None:
        pytest.skip()
    return fromfile
