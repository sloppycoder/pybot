import os
import sys

import pytest

cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(f"{cwd}/.."))


def pytest_addoption(parser):
    parser.addoption("--fromfile", action="store", default="tests/parts.txt", help="load test data from file")
    parser.addoption(
        "--batch",
        action="store",
        default="1",
        help="batch, the suffix to input and output file to use during test runs",
    )


@pytest.fixture(scope="session")
def fromfile(request):
    fromfile = request.config.option.fromfile
    if fromfile is None:
        pytest.skip()
    return fromfile


@pytest.fixture(scope="session")
def batch(request):
    batch = request.config.option.batch
    if batch is None:
        pytest.skip()
    return batch
