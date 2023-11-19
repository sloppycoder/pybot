import logging
import os
import sys

import pytest

cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(f"{cwd}/.."))

logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)

logging.getLogger("bot").setLevel(logging.getLevelName(os.environ.get("LOG_LEVEL", "info").upper()))


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
