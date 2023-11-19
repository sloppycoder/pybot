import logging
import os
import sys

print("===init logger===")
logging.basicConfig(
    stream=sys.stdout,
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)


# module level logger, to be used by other components in the same module
log = logging.getLogger(__name__)
# log.setLevel(os.environ.get("LOG_LEVEL", "INFO"))
