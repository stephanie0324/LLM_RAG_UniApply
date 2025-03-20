import logging

from config import settings

logging.basicConfig(
    format=settings.LOG_FORMAT,
    datefmt=settings.LOG_DATE_FORMAT,
    level=settings.LOG_LEVEL,
)
logger = logging.getLogger(__name__)
