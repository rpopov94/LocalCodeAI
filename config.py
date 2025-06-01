"""Project config."""
import os

from dotenv import load_dotenv

load_dotenv()

DOC_SOURSES = os.getenv('DOC_SOURCES', None)