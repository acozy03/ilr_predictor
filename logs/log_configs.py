import logging

#logging setup
logging.basicConfig(
    level=logging.INFO,  # Use DEBUG for verbose logs
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log"),      # Log to file
        logging.StreamHandler()                   # Log to console
    ]
)

logger = logging.getLogger(__name__)
