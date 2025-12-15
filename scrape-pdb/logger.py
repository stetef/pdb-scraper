#!/usr/bin/env python3
"""Logging configuration for PDB Phase 1 pipeline."""

import logging
import sys
from pathlib import Path
from typing import Optional
from tqdm import tqdm


class TqdmLoggingHandler(logging.Handler):
    """Custom handler that uses tqdm.write() to avoid breaking progress bars."""
    
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)
    
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)

def setup_logger(
    name: str = "pipeline",
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    console: bool = True
) -> logging.Logger:
    """..."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    # Console handler - use TqdmLoggingHandler instead of StreamHandler
    if console:
        ch = TqdmLoggingHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    return logger


class PipelineLogger:
    """Context manager for pipeline logging with structured sections."""
    
    def __init__(self, logger: logging.Logger, pdb_id: str):
        self.logger = logger
        self.pdb_id = pdb_id
        self.clusters_written = 0
    
    def __enter__(self):
        self.logger.info("="*60)
        self.logger.info(f"Processing PDB: {self.pdb_id}")
        self.logger.info("="*60)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.logger.info(f"Completed {self.pdb_id}: {self.clusters_written} clusters written")
        else:
            self.logger.error(f"Failed processing {self.pdb_id}: {exc_val}")
        self.logger.info("")
        return False  # Don't suppress exceptions
    
    def log_clusters(self, count: int):
        """Log cluster count."""
        self.clusters_written = count
        self.logger.info(f"  Clusters found: {count}")
    
    def log_metals(self, count: int, components: int):
        """Log metal statistics."""
        self.logger.info(f"  Metals detected: {count}")
        self.logger.info(f"  Connected components: {components}")
    
    def log_altloc(self, case: str, label: str):
        """Log altloc handling."""
        self.logger.debug(f"  AltLoc case {case}, label: {label}")