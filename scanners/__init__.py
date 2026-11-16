"""
Biometric USB Fingerprint Scanner Adapter Architecture
======================================================
Provides modular abstraction, USB hardware detection, simulated testing mode,
and vendor SDK integration hooks for multi-device fingerprint capture.
"""

from .base import (
    BaseScannerAdapter,
    ScannerStatus,
    ScannerException,
    ScannerNotConnectedError,
    DriverMissingError,
    ScannerCaptureError,
    ScannerBusyError,
    ScannerCancelledError
)
from .usb_detector import USBDetector, USBDeviceInfo
from .normalizer import normalize_fingerprint_image
from .simulated_adapter import SimulatedScannerAdapter
from .scanner1_adapter import Scanner1Adapter
from .scanner2_adapter import Scanner2Adapter
from .scanner_manager import ScannerManager, get_scanner_manager

__all__ = [
    'BaseScannerAdapter',
    'ScannerStatus',
    'ScannerException',
    'ScannerNotConnectedError',
    'DriverMissingError',
    'ScannerCaptureError',
    'USBDetector',
    'USBDeviceInfo',
    'normalize_fingerprint_image',
    'SimulatedScannerAdapter',
    'Scanner1Adapter',
    'Scanner2Adapter',
    'ScannerManager',
    'get_scanner_manager'
]
