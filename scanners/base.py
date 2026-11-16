"""
Base Scanner Adapter & Biometric Exceptions
===========================================
Defines the abstract interface and standard status contracts that every scanner
implementation (hardware vendor SDKs and simulated sensors) must fulfill.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Optional
from PIL import Image


class ScannerStatus(str, Enum):
    READY = "ready"                        # Device connected and SDK ready to capture
    DISCONNECTED = "disconnected"          # Physical USB device not detected
    DRIVER_MISSING = "driver_missing"      # Hardware detected but vendor SDK/DLL missing
    UNVERIFIED = "unverified"              # Hardware detected but model/vendor unconfirmed
    BUSY = "busy"                          # Active capture currently in progress
    SIMULATED = "simulated"                # Virtual sensor in test mode
    ERROR = "error"                        # Device fault or communication failure


class ScannerException(Exception):
    """Base exception for all scanner operations."""
    def __init__(self, message: str, code: str = "SCANNER_ERROR"):
        super().__init__(message)
        self.code = code
        self.message = message


class ScannerNotConnectedError(ScannerException):
    """Raised when an operation is requested on a disconnected scanner."""
    def __init__(self, message: str = "USB Fingerprint Scanner is disconnected."):
        super().__init__(message, code="NOT_CONNECTED")


class DriverMissingError(ScannerException):
    """Raised when physical hardware is present but vendor SDK / driver DLL is missing."""
    def __init__(self, message: str = "Vendor driver or SDK library is not installed."):
        super().__init__(message, code="DRIVER_MISSING")


class ScannerBusyError(ScannerException):
    """Raised when a capture is requested while the device is already busy."""
    def __init__(self, message: str = "Scanner is currently busy with another operation."):
        super().__init__(message, code="DEVICE_BUSY")


class ScannerCaptureError(ScannerException):
    """Raised when image capture fails or times out."""
    def __init__(self, message: str = "Fingerprint capture failed."):
        super().__init__(message, code="CAPTURE_FAILED")


class ScannerCancelledError(ScannerException):
    """Raised when an ongoing scan operation is explicitly cancelled by user."""
    def __init__(self, message: str = "Fingerprint scan operation was cancelled."):
        super().__init__(message, code="SCAN_CANCELLED")


class BaseScannerAdapter(ABC):
    """
    Abstract Base Class for modular fingerprint scanner adapters.
    Guarantees consistent device lifecycle, status reporting, and capture interfaces.
    """

    @property
    @abstractmethod
    def adapter_id(self) -> str:
        """Unique identifier, e.g. 'scanner_1', 'scanner_2', 'simulated'."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable display title, e.g. 'Scanner 1 (Primary USB Slot)'."""
        pass

    @property
    @abstractmethod
    def slot_number(self) -> int:
        """Slot number (1, 2, or 0 for simulated)."""
        pass

    @abstractmethod
    def is_hardware_connected(self) -> bool:
        """Returns True if the physical USB hardware is detected on the bus."""
        pass

    @abstractmethod
    def is_sdk_available(self) -> bool:
        """Returns True if the required vendor driver / SDK runtime is available."""
        pass

    @abstractmethod
    def get_status(self) -> ScannerStatus:
        """Returns current device status enum."""
        pass

    @abstractmethod
    def get_telemetry(self) -> Dict[str, Any]:
        """
        Returns structured telemetry dictionary:
        {
            'adapter_id': str,
            'display_name': str,
            'slot': int,
            'status': str,
            'status_label': str,
            'is_connected': bool,
            'is_biometric': bool,
            'sdk_available': bool,
            'vendor_name': str,
            'model_name': str,
            'vid': str,
            'pid': str,
            'can_capture': bool,
            'integration_notes': str
        }
        """
        pass

    @abstractmethod
    def capture(self, timeout_seconds: float = 10.0) -> Dict[str, Any]:
        """
        Capture fingerprint image from sensor and return normalized payload:
        {
            'success': True,
            'image_pil': PIL.Image.Image,
            'image_base64': str (data URL),
            'raw_dimensions': tuple (W, H),
            'normalized_dimensions': tuple (512, 512),
            'quality_score': float (0-100),
            'capture_time_ms': float,
            'scanner_id': str
        }
        """
        pass

    def cancel(self) -> bool:
        """Cancel an ongoing capture if supported by the adapter."""
        return False

    @abstractmethod
    def release(self) -> None:
        """Cleanly releases any device handles, memory, or SDK sessions."""
        pass

