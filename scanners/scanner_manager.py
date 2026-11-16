"""
Scanner Manager: Device Selection, Hot-Switching, and Concurrency Orchestration
================================================================================
Orchestrates active scanner selection between Scanner 1, Scanner 2, and Simulated mode.
Provides thread-safe capture locking, hot device switching without web server restarts,
and unified hardware telemetry reporting.
"""

import threading
import time
from typing import Dict, Any, List, Optional

from .base import (
    BaseScannerAdapter,
    ScannerStatus,
    ScannerException,
    ScannerNotConnectedError,
    DriverMissingError,
    ScannerCaptureError,
    ScannerBusyError
)
from .usb_detector import USBDetector, USBDeviceInfo
from .simulated_adapter import SimulatedScannerAdapter
from .scanner1_adapter import Scanner1Adapter
from .scanner2_adapter import Scanner2Adapter


class ScannerManager:
    """
    Central Manager for Multi-Scanner Biometric Acquisition.
    Thread-safe singleton managing device adapters, hot switching, and capture queues.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if getattr(self, '_initialized', False):
            return

        self._adapters: Dict[str, BaseScannerAdapter] = {
            'simulated': SimulatedScannerAdapter(),
            'scanner_1': Scanner1Adapter(),
            'scanner_2': Scanner2Adapter()
        }
        self._active_scanner_id = 'simulated'
        self._capture_lock = threading.Lock()
        self._cancel_event = threading.Event()
        self._is_capturing = False
        self._initialized = True

    @property
    def active_scanner_id(self) -> str:
        return self._active_scanner_id

    @property
    def active_adapter(self) -> BaseScannerAdapter:
        return self._adapters.get(self._active_scanner_id, self._adapters['simulated'])

    def select_scanner(self, scanner_id: str) -> Dict[str, Any]:
        """
        Switches the active scanner slot without restarting the Flask server.
        Safely releases previous device session and activates the requested adapter.
        """
        if scanner_id not in self._adapters:
            raise ScannerException(
                f"Unknown scanner ID '{scanner_id}'. Available slots: {list(self._adapters.keys())}"
            )

        with self._capture_lock:
            if self._is_capturing:
                raise ScannerBusyError("Cannot switch scanners while a capture is currently in progress.")

            # Gracefully release previous adapter if necessary
            old_adapter = self._adapters.get(self._active_scanner_id)
            if old_adapter and old_adapter.adapter_id != scanner_id:
                try:
                    old_adapter.release()
                except Exception:
                    pass

            self._active_scanner_id = scanner_id
            active = self._adapters[scanner_id]

        return {
            'success': True,
            'active_scanner_id': self._active_scanner_id,
            'telemetry': active.get_telemetry()
        }

    def get_status_overview(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Returns full telemetry overview for all scanner slots and USB bus peripherals.
        """
        # Query USB devices
        all_usb_devs = USBDetector.detect_devices(force_refresh=force_refresh)
        biometric_candidates = [d.to_dict() for d in all_usb_devs if d.is_biometric_candidate]
        unsupported_peripherals = [d.to_dict() for d in all_usb_devs if not d.is_biometric_candidate]

        # Gather adapter status
        slots_telemetry = []
        for sid in ['scanner_1', 'scanner_2', 'simulated']:
            adapter = self._adapters[sid]
            t = adapter.get_telemetry()
            t['is_active'] = (sid == self._active_scanner_id)
            slots_telemetry.append(t)

        return {
            'active_scanner_id': self._active_scanner_id,
            'slots': slots_telemetry,
            'hardware_summary': {
                'biometric_devices_detected': len(biometric_candidates),
                'total_usb_peripherals': len(all_usb_devs),
                'biometric_candidates': biometric_candidates,
                'unsupported_peripherals': unsupported_peripherals
            }
        }

    def cancel_current_scan(self) -> Dict[str, Any]:
        """Signals cancellation to the currently running capture operation."""
        self._cancel_event.set()
        active = self.active_adapter
        if active:
            try:
                active.cancel()
            except Exception:
                pass
        return {
            'success': True,
            'message': 'Scan cancellation signal sent.',
            'active_scanner_id': self._active_scanner_id
        }

    def capture_fingerprint(
        self,
        scanner_id: Optional[str] = None,
        timeout_seconds: float = 10.0
    ) -> Dict[str, Any]:
        """
        Executes live fingerprint capture on the specified or currently active scanner.
        Thread-safe, handles device disconnects, driver missing exceptions, and cancellation.
        """
        target_id = scanner_id or self._active_scanner_id
        if target_id not in self._adapters:
            raise ScannerException(f"Invalid scanner ID '{target_id}'.")

        adapter = self._adapters[target_id]

        # Reset cancel event before new capture
        self._cancel_event.clear()

        # Acquire capture lock
        acquired = self._capture_lock.acquire(timeout=timeout_seconds + 2.0)
        if not acquired:
            raise ScannerBusyError("Scanner lock timeout. Device is currently busy.")

        try:
            self._is_capturing = True
            try:
                # Pass cancel_event to adapter capture if supported
                result = adapter.capture(timeout_seconds=timeout_seconds, cancel_event=self._cancel_event)
            except TypeError:
                result = adapter.capture(timeout_seconds=timeout_seconds)
            return result
        finally:
            self._is_capturing = False
            self._capture_lock.release()

    def release_all(self) -> None:
        """Releases all active device handles."""
        for adapter in self._adapters.values():
            try:
                adapter.release()
            except Exception:
                pass


# Global singleton instance access
_manager_instance = None

def get_scanner_manager() -> ScannerManager:
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = ScannerManager()
    return _manager_instance
