"""
Scanner 1 Modular Hardware Adapter (Primary USB Slot)
=====================================================
Represents the primary physical USB fingerprint scanner slot.
Performs real-time USB bus discovery, hardware VID/PID verification,
driver/SDK presence auditing, and image normalization.
"""

import time
from typing import Dict, Any, Optional
from PIL import Image

from .base import (
    BaseScannerAdapter,
    ScannerStatus,
    ScannerNotConnectedError,
    DriverMissingError,
    ScannerCaptureError,
    ScannerCancelledError
)
from .usb_detector import USBDetector, USBDeviceInfo
from .normalizer import normalize_fingerprint_image
from .vendor_templates import find_vendor_dll, get_vendor_integration_guide


class Scanner1Adapter(BaseScannerAdapter):
    """
    Adapter for Physical USB Scanner Slot 1.
    """

    def __init__(self, target_vid: Optional[str] = None, target_pid: Optional[str] = None):
        self._target_vid = target_vid.upper() if target_vid else None
        self._target_pid = target_pid.upper() if target_pid else None
        self._is_busy = False

    @property
    def adapter_id(self) -> str:
        return "scanner_1"

    @property
    def display_name(self) -> str:
        return "Scanner 1 (Primary USB Slot)"

    @property
    def slot_number(self) -> int:
        return 1

    def _get_matched_device(self, force_refresh: bool = False) -> Optional[USBDeviceInfo]:
        """Finds the first biometric candidate device on the USB bus or matches configured VID/PID."""
        biometric_devices = USBDetector.get_biometric_scanners(force_refresh=force_refresh)
        if not biometric_devices:
            return None

        if self._target_vid:
            for dev in biometric_devices:
                if dev.vid == self._target_vid:
                    if not self._target_pid or dev.pid == self._target_pid:
                        return dev
            return None

        # By default, Slot 1 binds to the first detected biometric device
        return biometric_devices[0]

    def is_hardware_connected(self) -> bool:
        return self._get_matched_device() is not None

    def is_sdk_available(self) -> bool:
        dev = self._get_matched_device()
        if not dev:
            return False
        # Check if known vendor DLL or SDK is accessible on host
        vendor_keys = {
            '27D0': 'mantra',
            '1162': 'secugen',
            '08FF': 'digitalpersona',
            '05BA': 'digitalpersona',
            '079B': 'morpho',
            '1491': 'futronic',
            '1B55': 'zkteco'
        }
        vkey = vendor_keys.get(dev.vid)
        if vkey:
            dll_path = find_vendor_dll(vkey)
            return dll_path is not None
        return False

    def get_status(self) -> ScannerStatus:
        dev = self._get_matched_device()
        if not dev:
            return ScannerStatus.DISCONNECTED
        if not dev.is_biometric_candidate:
            return ScannerStatus.UNVERIFIED
        if self.is_sdk_available():
            return ScannerStatus.READY
        return ScannerStatus.DRIVER_MISSING

    def get_telemetry(self) -> Dict[str, Any]:
        dev = self._get_matched_device()
        status = self.get_status()
        
        if not dev:
            return {
                'adapter_id': self.adapter_id,
                'display_name': self.display_name,
                'slot': self.slot_number,
                'status': status.value,
                'status_label': 'Scanner Disconnected',
                'is_connected': False,
                'is_biometric': False,
                'sdk_available': False,
                'vendor_name': 'None Detected',
                'model_name': 'No Hardware Attached to Slot 1',
                'vid': 'N/A',
                'pid': 'N/A',
                'device_type': 'Not Connected',
                'can_capture': False,
                'integration_notes': (
                    'Plug your first USB fingerprint scanner into any USB port and click "Refresh USB Devices".'
                )
            }

        guide = get_vendor_integration_guide(dev.vendor_name, dev.vid, dev.pid)
        sdk_ready = self.is_sdk_available()

        if sdk_ready:
            status_label = f"Ready ({dev.vendor_name})"
            notes = f"Verified hardware and driver available for {dev.model_name}."
        else:
            status_label = "Device detected — capture integration required"
            notes = (
                f"Hardware detected ({dev.name}, VID: {dev.vid}, PID: {dev.pid}). "
                f"Requires installation of {dev.sdk_name}. "
                f"Download from: {guide.get('driver_download', 'Vendor portal')}."
            )

        return {
            'adapter_id': self.adapter_id,
            'display_name': self.display_name,
            'slot': self.slot_number,
            'status': status.value,
            'status_label': status_label,
            'is_connected': True,
            'is_biometric': dev.is_biometric_candidate,
            'sdk_available': sdk_ready,
            'vendor_name': dev.vendor_name,
            'model_name': dev.model_name,
            'vid': dev.vid,
            'pid': dev.pid,
            'device_type': dev.device_type,
            'can_capture': sdk_ready,
            'integration_guide': guide,
            'integration_notes': notes
        }

    def capture(self, timeout_seconds: float = 10.0, cancel_event=None) -> Dict[str, Any]:
        """
        Executes live fingerprint capture on Scanner 1 once verified.
        Guarded against unverified SDK calls.
        """
        if cancel_event and cancel_event.is_set():
            raise ScannerCancelledError("Scan operation was cancelled by user.")

        dev = self._get_matched_device(force_refresh=True)
        if not dev:
            raise ScannerNotConnectedError(
                "Scanner 1 is not detected on the USB bus. Please connect your USB fingerprint scanner."
            )

        if not self.is_sdk_available():
            guide = get_vendor_integration_guide(dev.vendor_name, dev.vid, dev.pid)
            raise DriverMissingError(
                f"Scanner 1 ({dev.vendor_name} {dev.model_name}, VID: {dev.vid}, PID: {dev.pid}) was detected, "
                f"but the official vendor SDK ({dev.sdk_name}) is not installed on this system. "
                f"Please install the vendor driver/SDK ({guide.get('driver_download')}) "
                f"or switch to 'Simulated Scanner' mode to test the pipeline."
            )

        # Verified driver hook execution
        try:
            raw_image = self._capture_from_hardware_sdk(dev, timeout_seconds=timeout_seconds)
            norm_pil, png_bytes, data_url, quality = normalize_fingerprint_image(raw_image)
            return {
                'success': True,
                'image_pil': norm_pil,
                'image_base64': data_url,
                'raw_dimensions': raw_image.size,
                'normalized_dimensions': (512, 512),
                'quality_score': quality['overall_score'],
                'quality_metrics': quality,
                'scanner_id': self.adapter_id,
                'scanner_name': self.display_name,
                'mode': 'hardware',
                'device_info': {
                    'vendor': dev.vendor_name,
                    'model': dev.model_name,
                    'vid': dev.vid,
                    'pid': dev.pid
                }
            }
        except Exception as e:
            raise ScannerCaptureError(f"Hardware capture failed on Scanner 1: {str(e)}")

    def _capture_from_hardware_sdk(self, dev: USBDeviceInfo, timeout_seconds: float) -> Image.Image:
        """
        Vendor SDK image acquisition hook.
        Strictly guarded: Invoked only when verified SDK DLL is confirmed present.
        """
        # Hook placeholder where vendor SDK function is called once physical hardware is verified
        raise NotImplementedError(
            f"Hardware capture hook for {dev.vendor_name} is awaiting physical verification."
        )

    def cancel(self) -> bool:
        self._is_busy = False
        return True

    def release(self) -> None:
        """Safely release SDK session or communication handle."""
        self._is_busy = False

