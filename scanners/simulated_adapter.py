"""
Simulated Biometric Scanner Adapter
===================================
Provides a realistic, fully functioning virtual fingerprint sensor that enables
end-to-end testing of the device switching interface, image capture workflow,
and downstream VGG16-SVM blood group prediction pipeline prior to physical
hardware identification and driver installation.
"""

import os
import glob
import time
import random
from typing import Dict, Any, Optional
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

from .base import BaseScannerAdapter, ScannerStatus, ScannerCaptureError, ScannerCancelledError
from .normalizer import normalize_fingerprint_image


class SimulatedScannerAdapter(BaseScannerAdapter):
    """
    Virtual Biometric Sensor Adapter.
    Cycles through genuine fingerprint impressions or synthesizes ridge patterns
    to validate the entire UI and ML pipeline without presenting as physical hardware.
    """

    def __init__(self, sample_dirs: Optional[list] = None):
        self._sample_index = 0
        self._sample_cache = []
        self._cancelled = False
        
        # Candidate dataset directories for realistic testing
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        candidate_paths = [
            os.path.join(base_dir, 'sample dataset'),
            os.path.join(base_dir, 'dataset', 'dataset_blood_group'),
            os.path.join(base_dir, 'dataset', 'Fingerprint Dataset for Blood Group Classification')
        ]
        if sample_dirs:
            candidate_paths = sample_dirs + candidate_paths

        # Discover available local impressions
        for path in candidate_paths:
            if os.path.exists(path):
                found = glob.glob(os.path.join(path, '**', '*.BMP'), recursive=True)
                if not found:
                    found = glob.glob(os.path.join(path, '**', '*.jpg'), recursive=True) + \
                            glob.glob(os.path.join(path, '**', '*.png'), recursive=True)
                if found:
                    self._sample_cache.extend(found[:50])

    @property
    def adapter_id(self) -> str:
        return "simulated"

    @property
    def display_name(self) -> str:
        return "Demo / Simulated Scanner (Virtual Test Mode)"

    @property
    def slot_number(self) -> int:
        return 0

    def is_hardware_connected(self) -> bool:
        # Returns True to indicate the demo service is ready, but telemetry explicitly declares is_simulated=True
        return True

    def is_sdk_available(self) -> bool:
        return True

    def get_status(self) -> ScannerStatus:
        return ScannerStatus.READY

    def get_telemetry(self) -> Dict[str, Any]:
        return {
            'adapter_id': self.adapter_id,
            'display_name': self.display_name,
            'slot': self.slot_number,
            'status': ScannerStatus.READY.value,
            'status_label': 'Demo Mode (Virtual)',
            'is_connected': True,
            'is_biometric': True,
            'is_simulated': True,
            'sdk_available': True,
            'vendor_name': 'Virtual Biometric Core (Software Simulation)',
            'model_name': 'Demo / Simulated Sensor (Virtual)',
            'vid': 'VIRTUAL',
            'pid': 'DEMO',
            'device_type': 'Demo / Software Simulation',
            'can_capture': True,
            'sample_pool_size': len(self._sample_cache),
            'integration_notes': (
                'DEMO MODE: Virtual Biometric Sensor — No physical hardware is connected. '
                'Using simulated fingerprint impressions for UI and ML pipeline testing.'
            )
        }

    def cancel(self) -> bool:
        self._cancelled = True
        return True

    def capture(self, timeout_seconds: float = 10.0, cancel_event=None) -> Dict[str, Any]:
        """
        Simulates physical finger placement, optical acquisition, and normalization.
        Supports cancellation.
        """
        self._cancelled = False
        t0 = time.time()
        
        # Emulate sensor exposure latency in small slices to check cancellation
        steps = 7
        slice_delay = 0.05
        for _ in range(steps):
            if self._cancelled or (cancel_event and cancel_event.is_set()):
                raise ScannerCancelledError("Scan operation was cancelled by user.")
            time.sleep(slice_delay)

        try:
            if self._cancelled or (cancel_event and cancel_event.is_set()):
                raise ScannerCancelledError("Scan operation was cancelled by user.")

            raw_img = self._acquire_sample_image()
            
            # Normalize to 512x512 canonical format
            pil_norm, png_bytes, data_url, quality = normalize_fingerprint_image(
                raw_img,
                target_size=(512, 512),
                maintain_aspect_ratio=True,
                auto_contrast=True
            )
            
            elapsed_ms = round((time.time() - t0) * 1000, 1)

            return {
                'success': True,
                'image_pil': pil_norm,
                'image_base64': data_url,
                'raw_dimensions': raw_img.size,
                'normalized_dimensions': (512, 512),
                'quality_score': quality['overall_score'],
                'quality_metrics': quality,
                'capture_time_ms': elapsed_ms,
                'scanner_id': self.adapter_id,
                'scanner_name': self.display_name,
                'mode': 'simulated',
                'is_simulated': True,
                'device_info': {
                    'vendor': 'Virtual Biometrics (Simulation)',
                    'model': 'Demo Sensor (Virtual Platen)',
                    'vid': 'VIRTUAL',
                    'pid': 'DEMO'
                }
            }
        except ScannerCancelledError:
            raise
        except Exception as e:
            raise ScannerCaptureError(f"Simulated capture failed: {str(e)}")


    def _acquire_sample_image(self) -> Image.Image:
        """Loads next sample from test pool or synthesizes a realistic ridge pattern."""
        if self._sample_cache:
            sample_path = self._sample_cache[self._sample_index % len(self._sample_cache)]
            self._sample_index += 1
            try:
                return Image.open(sample_path)
            except Exception:
                pass
        
        # Fallback: Mathematically synthesize a whorl/loop ridge pattern
        return self._synthesize_fingerprint()

    def _synthesize_fingerprint(self, width: int = 320, height: int = 400) -> Image.Image:
        """Generates a synthetic fingerprint impression using sinusoidal ridge contours."""
        y, x = np.mgrid[0:height, 0:width]
        cx, cy = width / 2.0, height / 2.0
        
        # Distance and angle from center
        r = np.sqrt((x - cx) ** 2 + ((y - cy) * 1.2) ** 2)
        theta = np.arctan2(y - cy, x - cx)
        
        # Spiral/Whorl ridge frequency
        frequency = 0.25
        ridges = np.sin(frequency * r + 2.0 * np.sin(theta))
        
        # Add subtle noise to mimic epidermal texture
        noise = np.random.normal(0, 0.15, (height, width))
        pattern = ridges + noise
        
        # Elliptical mask to represent finger contact area
        mask = (x - cx) ** 2 / ((width * 0.42) ** 2) + (y - cy) ** 2 / ((height * 0.44) ** 2)
        contact = np.clip(1.0 - mask, 0.0, 1.0)
        
        # Ridge pixel values (dark ridges on white background)
        img_arr = 255 - (pattern * 100 * contact + contact * 60)
        img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
        
        pil_img = Image.fromarray(img_arr, mode='L')
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=0.7))
        return pil_img

    def release(self) -> None:
        """No hardware handles to release for simulated sensor."""
        pass
