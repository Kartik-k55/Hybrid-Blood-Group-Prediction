"""
Vendor SDK Integration Registry & Driver Templates
==================================================
Documents the precise C-DLL / COM / HTTP driver hooks for major biometric manufacturers:
- Mantra Softech (MFS100 / MIS100)
- SecuGen Corporation (Hamster Plus / Pro 20 / IV)
- DigitalPersona / HID (U.are.U 4500 / 5160 / 5300)
- IDEMIA / Morpho (MSO 1300 E2/E3)
- Futronic (FS80 / FS88)
- ZKTeco (SLK20R / Live20R / ZK9500)

Note: In accordance with project requirements, no fake SDK calls are executed.
Hardware capture hooks remain guarded until genuine hardware is connected and the vendor
SDK library or service is verified on the host system.
"""

import os
import sys
import ctypes
from typing import Dict, Any, Optional, Tuple


# Known system DLL locations on Windows
VENDOR_DLL_CANDIDATES = {
    'mantra': [
        r'C:\Program Files\Mantra\MFS100\Driver\MFS100.dll',
        r'C:\Program Files (x86)\Mantra\MFS100\Driver\MFS100.dll',
        'MFS100.dll'
    ],
    'secugen': [
        r'C:\Program Files\SecuGen\FDx SDK Pro for Windows\bin\sgfplib.dll',
        r'C:\Windows\System32\sgfplib.dll',
        'sgfplib.dll'
    ],
    'digitalpersona': [
        r'C:\Program Files\DigitalPersona\U.are.U SDK\Windows\Lib\x64\dpfpdd.dll',
        r'C:\Windows\System32\dpfpdd.dll',
        'dpfpdd.dll'
    ],
    'morpho': [
        r'C:\Program Files\Morpho\MorphoSmart SDK\bin\MSO100.dll',
        r'C:\Windows\System32\MSO100.dll',
        'MSO100.dll'
    ],
    'futronic': [
        r'C:\Windows\System32\ftrScanAPI.dll',
        r'C:\Windows\SysWOW64\ftrScanAPI.dll',
        'ftrScanAPI.dll'
    ],
    'zkteco': [
        r'C:\Program Files (x86)\ZKTeco\zkfp.dll',
        r'C:\Windows\System32\zkfp.dll',
        'zkfp.dll'
    ]
}


def find_vendor_dll(vendor_key: str) -> Optional[str]:
    """Checks whether the specified vendor's C-shared library exists on the host."""
    candidates = VENDOR_DLL_CANDIDATES.get(vendor_key.lower(), [])
    for path in candidates:
        if os.path.isabs(path) and os.path.exists(path):
            return path
        # Try loading by name if in system PATH
        try:
            ctypes.CDLL(path)
            return path
        except Exception:
            continue
    return None


def get_vendor_integration_guide(vendor_name: str, vid: str, pid: str) -> Dict[str, str]:
    """
    Returns step-by-step instructions for integrating the identified scanner model.
    """
    vid_clean = vid.upper().strip()
    
    if vid_clean == '27D0':  # Mantra
        return {
            'vendor': 'Mantra Softech',
            'driver_download': 'https://download.mantratecapp.com/ (Download MFS100 Driver & MFS100 Client Service)',
            'recommended_sdk': 'Mantra MFS100 SDK or Local RD Service (Port 11100)',
            'c_dll_name': 'MFS100.dll',
            'integration_step': (
                'Option A (Local RD Service HTTP): Query http://127.0.0.1:11100/rd/capture via requests.POST '
                'with PID XML; returns raw base64 biometric image.\n'
                'Option B (Direct DLL): Load MFS100.dll using ctypes.cdll.LoadLibrary and call MFS100Init() -> MFS100StartCapture().'
            )
        }
    elif vid_clean == '1162':  # SecuGen
        return {
            'vendor': 'SecuGen Corporation',
            'driver_download': 'https://secugen.com/drivers/ (Download WBF Driver and FDx SDK Pro for Windows)',
            'recommended_sdk': 'SecuGen FDx SDK Pro for Windows (C/C++ or .NET)',
            'c_dll_name': 'sgfplib.dll',
            'integration_step': (
                'Install SecuGen FDx SDK Pro. In Python, load sgfplib.dll via ctypes, initialize device with '
                'SGFPM_Create() -> SGFPM_Init() -> SGFPM_OpenDevice() -> SGFPM_GetImage().'
            )
        }
    elif vid_clean in ['08FF', '05BA']:  # DigitalPersona / HID
        return {
            'vendor': 'DigitalPersona / HID Global',
            'driver_download': 'https://www.hidglobal.com/drivers (Download DigitalPersona U.are.U RTE and SDK)',
            'recommended_sdk': 'DigitalPersona U.are.U SDK for Windows',
            'c_dll_name': 'dpfpdd.dll',
            'integration_step': (
                'Install U.are.U Windows Runtime. Load dpfpdd.dll via ctypes; call dpfpdd_init() -> '
                'dpfpdd_open() -> dpfpdd_capture() to obtain 8-bit grayscale pixel stream.'
            )
        }
    elif vid_clean == '079B':  # Morpho / IDEMIA
        return {
            'vendor': 'IDEMIA / MorphoSmart',
            'driver_download': 'https://biometric-devices.idemia.com/ (MorphoSmart USB Driver & MorphoBVP SDK)',
            'recommended_sdk': 'MorphoSmart Host SDK or Morpho RD Service',
            'c_dll_name': 'MSO100.dll',
            'integration_step': (
                'Install MorphoSmart USB driver and SDK. Interface via Morpho MSO API or local RD Service daemon.'
            )
        }
    elif vid_clean == '1491':  # Futronic
        return {
            'vendor': 'Futronic Technology',
            'driver_download': 'http://www.futronic-tech.com/drivers.html (Futronic Windows Drivers & Scan API)',
            'recommended_sdk': 'Futronic Windows API SDK',
            'c_dll_name': 'ftrScanAPI.dll',
            'integration_step': (
                'Place ftrScanAPI.dll in system path. Call ftrScanOpenDevice() -> ftrScanGetFrame() to capture 320x480 frame.'
            )
        }
    elif vid_clean == '1B55':  # ZKTeco
        return {
            'vendor': 'ZKTeco',
            'driver_download': 'https://www.zkteco.com/en/download_cat/14.html (ZKTeco Fingerprint Sensor Driver)',
            'recommended_sdk': 'ZKTeco Live Fingerprint SDK (zkfp.dll)',
            'c_dll_name': 'zkfp.dll',
            'integration_step': (
                'Place zkfp.dll in Windows path. Call ZKFPM_Init() -> ZKFPM_OpenDevice() -> ZKFPM_AcquireFingerprint().'
            )
        }
    else:
        return {
            'vendor': vendor_name or 'Unknown Manufacturer',
            'driver_download': 'Search manufacturer portal using detected Vendor ID and Product ID.',
            'recommended_sdk': 'Official Vendor C / WinUSB / WBF Driver',
            'c_dll_name': 'Vendor-specific DLL',
            'integration_step': (
                f'Hardware detected with VID 0x{vid_clean} and PID 0x{pid.upper()}. '
                f'Refer to the vendor documentation to install the official Windows driver and SDK.'
            )
        }
