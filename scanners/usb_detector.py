"""
USB Hardware Detection & Biometric Device Classifier
====================================================
Discovers physical USB devices connected to the host system, extracts Vendor ID (VID)
and Product ID (PID), and strictly validates whether each peripheral is a known
biometric fingerprint scanner versus an unsupported generic USB device.
"""

import re
import json
import time
import subprocess
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple


# Registry of known Biometric Fingerprint Scanner Hardware Vendors
KNOWN_BIOMETRIC_VENDORS = {
    '27D0': {
        'vendor': 'Mantra Softech India Pvt. Ltd.',
        'models': {
            '0001': 'MFS100 Optical Fingerprint Sensor (V1)',
            '0002': 'MFS100 Optical Fingerprint Sensor (V2)',
            '0003': 'MIS100V2 Iris / Fingerprint Scanner',
            '0007': 'MFS500 Optical Fingerprint Sensor'
        },
        'default_model': 'Mantra MFS100 Optical Scanner',
        'sdk_name': 'Mantra MFS100 SDK / RD Service'
    },
    '1162': {
        'vendor': 'SecuGen Corporation',
        'models': {
            '0320': 'SecuGen Hamster IV',
            '0330': 'SecuGen Hamster Plus (HSDU03P)',
            '1000': 'SecuGen Hamster Pro 20 (HU20)',
            '2000': 'SecuGen Hamster Pro 10',
            '2200': 'SecuGen Hamster Pro'
        },
        'default_model': 'SecuGen Hamster Biometric Scanner',
        'sdk_name': 'SecuGen FDx SDK Pro for Windows'
    },
    '08FF': {
        'vendor': 'AuthenTec / DigitalPersona / HID',
        'models': {
            '2580': 'DigitalPersona Fingerprint Sensor',
            '2810': 'AuthenTec AES2810 Fingerprint Reader',
            '1600': 'DigitalPersona U.are.U 4000/4500'
        },
        'default_model': 'DigitalPersona / AuthenTec Sensor',
        'sdk_name': 'DigitalPersona One Touch / U.are.U SDK'
    },
    '05BA': {
        'vendor': 'DigitalPersona, Inc. (HID Global)',
        'models': {
            '000A': 'DigitalPersona U.are.U 4500 Fingerprint Reader',
            '000B': 'DigitalPersona U.are.U 5160 Fingerprint Reader',
            '000C': 'DigitalPersona U.are.U 5300 Fingerprint Reader'
        },
        'default_model': 'DigitalPersona U.are.U Optical Scanner',
        'sdk_name': 'DigitalPersona U.are.U Windows SDK (dpfpdd.dll)'
    },
    '079B': {
        'vendor': 'IDEMIA / Morpho (Safran)',
        'models': {
            '0024': 'MorphoSmart MSO 1300 E',
            '0045': 'MorphoSmart MSO 1300 E2',
            '0046': 'MorphoSmart MSO 1300 E3'
        },
        'default_model': 'MorphoSmart MSO 1300 Series Scanner',
        'sdk_name': 'Morpho MSO SDK / MorphoBVP'
    },
    '1491': {
        'vendor': 'Futronic Technology Co. Ltd.',
        'models': {
            '0088': 'Futronic FS80 / FS88 USB 2.0 Optical Scanner',
            '0090': 'Futronic FS90 Optical Scanner',
            '0025': 'Futronic FS80H Optical Scanner'
        },
        'default_model': 'Futronic Optical Fingerprint Scanner',
        'sdk_name': 'Futronic Windows API SDK (ftrScanAPI.dll)'
    },
    '1B55': {
        'vendor': 'ZKTeco Inc.',
        'models': {
            '0120': 'ZKTeco SLK20R USB Fingerprint Scanner',
            '0130': 'ZKTeco ZK9500 Optical Scanner',
            '0140': 'ZKTeco Live20R Optical Scanner'
        },
        'default_model': 'ZKTeco Live Biometric Scanner',
        'sdk_name': 'ZKTeco Fingerprint SDK (zkfp.dll)'
    },
    '0A5C': {
        'vendor': 'Broadcom / Startek',
        'models': {
            'E000': 'Startek FM220 Optical Fingerprint Scanner',
            'E001': 'Startek FM220U Optical Scanner'
        },
        'default_model': 'Startek FM220 Biometric Scanner',
        'sdk_name': 'Startek FM220 SDK (libfpapi.dll)'
    },
    '16D1': {
        'vendor': 'Suprema Inc.',
        'models': {
            '0400': 'Suprema BioMini Optical Scanner',
            '0401': 'Suprema BioMini Plus 2',
            '0402': 'Suprema BioMini Slim'
        },
        'default_model': 'Suprema BioMini Biometric Scanner',
        'sdk_name': 'Suprema BioMini SDK (UFScanner.dll)'
    },
    '138A': {
        'vendor': 'Validity Sensors, Inc. (Synaptics)',
        'models': {
            '0001': 'Validity VFS Fingerprint Sensor',
            '003C': 'Validity VFS451 Fingerprint Reader',
            '0090': 'Validity VFS7500 Fingerprint Sensor'
        },
        'default_model': 'Synaptics / Validity Biometric Sensor',
        'sdk_name': 'Synaptics WBF Driver'
    },
    '06CB': {
        'vendor': 'Synaptics, Inc.',
        'models': {},
        'default_model': 'Synaptics Biometric Sensor',
        'sdk_name': 'Synaptics Falcon/Prometheus WBF Driver'
    },
    '27C6': {
        'vendor': 'Shenzhen Goodix Technology Co., Ltd.',
        'models': {},
        'default_model': 'Goodix Fingerprint Sensor',
        'sdk_name': 'Goodix WBF Biometric Driver'
    },
    '04F3': {
        'vendor': 'ELAN Microelectronics Corp.',
        'models': {},
        'default_model': 'ELAN Fingerprint Sensor',
        'sdk_name': 'ELAN WBF Biometric Driver'
    }
}

# Generic non-biometric keyword filters
NON_BIOMETRIC_KEYWORDS = [
    'camera', 'webcam', 'video', 'bluetooth', 'audio', 'sound', 'microphone',
    'mouse', 'keyboard', 'composite', 'hub', 'mass storage', 'disk', 'flash drive',
    'ethernet', 'wifi', 'wireless', 'controller', 'host controller', 'root hub'
]


@dataclass
class USBDeviceInfo:
    name: str
    device_id: str
    vid: str
    pid: str
    manufacturer: str
    status: str
    present: bool
    is_biometric_candidate: bool
    vendor_name: str
    model_name: str
    sdk_name: str
    device_type: str
    notes: str

    def to_dict(self) -> Dict:
        return asdict(self)


class USBDetector:
    """
    Detects and inspects connected USB devices on Windows using WMI CIM queries.
    Maintains a fast TTL cache to prevent repeated PowerShell process spawns.
    """
    _cached_devices: List[USBDeviceInfo] = []
    _last_query_time: float = 0.0
    _cache_ttl_seconds: float = 3.0

    VID_PID_REGEX = re.compile(r'VID_([0-9A-Fa-f]{4})&PID_([0-9A-Fa-f]{4})', re.IGNORECASE)

    @classmethod
    def detect_devices(cls, force_refresh: bool = False) -> List[USBDeviceInfo]:
        """
        Enumerate all connected USB devices and classify them.
        Uses cached results if called within cache TTL unless force_refresh is True.
        """
        now = time.time()
        if not force_refresh and cls._cached_devices and (now - cls._last_query_time < cls._cache_ttl_seconds):
            return cls._cached_devices

        raw_devices = cls._query_windows_usb_devices()
        classified = [cls._classify_device(d) for d in raw_devices]
        
        # Sort so biometric candidates appear at the top
        classified.sort(key=lambda x: (not x.is_biometric_candidate, x.name.lower()))

        cls._cached_devices = classified
        cls._last_query_time = now
        return classified

    @classmethod
    def get_biometric_scanners(cls, force_refresh: bool = False) -> List[USBDeviceInfo]:
        """Returns only connected devices verified as biometric fingerprint scanners."""
        all_devs = cls.detect_devices(force_refresh=force_refresh)
        return [d for d in all_devs if d.is_biometric_candidate]

    @classmethod
    def get_unsupported_devices(cls, force_refresh: bool = False) -> List[USBDeviceInfo]:
        """Returns connected USB peripherals that are NOT biometric scanners."""
        all_devs = cls.detect_devices(force_refresh=force_refresh)
        return [d for d in all_devs if not d.is_biometric_candidate]

    @classmethod
    def _query_windows_usb_devices(cls) -> List[Dict]:
        """Queries WMI Win32_PnPEntity for USB devices with VID/PID."""
        ps_cmd = (
            'Get-CimInstance -ClassName Win32_PnPEntity -Filter "DeviceID LIKE \'USB%VID%\'" | '
            'Select-Object Name, DeviceID, Manufacturer, Status, Present | '
            'ConvertTo-Json -Compress'
        )
        try:
            res = subprocess.run(
                ['powershell', '-NoProfile', '-NonInteractive', '-Command', ps_cmd],
                capture_output=True,
                text=True,
                timeout=6
            )
            if res.returncode != 0 or not res.stdout.strip():
                return []
            
            data = json.loads(res.stdout)
            if isinstance(data, dict):
                data = [data]
            return data
        except Exception as e:
            # Fallback or empty if execution times out
            return []

    @classmethod
    def _classify_device(cls, raw: Dict) -> USBDeviceInfo:
        """Classifies a raw device dictionary into a structured USBDeviceInfo."""
        name = raw.get('Name') or 'Unknown USB Device'
        dev_id = raw.get('DeviceID') or ''
        mfr = raw.get('Manufacturer') or 'Unknown Manufacturer'
        status = raw.get('Status') or 'Unknown'
        present = bool(raw.get('Present', True))

        vid = "Unknown"
        pid = "Unknown"
        m = cls.VID_PID_REGEX.search(dev_id)
        if m:
            vid = m.group(1).upper()
            pid = m.group(2).upper()

        name_lower = name.lower()
        dev_id_lower = dev_id.lower()
        mfr_lower = mfr.lower()

        is_biometric = False
        vendor_info = KNOWN_BIOMETRIC_VENDORS.get(vid)
        vendor_name = vendor_info['vendor'] if vendor_info else mfr
        sdk_name = vendor_info.get('sdk_name', 'Vendor SDK Required') if vendor_info else 'Not Applicable'
        model_name = "Unidentified Model"
        device_type = "Generic USB Peripheral"
        notes = ""

        # 1. Match known biometric vendor VID
        if vendor_info:
            is_biometric = True
            model_name = vendor_info['models'].get(pid, vendor_info['default_model'])
            device_type = "Biometric Fingerprint Scanner"
            notes = (
                f"Recognized biometric manufacturer: {vendor_name}. "
                f"Requires {sdk_name} for image acquisition."
            )
        # 2. Check for explicit biometric keywords in Name/Manufacturer
        elif any(k in name_lower or k in mfr_lower for k in ['fingerprint', 'biometric', 'validity', 'authentec', 'secugen', 'mantra']):
            is_biometric = True
            device_type = "Biometric Fingerprint Scanner (Unregistered VID)"
            model_name = name
            notes = "Device reports biometric descriptor in USB descriptor table."
        else:
            # 3. Non-biometric device check
            is_biometric = False
            for k in NON_BIOMETRIC_KEYWORDS:
                if k in name_lower:
                    device_type = f"USB {k.capitalize()}"
                    break
            notes = "Unsupported peripheral. This device is not an optical or capacitive fingerprint scanner."

        return USBDeviceInfo(
            name=name,
            device_id=dev_id,
            vid=vid,
            pid=pid,
            manufacturer=mfr,
            status=status,
            present=present,
            is_biometric_candidate=is_biometric,
            vendor_name=vendor_name,
            model_name=model_name,
            sdk_name=sdk_name,
            device_type=device_type,
            notes=notes
        )
