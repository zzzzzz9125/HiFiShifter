import ctypes
import os
import pathlib
import platform
from dataclasses import dataclass
from typing import Optional

import numpy as np


VSLIB_MAX_PATH = 256


class VslibUnavailableError(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class VslibError(RuntimeError):
    def __init__(self, step: str, code: int):
        super().__init__(f"VSLIB call '{step}' failed with code {code}")
        self.step = step
        self.code = code


@dataclass
class VslibStatus:
    available: bool
    dll_path: Optional[pathlib.Path]
    error: Optional[str]


class VSPRJINFO(ctypes.Structure):
    _fields_ = [
        ("masterVolume", ctypes.c_double),
        ("sampFreq", ctypes.c_int),
    ]


class VSITEMINFO(ctypes.Structure):
    _fields_ = [
        ("fileName", ctypes.c_char * VSLIB_MAX_PATH),
        ("sampFreq", ctypes.c_int),
        ("channel", ctypes.c_int),
        ("sampleOrg", ctypes.c_int),
        ("sampleEdit", ctypes.c_int),
        ("ctrlPntPs", ctypes.c_int),
        ("ctrlPntNum", ctypes.c_int),
        ("synthMode", ctypes.c_int),
        ("trackNum", ctypes.c_int),
        ("offset", ctypes.c_int),
    ]


class VSCPINFOEX(ctypes.Structure):
    _fields_ = [
        ("dynOrg", ctypes.c_double),
        ("dynEdit", ctypes.c_double),
        ("volume", ctypes.c_double),
        ("pan", ctypes.c_double),
        ("spcDyn", ctypes.c_double),
        ("pitAna", ctypes.c_int),
        ("pitOrg", ctypes.c_int),
        ("pitEdit", ctypes.c_int),
        ("formant", ctypes.c_int),
        ("pitFlgOrg", ctypes.c_int),
        ("pitFlgEdit", ctypes.c_int),
        ("breathiness", ctypes.c_int),
        ("eq1", ctypes.c_int),
        ("eq2", ctypes.c_int),
    ]


class VslibEngine:
    """Thin ctypes wrapper around vslib.dll / vslib_x64.dll."""

    def __init__(self):
        self._lib: ctypes.WinDLL | None = None
        self._dll_path: pathlib.Path | None = None
        self._load_error: Optional[str] = None
        self._status: Optional[VslibStatus] = None

    @property
    def status(self) -> VslibStatus:
        if self._status is None:
            try:
                self._ensure_loaded()
                self._status = VslibStatus(True, self._dll_path, None)
            except VslibUnavailableError as exc:
                self._status = VslibStatus(False, self._dll_path, exc.reason)
        return self._status

    def _ensure_loaded(self):
        if self._lib is not None:
            return

        if os.name != "nt":
            raise VslibUnavailableError("VSLIB is only available on Windows")

        repo_root = pathlib.Path(__file__).resolve().parents[2]
        vslib_dir = repo_root / "vslib"

        arch_bits, _ = platform.architecture()
        is_64 = arch_bits == "64bit"
        candidates = [
            vslib_dir / ("vslib_x64.dll" if is_64 else "vslib.dll"),
            vslib_dir / "vslib.dll",
            vslib_dir / "vslib_x64.dll",
        ]

        last_error: Optional[str] = None
        for cand in candidates:
            if not cand.exists():
                continue
            try:
                self._lib = ctypes.WinDLL(str(cand))
                self._dll_path = cand
                break
            except OSError as exc:  # pragma: no cover - platform-specific
                last_error = str(exc)
                self._lib = None
                continue

        if self._lib is None:
            reason = last_error or "vslib.dll not found"
            raise VslibUnavailableError(reason)

        self._bind_signatures()

    # Keep signatures in one place to avoid repetitive setup
    def _bind_signatures(self):
        assert self._lib is not None
        lib = self._lib

        lib.VslibCreateProject.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        lib.VslibCreateProject.restype = ctypes.c_int

        lib.VslibDeleteProject.argtypes = [ctypes.c_void_p]
        lib.VslibDeleteProject.restype = ctypes.c_int

        lib.VslibAddItem.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
        lib.VslibAddItem.restype = ctypes.c_int

        lib.VslibGetItemInfo.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(VSITEMINFO)]
        lib.VslibGetItemInfo.restype = ctypes.c_int

        lib.VslibGetCtrlPntInfoEx.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(VSCPINFOEX),
        ]
        lib.VslibGetCtrlPntInfoEx.restype = ctypes.c_int

        lib.VslibSetCtrlPntInfoEx.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(VSCPINFOEX),
        ]
        lib.VslibSetCtrlPntInfoEx.restype = ctypes.c_int

        lib.VslibGetProjectInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(VSPRJINFO)]
        lib.VslibGetProjectInfo.restype = ctypes.c_int

        lib.VslibSetProjectInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(VSPRJINFO)]
        lib.VslibSetProjectInfo.restype = ctypes.c_int

        lib.VslibGetMixSample.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        lib.VslibGetMixSample.restype = ctypes.c_int

        lib.VslibGetMixData.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.VslibGetMixData.restype = ctypes.c_int

        lib.VslibFreq2Cent.argtypes = [ctypes.c_double]
        lib.VslibFreq2Cent.restype = ctypes.c_int

    @staticmethod
    def _encode_path(path: pathlib.Path) -> bytes:
        # Use Windows MBCS encoding so JP paths work; fallback to UTF-8.
        try:
            return os.fsencode(str(path))
        except Exception:
            return str(path).encode("utf-8", errors="ignore")

    @staticmethod
    def _midi_to_hz(f0_midi: np.ndarray) -> np.ndarray:
        hz = np.zeros_like(f0_midi, dtype=np.float64)
        mask = ~np.isnan(f0_midi)
        hz[mask] = 440.0 * np.power(2.0, (f0_midi[mask] - 69.0) / 12.0)
        return hz

    def _check(self, rc: int, step: str):
        if rc != 0:
            raise VslibError(step, rc)

    def synthesize_from_pitch(
        self,
        wav_path: pathlib.Path,
        f0_midi_original: np.ndarray,
        f0_midi_edited: np.ndarray,
        *,
        sample_rate: int,
        hop_size: int,
    ) -> np.ndarray:
        """Synthesize audio using VSLIB with a per-frame MIDI F0 contour.

        If a control point has no pitch change (original == edited), we keep
        the original pitch/flag so consonants stay unprocessed.
        """
        self._ensure_loaded()
        lib = self._lib
        assert lib is not None

        prj = ctypes.c_void_p()
        self._check(lib.VslibCreateProject(ctypes.byref(prj)), "VslibCreateProject")

        try:
            item_num = ctypes.c_int()
            self._check(
                lib.VslibAddItem(prj, self._encode_path(wav_path), ctypes.byref(item_num)),
                "VslibAddItem",
            )

            # Force project sample rate to match the working buffer to avoid resample mismatches
            prj_info = VSPRJINFO()
            if lib.VslibGetProjectInfo(prj, ctypes.byref(prj_info)) == 0:
                if prj_info.sampFreq != int(sample_rate):
                    prj_info.sampFreq = int(sample_rate)
                    lib.VslibSetProjectInfo(prj, ctypes.byref(prj_info))

            info = VSITEMINFO()
            self._check(lib.VslibGetItemInfo(prj, item_num.value, ctypes.byref(info)), "VslibGetItemInfo")

            ctrl_rate = max(1, int(info.ctrlPntPs))
            ctrl_num = int(info.ctrlPntNum)
            if ctrl_num <= 0:
                raise VslibError("CtrlPointCount", -1)

            frame_dt = float(hop_size) / float(sample_rate)
            frame_times = np.arange(len(f0_midi_edited), dtype=np.float64) * frame_dt
            ctrl_times = np.arange(ctrl_num, dtype=np.float64) / float(ctrl_rate)

            f0_hz_orig = self._midi_to_hz(f0_midi_original)
            f0_hz_orig = np.nan_to_num(f0_hz_orig, nan=0.0, posinf=0.0, neginf=0.0)
            target_orig = np.interp(ctrl_times, frame_times, f0_hz_orig, left=0.0, right=0.0)

            f0_hz_edit = self._midi_to_hz(f0_midi_edited)
            f0_hz_edit = np.nan_to_num(f0_hz_edit, nan=0.0, posinf=0.0, neginf=0.0)
            target_edit = np.interp(ctrl_times, frame_times, f0_hz_edit, left=0.0, right=0.0)

            voiced = target_edit > 1e-3
            no_change = np.isclose(target_orig, target_edit, rtol=0, atol=1e-3)

            # Update every control point
            for idx in range(ctrl_num):
                cp = VSCPINFOEX()
                self._check(
                    lib.VslibGetCtrlPntInfoEx(prj, item_num.value, idx, ctypes.byref(cp)),
                    "VslibGetCtrlPntInfoEx",
                )

                if no_change[idx]:
                    cp.pitEdit = cp.pitOrg
                    cp.pitFlgEdit = cp.pitFlgOrg
                elif voiced[idx]:
                    cent = int(round(lib.VslibFreq2Cent(ctypes.c_double(target_edit[idx]))))
                    cp.pitEdit = cent
                    cp.pitFlgEdit = 1
                else:
                    cp.pitFlgEdit = 0
                self._check(
                    lib.VslibSetCtrlPntInfoEx(prj, item_num.value, idx, ctypes.byref(cp)),
                    "VslibSetCtrlPntInfoEx",
                )

            total_samples = ctypes.c_int()
            self._check(lib.VslibGetMixSample(prj, ctypes.byref(total_samples)), "VslibGetMixSample")
            n = int(total_samples.value)
            if n <= 0:
                raise VslibError("MixSampleZero", -1)

            buf = (ctypes.c_short * n)()
            self._check(
                lib.VslibGetMixData(prj, buf, 16, 1, 0, n),
                "VslibGetMixData",
            )

            audio = np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0
            return audio
        finally:
            try:
                lib.VslibDeleteProject(prj)
            except Exception:
                pass
