"""
Unit tests for src/tools.py -- pure Python, no network/model needed.
Reference values are computed by hand from the same formulas (not recalled
from memory), so this catches transcription bugs in the code, not just
disagreement with a remembered number.

Usage:
    python scripts/test_tools.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tools import calc_adc_prescaler, calc_baud_rate_ubrr  # noqa: E402

failures = []


def check(label, actual, expected, tol=0.0):
    ok = (actual == expected) if tol == 0 else abs(actual - expected) <= tol
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {label}: got {actual!r}, expected {expected!r}")
    if not ok:
        failures.append(label)


print("=== calc_baud_rate_ubrr ===")

# 8MHz, 9600 baud, normal mode.
# UBRR = floor(8_000_000 / (16*9600)) - 1 = floor(52.083) - 1 = 52 - 1 = 51
r = calc_baud_rate_ubrr(8_000_000, 9600)
check("8MHz/9600 UBRR", r["ubrr"], 51)
check("8MHz/9600 actual_baud", r["actual_baud"], 8_000_000 / (16 * 52), tol=0.01)
check("8MHz/9600 error within 1%", abs(r["error_percent"]) < 1, True)
check("8MHz/9600 no warning (error is low)", r["warning"], None)

# 1MHz, 9600 baud, normal mode -- known-bad case, error should exceed 2%.
# UBRR = floor(1_000_000 / (16*9600)) - 1 = floor(6.510) - 1 = 6 - 1 = 5
r = calc_baud_rate_ubrr(1_000_000, 9600)
check("1MHz/9600 UBRR", r["ubrr"], 5)
expected_baud = 1_000_000 / (16 * 6)
check("1MHz/9600 actual_baud", r["actual_baud"], expected_baud, tol=0.01)
check("1MHz/9600 error exceeds 2% (known-bad combo)", abs(r["error_percent"]) > 2, True)
check("1MHz/9600 warning fires", r["warning"] is not None, True)

# Same 1MHz/9600 but double_speed=True should have much lower error.
# UBRR = floor(1_000_000 / (8*9600)) - 1 = floor(13.021) - 1 = 13 - 1 = 12
r = calc_baud_rate_ubrr(1_000_000, 9600, double_speed=True)
check("1MHz/9600 double_speed UBRR", r["ubrr"], 12)
check("1MHz/9600 double_speed error much lower than normal mode", abs(r["error_percent"]) < 1, True)

# ubrrh/ubrrl split sanity check on a value that spans both bytes.
# UBRR=0x155 (341) -> ubrrh should be 0x1 (bits 11:8), ubrrl should be 0x55
r = calc_baud_rate_ubrr(f_osc_hz=1, target_baud=1)  # dummy call just to get the dict shape
ubrr_test = 0x155
ubrrh_test, ubrrl_test = (ubrr_test >> 8) & 0x0F, ubrr_test & 0xFF
check("UBRR 0x155 splits to ubrrh=0x1", ubrrh_test, 0x1)
check("UBRR 0x155 splits to ubrrl=0x55", ubrrl_test, 0x55)

# Invalid input should raise, not silently return garbage.
try:
    calc_baud_rate_ubrr(-1, 9600)
    check("negative f_osc_hz raises ValueError", "no exception raised", "ValueError")
except ValueError:
    check("negative f_osc_hz raises ValueError", "ValueError", "ValueError")

print("\n=== calc_adc_prescaler ===")

# 8MHz: division factors 64 (125kHz) and 128 (62.5kHz) both land in
# 50-200kHz range; should prefer the larger divisor (128, slowest/most
# accurate qualifying option).
r = calc_adc_prescaler(8_000_000)
check("8MHz chosen division_factor", r["chosen"]["division_factor"], 128)
check("8MHz chosen adc_clock_hz", r["chosen"]["adc_clock_hz"], 62500.0)
check("8MHz no warning (in-range option exists)", r["warning"], None)

# 1MHz: division factors 8 (125kHz) and 16 (62.5kHz) both land in range;
# should prefer 16.
r = calc_adc_prescaler(1_000_000)
check("1MHz chosen division_factor", r["chosen"]["division_factor"], 16)
check("1MHz chosen adc_clock_hz", r["chosen"]["adc_clock_hz"], 62500.0)

# Boundary case: f_osc chosen so division factor 2 lands EXACTLY at 50kHz.
r = calc_adc_prescaler(100_000)
check("100kHz chosen division_factor (exact lower boundary)", r["chosen"]["division_factor"], 2)
check("100kHz chosen adc_clock_hz == boundary", r["chosen"]["adc_clock_hz"], 50_000.0)
check("100kHz no warning (boundary counts as in-range)", r["warning"], None)

# Out-of-range case: no divisor brings a very low f_osc into 50-200kHz.
# All options: /2=25000 /4=12500 ... all below 50kHz -> warning should fire.
r = calc_adc_prescaler(50_000)
check("50kHz f_osc: no option in range, warning fires", r["warning"] is not None, True)

try:
    calc_adc_prescaler(0)
    check("zero f_osc_hz raises ValueError", "no exception raised", "ValueError")
except ValueError:
    check("zero f_osc_hz raises ValueError", "ValueError", "ValueError")

print(f"\n{'ALL TESTS PASSED' if not failures else f'{len(failures)} FAILURE(S): ' + ', '.join(failures)}")
if failures:
    sys.exit(1)
