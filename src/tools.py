"""
Phase 4 - Calculator tools
----------------------------
Concrete, deterministic calculators the agent calls instead of computing
register math via next-token prediction -- exactly the kind of arithmetic
LLMs get subtly, confidently wrong. Each function's docstring is also what
gets sent to the LLM as its tool description (see TOOL_SCHEMAS below).

Formulas and constants here are transcribed from specific ATmega32 datasheet
sections (cited in each docstring) -- adapting this to a different chip
means changing these, not just swapping the corpus.
"""
import math

# ---------------------------------------------------------------------------
# USART baud rate
# ---------------------------------------------------------------------------


def calc_baud_rate_ubrr(f_osc_hz: float, target_baud: float, double_speed: bool = False) -> dict:
    """
    Compute the UBRR register value for a target USART baud rate.

    Source: ATmega32 datasheet, "USART Baud Rate Registers" (UBRRH/UBRRL)
    and the "Equation for Calculating UBRR Value" in the baud rate generator
    section.

    Normal mode:        UBRR = floor(f_osc / (16 * baud)) - 1
    Double speed (U2X):  UBRR = floor(f_osc / (8  * baud)) - 1

    An integer UBRR usually can't hit the target baud rate exactly, so this
    also returns the ACTUAL baud rate that value produces and the resulting
    error percentage -- "what UBRR do I use" is an incomplete answer without
    knowing how far off the real rate lands. The datasheet's own
    "Asynchronous Operational Range" table treats error above ~2% as a real
    risk of receiver framing errors, so a warning is attached past that.
    """
    if f_osc_hz <= 0 or target_baud <= 0:
        raise ValueError("f_osc_hz and target_baud must be positive")

    divisor = 8 if double_speed else 16
    ubrr_exact = (f_osc_hz / (divisor * target_baud)) - 1
    ubrr = max(0, math.floor(ubrr_exact + 1e-9))  # epsilon guards float error at exact boundaries

    actual_baud = f_osc_hz / (divisor * (ubrr + 1))
    error_percent = ((actual_baud - target_baud) / target_baud) * 100

    return {
        "ubrr": ubrr,
        "ubrrh": (ubrr >> 8) & 0x0F,  # UBRR is a 12-bit value; high nibble -> UBRRH
        "ubrrl": ubrr & 0xFF,
        "actual_baud": round(actual_baud, 2),
        "target_baud": target_baud,
        "error_percent": round(error_percent, 3),
        "double_speed": double_speed,
        "f_osc_hz": f_osc_hz,
        "warning": (
            f"Error {error_percent:.2f}% exceeds the ~2% the datasheet's Asynchronous "
            f"Operational Range table treats as safe for reliable framing -- consider "
            f"double_speed or a different oscillator/baud combination."
        ) if abs(error_percent) > 2.0 else None,
    }


# ---------------------------------------------------------------------------
# ADC prescaler
# ---------------------------------------------------------------------------

# ADPS2:0 -> division factor, transcribed directly from the ADC Prescaler
# Select Bits table (ATmega32 datasheet, "Bits 2:0 - ADPS2:0" section).
# ADPS2:0 = 000 and 001 both map to division factor 2 -- that's the real
# table, not a transcription error.
_ADC_PRESCALER_TABLE = {
    "000": 2, "001": 2, "010": 4, "011": 8,
    "100": 16, "101": 32, "110": 64, "111": 128,
}

# Recommended ADC input clock range for full 10-bit resolution (ATmega32
# datasheet, ADC Characteristics / conversion timing guidance). Above this
# range accuracy degrades; faster conversions are fine if 10-bit accuracy
# isn't required. Verify against your corpus's "ADC Characteristics" chunk
# before trusting this for anything safety-critical -- it's baked into code
# here, not retrieved, so it isn't covered by Phase 5's verification step.
_ADC_CLOCK_MIN_HZ = 50_000
_ADC_CLOCK_MAX_HZ = 200_000


def calc_adc_prescaler(f_osc_hz: float) -> dict:
    """
    Pick an ADC prescaler (ADPS2:0) that brings the ADC input clock into the
    recommended 50kHz-200kHz range for full 10-bit resolution.

    Returns the chosen ADPS2:0 bits, the resulting division factor and ADC
    clock frequency, plus every option so the caller can see the
    alternatives and why one was picked over another.
    """
    if f_osc_hz <= 0:
        raise ValueError("f_osc_hz must be positive")

    options = []
    for bits, div in _ADC_PRESCALER_TABLE.items():
        adc_clock = f_osc_hz / div
        in_range = _ADC_CLOCK_MIN_HZ <= adc_clock <= _ADC_CLOCK_MAX_HZ
        options.append({
            "adps2_0": bits, "division_factor": div,
            "adc_clock_hz": round(adc_clock, 1), "in_recommended_range": in_range,
        })

    in_range_options = [o for o in options if o["in_recommended_range"]]
    if in_range_options:
        # prefer the LARGEST division factor still in range: slowest (most
        # accurate, lowest power) conversion that still qualifies
        chosen = max(in_range_options, key=lambda o: o["division_factor"])
        warning = None
    else:
        target_mid = (_ADC_CLOCK_MIN_HZ + _ADC_CLOCK_MAX_HZ) / 2
        chosen = min(options, key=lambda o: abs(o["adc_clock_hz"] - target_mid))
        warning = (
            f"No prescaler brings the ADC clock into the recommended "
            f"{_ADC_CLOCK_MIN_HZ/1000:.0f}-{_ADC_CLOCK_MAX_HZ/1000:.0f}kHz range at "
            f"{f_osc_hz/1e6:.3f}MHz -- closest option shown, but full 10-bit "
            f"resolution isn't guaranteed."
        )

    return {
        "chosen": chosen,
        "all_options": options,
        "f_osc_hz": f_osc_hz,
        "warning": warning,
    }


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI-compatible function-calling format) + dispatch table
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calc_baud_rate_ubrr",
            "description": (
                "Compute the USART UBRR register value for a target baud rate "
                "given the oscillator frequency, including the actual achieved "
                "baud rate and error percentage. Use this whenever the question "
                "asks for a UBRR value, a baud rate setting, or 'what baud rate "
                "do I get with UBRR=X' -- do not compute this arithmetic yourself."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "f_osc_hz": {"type": "number",
                                 "description": "Oscillator/clock frequency in Hz, e.g. 8000000 for 8MHz"},
                    "target_baud": {"type": "number", "description": "Desired baud rate, e.g. 9600"},
                    "double_speed": {"type": "boolean",
                                      "description": "True if U2X double-speed mode is enabled. Default false."},
                },
                "required": ["f_osc_hz", "target_baud"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calc_adc_prescaler",
            "description": (
                "Compute the ADC prescaler (ADPS2:0 bits) that brings the ADC input "
                "clock into the recommended 50kHz-200kHz range for full 10-bit "
                "resolution, given the oscillator frequency. Use this whenever the "
                "question asks about ADC prescaler, ADC clock speed, or ADC "
                "conversion speed/accuracy tradeoffs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "f_osc_hz": {"type": "number",
                                 "description": "Oscillator/clock frequency in Hz, e.g. 16000000 for 16MHz"},
                },
                "required": ["f_osc_hz"],
            },
        },
    },
]

TOOL_DISPATCH = {
    "calc_baud_rate_ubrr": calc_baud_rate_ubrr,
    "calc_adc_prescaler": calc_adc_prescaler,
}
