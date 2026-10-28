"""
ESP32 calculator tools -- separate module from tools.py (ATmega32) because
these two chips' register math has nothing in common, and an agent
answering questions about one datasheet shouldn't see tool schemas for a
totally different chip. Not yet wired into agent.py's dispatch (that needs
a doc-id -> tool-set mapping agent.py doesn't have yet) -- these are built
and unit-tested standalone first, same discipline as tools.py: grounded in
actual retrieved corpus text, not memory of ESP32 register maps.

Source for every constant below: ESP32 Technical Reference Manual v5.8,
Chapter 31 "On-Chip Sensors and Analog Signal Processing", page 751,
Registers 31.4/31.5 (SENS_SAR_ATTEN1_REG / SENS_SAR_ATTEN2_REG) -- confirmed
against a screenshot of the actual PDF page, not recalled from training
data. (Worth noting: my initial recollection of ESP32 attenuation values
was the ESP-IDF *software* API's labels -- 0/2.5/6/11 dB -- which are NOT
what this register's raw 2-bit encoding uses. Different abstraction layer,
different numbers. Don't conflate them.)
"""

# 2-bit code -> dB attenuation. Deliberately non-monotonic in the real
# hardware: code 11 gives LESS attenuation (1 dB) than code 10 (6 dB).
# Encoded exactly as documented, not "corrected" to look more sensible.
_ESP32_ATTEN_CODE_TO_DB = {"00": 0, "01": 3, "10": 6, "11": 1}
_ESP32_ATTEN_DB_TO_CODE = {db: code for code, db in _ESP32_ATTEN_CODE_TO_DB.items()}


def calc_esp32_adc_atten_code(atten_db: float) -> dict:
    """
    Look up the 2-bit ADC attenuation code for a target dB attenuation.

    Source: ESP32 TRM v5.8, Registers 31.4/31.5 (SENS_SAR_ATTEN1_REG /
    SENS_SAR_ATTEN2_REG): "2-bit attenuation for each pad, 11: 1 dB,
    10: 6 dB, 01: 3 dB, 00: 0 dB."

    Only the four values actually documented (0, 1, 3, 6 dB) are valid --
    unlike calc_adc_prescaler's "closest option" fallback for the ATmega32,
    there's no continuous range to approximate here, so an unsupported
    value is a hard error rather than a silently-substituted nearest match.
    """
    if atten_db not in _ESP32_ATTEN_DB_TO_CODE:
        valid = sorted(_ESP32_ATTEN_DB_TO_CODE)
        raise ValueError(
            f"{atten_db} dB is not a documented attenuation value for this register. "
            f"Valid values are {valid} (see SENS_SAR_ATTEN1_REG/SENS_SAR_ATTEN2_REG)."
        )
    code = _ESP32_ATTEN_DB_TO_CODE[atten_db]
    return {
        "atten_db": atten_db,
        "atten_bits": code,
        "note": (
            "This is the SAR ADC hardware register encoding (TRM Registers 31.4/31.5), "
            "not the ESP-IDF adc_atten_t software API's dB labels -- those are a "
            "different abstraction with different values (0/2.5/6/11 dB) and do not "
            "map directly onto this register's raw bits."
        ),
    }


_ESP32_APB_CLK_HZ = 80_000_000  # confirmed: TRM v5.8 Ch.19 "UART Architecture" (page 312),
                                  # "The UART block can derive its clock from... the 80-MHz APB_CLK"


def calc_esp32_baud_rate(target_baud: float, apb_clk_hz: float = _ESP32_APB_CLK_HZ) -> dict:
    """
    Compute UART_CLKDIV / UART_CLKDIV_FRAG for a target baud rate.

    CONFIDENCE NOTE -- read before trusting this like calc_baud_rate_ubrr:
    Two different things are grounded here to two different degrees.
      - CONFIRMED from a screenshot of the actual TRM page 330 (Register
        19.6, UART_CLKDIV_REG): UART_CLKDIV_FRAG is bits [23:20] (4 bits),
        UART_CLKDIV is bits [19:0] (20 bits). These bit widths are a direct
        primary-source fact, not an inference.
      - INFERRED, not confirmed: the exact arithmetic combining them. No
        equation like "divisor = CLKDIV + CLKDIV_FRAG/16" was found stated
        anywhere in the retrieved corpus (unlike the ATmega32 UBRR formula,
        or even ESP32's own SPI clock formula, both of which were verbatim
        quoted equations). The /16 scaling used here follows from the
        confirmed 4-bit field width under the standard fixed-point-divider
        convention used near-universally in this class of hardware -- but
        it is an engineering inference from confirmed structure, not a
        quoted formula. Flagged explicitly rather than presented with the
        same confidence as the rest of this codebase's grounded tools.
      - The reset value (UART_CLKDIV=694, FRAG=0) gives ~115,273.8 Hz --
        ~0.06% off the standard 115200 baud bootloader default. Consistent
        with the inferred formula, not proof of it.

    Also note: no ESP32-specific documented error-tolerance table (like the
    ATmega32's "Asynchronous Operational Range") was found in this corpus,
    so the error_percent warning threshold below is a generic engineering
    rule of thumb, NOT sourced from this datasheet the way the ATmega32
    tool's 2% threshold was from its own document.
    """
    if apb_clk_hz <= 0 or target_baud <= 0:
        raise ValueError("apb_clk_hz and target_baud must be positive")

    divisor = apb_clk_hz / target_baud
    clkdiv = int(divisor)
    frag = round((divisor - clkdiv) * 16)
    if frag == 16:  # rounding carried the fraction over into the integer part
        frag = 0
        clkdiv += 1
    clkdiv = max(0, min(clkdiv, 0xFFFFF))  # clamp to the confirmed 20-bit field width
    frag = max(0, min(frag, 0xF))          # clamp to the confirmed 4-bit field width

    effective_divisor = clkdiv + frag / 16
    actual_baud = apb_clk_hz / effective_divisor if effective_divisor else 0
    error_percent = ((actual_baud - target_baud) / target_baud) * 100 if target_baud else 0

    return {
        "uart_clkdiv": clkdiv,
        "uart_clkdiv_frag": frag,
        "actual_baud": round(actual_baud, 2),
        "target_baud": target_baud,
        "error_percent": round(error_percent, 3),
        "apb_clk_hz": apb_clk_hz,
        "warning": (
            f"Error {error_percent:.2f}% -- no ESP32-specific documented tolerance was "
            f"found in this corpus to compare against (unlike the ATmega32 tool's grounded "
            f"2% threshold); this is a generic >1% flag, not a sourced limit."
        ) if abs(error_percent) > 1.0 else None,
    }

# Channel-to-bit-position mapping is documented for the first couple of
# channels only ("[1:0] is used for ADC1_CH0, [3:2] is used for ADC1_CH1,
# etc.") -- the pattern is presumably linear (channel n -> bits [2n+1:2n])
# but that's an inference from "etc.", not a value seen printed for every
# channel, and the register's actual bit-layout table (seen separately)
# didn't obviously confirm uniform packing all the way across for
# SENS_SAR_ATTEN2_REG specifically. Deliberately NOT building a
# channel-position calculator on that basis -- see FINDINGS.md / README
# for why guessing past what's actually confirmed defeats the point of
# this whole exercise.
