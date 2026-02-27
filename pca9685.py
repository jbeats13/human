#!/usr/bin/env python3
"""
PCA9685 16-channel PWM / servo driver (I2C).

Compatible with ArduCAM PCA9685 usage and pan-tilt kits.
https://github.com/ArduCAM/PCA9685

Requires: smbus (on Raspberry Pi: enable I2C in raspi-config, then pip install smbus2)
"""

from __future__ import annotations

import time

# Default I2C address for PCA9685
PCA9685_ADDRESS = 0x40

# Register map
MODE1 = 0x00
MODE2 = 0x01
PRESCALE = 0xFE
LED0_ON_L = 0x06
LED0_ON_H = 0x07
LED0_OFF_L = 0x08
LED0_OFF_H = 0x09
ALLLED_ON_L = 0xFA
ALLLED_ON_H = 0xFB
ALLLED_OFF_L = 0xFC
ALLLED_OFF_H = 0xFD

# Servo: 50 Hz typical, pulse width ~1–2 ms (1000–2000 µs) for 0–180°
SERVO_FREQ_HZ = 50
# 12-bit resolution: 4096 steps; at 50 Hz period = 20 ms = 20000 µs
PWM_RESOLUTION = 4096
PWM_PERIOD_US = 1_000_000 // SERVO_FREQ_HZ  # 20000 µs


def _get_bus(bus_id: int):
    """Import and return SMBus. Prefer smbus2; fallback to smbus."""
    try:
        import smbus2

        return smbus2.SMBus(bus_id)
    except ImportError:
        try:
            import smbus

            return smbus.SMBus(bus_id)
        except ImportError:
            raise ImportError(
                "PCA9685 requires smbus. On Raspberry Pi: sudo apt install python3-smbus, or pip install smbus2"
            ) from None


class PCA9685:
    """16-channel PWM servo driver via I2C (PCA9685).

    Use setPWMFreq(50) for servos, then setServoPulse(channel, pulse_us) or setRotationAngle(channel, angle_0_180).
    """

    def __init__(
        self,
        address: int = PCA9685_ADDRESS,
        bus_id: int = 1,
        debug: bool = False,
    ):
        self._address = address
        self._debug = bool(debug)
        self._bus = _get_bus(bus_id)
        self._write(MODE1, 0x00)

    def _write(self, reg: int, value: int) -> None:
        """Write an 8-bit value to a register."""
        self._bus.write_byte_data(self._address, reg, value & 0xFF)
        if self._debug:
            print(f"I2C: Write 0x{value:02X} to reg 0x{reg:02X}")

    def _read(self, reg: int) -> int:
        """Read an 8-bit value from a register."""
        return self._bus.read_byte_data(self._address, reg)

    def setPWMFreq(self, freq: float) -> None:
        """Set PWM frequency in Hz (e.g. 50 for servos)."""
        prescale_val = 25_000_000.0  # 25 MHz
        prescale_val /= float(PWM_RESOLUTION)
        prescale_val /= float(freq)
        prescale_val -= 1.0
        prescale = round(prescale_val)
        if self._debug:
            print(f"Setting PWM frequency to {freq} Hz, prescale={prescale}")

        old_mode = self._read(MODE1)
        self._write(MODE1, (old_mode & 0x7F) | 0x10)  # sleep
        self._write(PRESCALE, prescale)
        self._write(MODE1, old_mode)
        time.sleep(0.005)
        self._write(MODE1, old_mode | 0x80)
        self._write(MODE2, 0x04)

    def setPWM(self, channel: int, on: int, off: int) -> None:
        """Set one channel: on/off are 0–4095 tick values."""
        if not 0 <= channel <= 15:
            raise ValueError("channel must be 0–15")
        base = LED0_ON_L + 4 * channel
        self._write(base + 0, on & 0xFF)
        self._write(base + 1, on >> 8)
        self._write(base + 2, off & 0xFF)
        self._write(base + 3, off >> 8)

    def setServoPulse(self, channel: int, pulse_us: float) -> None:
        """Set servo pulse width in microseconds (e.g. 1000–2000 for 0–180°). PWM frequency must be 50 Hz (call
        setPWMFreq(50) first).
        """
        # pulse_us / period_us = tick / 4096  =>  tick = pulse_us * 4096 / 20000
        tick = int(pulse_us * PWM_RESOLUTION / PWM_PERIOD_US)
        tick = max(0, min(PWM_RESOLUTION, tick))
        self.setPWM(channel, 0, tick)

    def setRotationAngle(self, channel: int, angle: float) -> None:
        """Set servo angle in degrees (0–180). Maps 0° → ~500 µs, 180° → ~2500 µs (adjust for your servo).
        """
        if not 0 <= angle <= 180:
            raise ValueError("angle must be 0–180")
        # Common mapping: 0° = 500 µs, 180° = 2500 µs
        pulse_us = 500 + (angle / 180.0) * 2000
        self.setServoPulse(channel, pulse_us)

    def start(self) -> None:
        """Restore output state (e.g. after init)."""
        self._write(MODE2, 0x04)

    def exit(self) -> None:
        """Turn off outputs (optional cleanup)."""
        self._write(MODE2, 0x00)

    def close(self) -> None:
        """Close I2C bus."""
        self._bus.close()


def create_pca9685(
    address: int = PCA9685_ADDRESS,
    bus_id: int = 1,
    servo_freq: int = SERVO_FREQ_HZ,
    debug: bool = False,
) -> PCA9685 | None:
    """Create and initialize a PCA9685 for servo use. Returns None if I2C/smbus is not available (e.g. not on a Pi).
    """
    try:
        pca = PCA9685(address=address, bus_id=bus_id, debug=debug)
        pca.setPWMFreq(servo_freq)
        pca.start()
        return pca
    except (ImportError, OSError) as e:
        if debug:
            print(f"PCA9685 not available: {e}")
        return None
