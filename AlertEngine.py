"""
Alert Engine for CCTV Surveillance System
==========================================
Handles sound and visual alerts based on threat level.

Alert Levels:
- SAFE → No sound, green indicator
- UNCERTAIN → Short beep, yellow indicator
- SUSPICIOUS → Loud siren + red flashing, emergency alert
"""

import time
import threading
import platform
from enum import Enum
from typing import Optional, Callable
from dataclasses import dataclass


class AlertLevel(Enum):
    """Alert levels"""
    SAFE = "SAFE"
    UNCERTAIN = "UNCERTAIN"
    SUSPICIOUS = "SUSPICIOUS"


@dataclass
class AlertState:
    """Current alert state"""
    level: AlertLevel
    is_active: bool
    start_time: Optional[float] = None
    duration: float = 0.0


class AlertEngine:
    """
    Alert Engine for CCTV Surveillance System
    
    Features:
    - Sound alerts (beep for uncertain, siren for suspicious)
    - Visual alert state management
    - Alert duration tracking
    - Thread-safe alert handling
    """
    
    def __init__(
        self,
        uncertain_beep_duration: float = 0.2,
        uncertain_beep_frequency: int = 800,
        suspicious_siren_duration: float = 2.0,
        suspicious_siren_frequency: int = 1000,
        alert_cooldown: float = 3.0
    ):
        """
        Initialize Alert Engine
        
        Args:
            uncertain_beep_duration: Duration of uncertain beep in seconds (default: 0.2)
            uncertain_beep_frequency: Frequency of uncertain beep in Hz (default: 800)
            suspicious_siren_duration: Duration of suspicious siren in seconds (default: 2.0)
            suspicious_siren_frequency: Base frequency of suspicious siren in Hz (default: 1000)
            alert_cooldown: Cooldown period between alerts in seconds (default: 3.0)
        """
        self.uncertain_beep_duration = uncertain_beep_duration
        self.uncertain_beep_frequency = uncertain_beep_frequency
        self.suspicious_siren_duration = suspicious_siren_duration
        self.suspicious_siren_frequency = suspicious_siren_frequency
        self.alert_cooldown = alert_cooldown
        
        self.current_state = AlertState(level=AlertLevel.SAFE, is_active=False)
        self.last_alert_time = 0.0
        self.alert_lock = threading.Lock()
        
        # Platform-specific sound generation
        self.platform = platform.system()
        self.sound_enabled = True
        
        # Callback for visual alerts (set by GUI)
        self.visual_alert_callback: Optional[Callable[[AlertLevel, bool], None]] = None
    
    def set_visual_alert_callback(self, callback: Callable[[AlertLevel, bool], None]):
        """Set callback for visual alerts (called from GUI)"""
        self.visual_alert_callback = callback
    
    def _play_beep_windows(self, frequency: int, duration: float):
        """Play beep on Windows"""
        try:
            import winsound
            winsound.Beep(frequency, int(duration * 1000))
        except Exception as e:
            print(f"Warning: Could not play beep: {e}")
    
    def _play_beep_linux(self, frequency: int, duration: float):
        """Play beep on Linux"""
        try:
            import os
            # Use system beep command
            os.system(f'beep -f {frequency} -l {int(duration * 1000)}')
        except Exception as e:
            print(f"Warning: Could not play beep: {e}")
    
    def _play_beep_macos(self, frequency: int, duration: float):
        """Play beep on macOS"""
        try:
            import os
            # Use say command or system beep
            os.system(f'afplay /System/Library/Sounds/Glass.aiff')
        except Exception as e:
            print(f"Warning: Could not play beep: {e}")
    
    def _play_siren(self, duration: float):
        """Play siren sound (repeated beeps)"""
        if not self.sound_enabled:
            return
        
        def siren_thread():
            start_time = time.time()
            beep_interval = 0.3  # Beep every 0.3 seconds
            beep_count = 0
            
            while (time.time() - start_time) < duration:
                # Vary frequency for siren effect
                freq = self.suspicious_siren_frequency + (beep_count % 3) * 200
                
                if self.platform == "Windows":
                    self._play_beep_windows(freq, 0.2)
                elif self.platform == "Linux":
                    self._play_beep_linux(freq, 0.2)
                elif self.platform == "Darwin":  # macOS
                    self._play_beep_macos(freq, 0.2)
                
                beep_count += 1
                time.sleep(beep_interval)
        
        thread = threading.Thread(target=siren_thread, daemon=True)
        thread.start()
    
    def _play_uncertain_beep(self):
        """Play short beep for uncertain state"""
        if not self.sound_enabled:
            return
        
        def beep_thread():
            if self.platform == "Windows":
                self._play_beep_windows(self.uncertain_beep_frequency, self.uncertain_beep_duration)
            elif self.platform == "Linux":
                self._play_beep_linux(self.uncertain_beep_frequency, self.uncertain_beep_duration)
            elif self.platform == "Darwin":  # macOS
                self._play_beep_macos(self.uncertain_beep_frequency, self.uncertain_beep_duration)
        
        thread = threading.Thread(target=beep_thread, daemon=True)
        thread.start()
    
    def trigger_alert(self, level: AlertLevel, force: bool = False):
        """
        Trigger alert based on level
        
        Args:
            level: Alert level (SAFE, UNCERTAIN, SUSPICIOUS)
            force: Force alert even if in cooldown (default: False)
        """
        with self.alert_lock:
            current_time = time.time()
            
            # Check cooldown (unless forced)
            if not force and (current_time - self.last_alert_time) < self.alert_cooldown:
                return
            
            # Update state
            if level == AlertLevel.SAFE:
                self.current_state = AlertState(
                    level=level,
                    is_active=False,
                    start_time=None,
                    duration=0.0
                )
                # No sound for safe
                if self.visual_alert_callback:
                    self.visual_alert_callback(level, False)
                return
            
            # For uncertain and suspicious, check if we should alert
            if level == AlertLevel.UNCERTAIN:
                # Play short beep
                self._play_uncertain_beep()
                self.current_state = AlertState(
                    level=level,
                    is_active=True,
                    start_time=current_time,
                    duration=self.uncertain_beep_duration
                )
                self.last_alert_time = current_time
                
                if self.visual_alert_callback:
                    self.visual_alert_callback(level, True)
            
            elif level == AlertLevel.SUSPICIOUS:
                # Play loud siren
                self._play_siren(self.suspicious_siren_duration)
                self.current_state = AlertState(
                    level=level,
                    is_active=True,
                    start_time=current_time,
                    duration=self.suspicious_siren_duration
                )
                self.last_alert_time = current_time
                
                if self.visual_alert_callback:
                    self.visual_alert_callback(level, True)
    
    def get_current_state(self) -> AlertState:
        """Get current alert state"""
        with self.alert_lock:
            return self.current_state
    
    def clear_alert(self):
        """Clear current alert"""
        with self.alert_lock:
            self.current_state = AlertState(
                level=AlertLevel.SAFE,
                is_active=False,
                start_time=None,
                duration=0.0
            )
            if self.visual_alert_callback:
                self.visual_alert_callback(AlertLevel.SAFE, False)
    
    def enable_sound(self):
        """Enable sound alerts"""
        self.sound_enabled = True
    
    def disable_sound(self):
        """Disable sound alerts"""
        self.sound_enabled = False


# Global alert engine instance (can be imported and used)
_global_alert_engine: Optional[AlertEngine] = None


def get_alert_engine() -> AlertEngine:
    """Get or create global alert engine instance"""
    global _global_alert_engine
    if _global_alert_engine is None:
        _global_alert_engine = AlertEngine()
    return _global_alert_engine


# Example usage
if __name__ == "__main__":
    print("Testing AlertEngine...")
    
    engine = AlertEngine()
    
    print("Playing uncertain beep...")
    engine.trigger_alert(AlertLevel.UNCERTAIN)
    time.sleep(1)
    
    print("Playing suspicious siren...")
    engine.trigger_alert(AlertLevel.SUSPICIOUS)
    time.sleep(3)
    
    print("Clearing alert...")
    engine.clear_alert()
    
    print("Test complete!")

