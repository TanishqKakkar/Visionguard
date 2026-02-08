"""
Intelligent Suspicious Logic Engine
====================================
This is the "AI Controller" of the surveillance system.

Features:
- Confidence threshold logic (prevents false alarms)
- Temporal voting buffer (smooths predictions over time)
- Event stabilization logic (requires consistent predictions before alerting)

Without this → model becomes just a classifier, not a surveillance system.
"""

import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class AlertLevel(Enum):
    """Alert levels for the surveillance system"""
    SAFE = "SAFE"
    UNCERTAIN = "UNCERTAIN"
    SUSPICIOUS = "SUSPICIOUS"


@dataclass
class PredictionResult:
    """Structured prediction result"""
    predicted_class: str
    confidence: float
    all_probabilities: Dict[str, float]
    timestamp: float
    raw_prediction_index: int


@dataclass
class StabilizedResult:
    """Stabilized prediction after temporal voting"""
    alert_level: AlertLevel
    stabilized_class: str
    stabilized_confidence: float
    vote_count: int
    buffer_size: int
    is_stable: bool
    all_probabilities: Dict[str, float]


class SuspiciousLogicEngine:
    """
    Intelligent Suspicious Logic Engine
    
    This engine processes raw model predictions and applies:
    1. Confidence threshold filtering
    2. Temporal voting buffer (majority voting over time window)
    3. Event stabilization (requires N consecutive suspicious predictions)
    """
    
    def __init__(
        self,
        class_names: List[str],
        confidence_threshold: float = 0.65,
        high_confidence_threshold: float = 0.85,
        temporal_buffer_size: int = 5,
        stabilization_required: int = 3,
        uncertain_threshold: float = 0.50,
        suspicious_classes: Optional[List[str]] = None
    ):
        """
        Initialize the Suspicious Logic Engine
        
        Args:
            class_names: List of class names (e.g., ['Emergency', 'Robbery', ...])
            confidence_threshold: Minimum confidence for non-suspicious classes (default: 0.65)
            high_confidence_threshold: Minimum confidence for high-risk classes like Robbery (default: 0.85)
            temporal_buffer_size: Number of recent predictions to keep in buffer (default: 5)
            stabilization_required: Number of consecutive suspicious predictions needed (default: 3)
            uncertain_threshold: Below this confidence, treat as uncertain (default: 0.50)
            suspicious_classes: Classes that are considered suspicious (default: all except 'Normal')
        """
        self.class_names = class_names
        self.confidence_threshold = confidence_threshold
        self.high_confidence_threshold = high_confidence_threshold
        self.temporal_buffer_size = temporal_buffer_size
        self.stabilization_required = stabilization_required
        self.uncertain_threshold = uncertain_threshold
        
        # Define suspicious classes (threat classes)
        if suspicious_classes is None:
            # Default: all classes except Normal (if it exists)
            self.suspicious_classes = [c for c in class_names if c != 'Normal']
        else:
            self.suspicious_classes = suspicious_classes
        
        # Temporal voting buffer: stores recent predictions
        self.prediction_buffer: deque = deque(maxlen=temporal_buffer_size)
        
        # Stabilization state
        self.consecutive_suspicious_count = 0
        self.last_stabilized_class = None
        self.last_stabilized_confidence = 0.0
        
        # Statistics
        self.total_predictions = 0
        self.filtered_predictions = 0
        self.stabilized_alerts = 0
        
    def _apply_confidence_threshold(
        self, 
        predicted_class: str, 
        confidence: float,
        all_probabilities: Dict[str, float]
    ) -> Tuple[str, float, bool]:
        """
        Apply confidence threshold logic
        IMPORTANT: Only Emergency class has threshold filtering
        All other classes (Robbery, Violence, Weaponized, Trespassing) pass through regardless of confidence
        
        Returns:
            (filtered_class, filtered_confidence, is_valid)
        """
        # ONLY Emergency class requires threshold filtering
        # All other classes (Robbery, Violence, Weaponized, Trespassing) are shown even with low confidence
        if predicted_class == 'Emergency':
            # Emergency requires very high confidence threshold (0.95 = 95%) to prevent false positives
            emergency_threshold = 0.95  # 95% confidence required for Emergency
            if confidence < emergency_threshold:
                # Check if second-best prediction is too close
                sorted_probs = sorted(all_probabilities.items(), key=lambda x: x[1], reverse=True)
                if len(sorted_probs) > 1:
                    top_conf = sorted_probs[0][1]
                    second_conf = sorted_probs[1][1]
                    diff = top_conf - second_conf
                    
                    # If confidence is too low OR predictions are too close, filter it
                    if confidence < emergency_threshold or diff < 0.25:
                        # Return as 'Normal' (safe) with high confidence
                        return 'Normal', 1.0, False
            else:
                # Confidence is >= 0.95, but still check if Emergency prediction is too close to other classes
                # This prevents false positives when model is uncertain (even at high confidence)
                sorted_probs = sorted(all_probabilities.items(), key=lambda x: x[1], reverse=True)
                if len(sorted_probs) > 1:
                    top_conf = sorted_probs[0][1]
                    second_conf = sorted_probs[1][1]
                    diff = top_conf - second_conf
                    
                    # If Emergency confidence is not significantly higher than second-best, filter it
                    # Even at 95%+ confidence, require at least 30% difference from second-best
                    if diff < 0.30:
                        return 'Normal', 1.0, False
        
        # For all other classes (Robbery, Violence, Weaponized, Trespassing):
        # NO threshold filtering - always show them regardless of confidence
        # This allows low-confidence detections to be visible
        
        # Valid prediction (no filtering for non-Emergency classes)
        return predicted_class, confidence, True
    
    def _temporal_voting(self) -> Tuple[str, float, Dict[str, float]]:
        """
        Apply temporal voting buffer (majority voting over recent predictions)
        
        Returns:
            (voted_class, voted_confidence, averaged_probabilities)
        """
        if len(self.prediction_buffer) == 0:
            return 'Normal', 1.0, {}
        
        # Count votes for each class
        class_votes: Dict[str, List[float]] = {}
        class_probs_sum: Dict[str, float] = {}
        
        for pred in self.prediction_buffer:
            cls = pred.predicted_class
            conf = pred.confidence
            
            if cls not in class_votes:
                class_votes[cls] = []
                class_probs_sum[cls] = {}
            
            class_votes[cls].append(conf)
            
            # Sum probabilities for averaging
            for prob_class, prob_val in pred.all_probabilities.items():
                if prob_class not in class_probs_sum[cls]:
                    class_probs_sum[cls][prob_class] = 0.0
                class_probs_sum[cls][prob_class] += prob_val
        
        # Find class with most votes (weighted by confidence)
        best_class = None
        best_score = 0.0
        
        for cls, confidences in class_votes.items():
            # Weighted vote: sum of confidences
            score = sum(confidences)
            if score > best_score:
                best_score = score
                best_class = cls
        
        if best_class is None:
            return 'Normal', 1.0, {}
        
        # Calculate averaged confidence and probabilities
        avg_confidence = np.mean(class_votes[best_class])
        
        # Average probabilities
        num_votes = len(class_votes[best_class])
        avg_probs = {
            k: v / num_votes 
            for k, v in class_probs_sum[best_class].items()
        }
        
        return best_class, avg_confidence, avg_probs
    
    def _check_stabilization(
        self, 
        voted_class: str, 
        voted_confidence: float
    ) -> Tuple[bool, AlertLevel]:
        """
        Check if prediction is stable enough to trigger alert
        
        Returns:
            (is_stable, alert_level)
        """
        is_suspicious = voted_class in self.suspicious_classes
        
        if is_suspicious:
            # Increment consecutive suspicious count
            self.consecutive_suspicious_count += 1
            
            # Check if we have enough consecutive suspicious predictions
            if self.consecutive_suspicious_count >= self.stabilization_required:
                # Stable suspicious event
                self.last_stabilized_class = voted_class
                self.last_stabilized_confidence = voted_confidence
                return True, AlertLevel.SUSPICIOUS
            else:
                # Not stable yet, but suspicious
                return False, AlertLevel.UNCERTAIN
        else:
            # Safe prediction - reset counter
            self.consecutive_suspicious_count = 0
            self.last_stabilized_class = voted_class
            self.last_stabilized_confidence = voted_confidence
            
            # Check confidence level
            if voted_confidence < self.uncertain_threshold:
                return True, AlertLevel.UNCERTAIN
            else:
                return True, AlertLevel.SAFE
    
    def process_prediction(
        self,
        predicted_class: str,
        confidence: float,
        all_probabilities: Dict[str, float],
        predicted_index: int,
        timestamp: Optional[float] = None
    ) -> StabilizedResult:
        """
        Process a raw model prediction through the intelligent logic engine
        
        Args:
            predicted_class: Raw predicted class from model
            confidence: Raw confidence score
            all_probabilities: All class probabilities
            predicted_index: Raw prediction index
            timestamp: Optional timestamp (defaults to current time)
        
        Returns:
            StabilizedResult with alert level and stabilized prediction
        """
        import time
        if timestamp is None:
            timestamp = time.time()
        
        self.total_predictions += 1
        
        # Step 1: Apply confidence threshold
        filtered_class, filtered_confidence, is_valid = self._apply_confidence_threshold(
            predicted_class, confidence, all_probabilities
        )
        
        if not is_valid:
            self.filtered_predictions += 1
        
        # Create prediction result
        pred_result = PredictionResult(
            predicted_class=filtered_class,
            confidence=filtered_confidence,
            all_probabilities=all_probabilities,  # Keep original for averaging
            timestamp=timestamp,
            raw_prediction_index=predicted_index
        )
        
        # Step 2: Add to temporal buffer
        self.prediction_buffer.append(pred_result)
        
        # Step 3: Apply temporal voting
        voted_class, voted_confidence, avg_probs = self._temporal_voting()
        
        # Step 4: Check stabilization
        is_stable, alert_level = self._check_stabilization(voted_class, voted_confidence)
        
        if is_stable and alert_level == AlertLevel.SUSPICIOUS:
            self.stabilized_alerts += 1
        
        return StabilizedResult(
            alert_level=alert_level,
            stabilized_class=voted_class,
            stabilized_confidence=voted_confidence,
            vote_count=len(self.prediction_buffer),
            buffer_size=self.temporal_buffer_size,
            is_stable=is_stable,
            all_probabilities=avg_probs if avg_probs else all_probabilities
        )
    
    def reset(self):
        """Reset the engine state (clear buffer, reset counters)"""
        self.prediction_buffer.clear()
        self.consecutive_suspicious_count = 0
        self.last_stabilized_class = None
        self.last_stabilized_confidence = 0.0
    
    def get_statistics(self) -> Dict:
        """Get engine statistics"""
        return {
            'total_predictions': self.total_predictions,
            'filtered_predictions': self.filtered_predictions,
            'filter_rate': self.filtered_predictions / max(self.total_predictions, 1),
            'stabilized_alerts': self.stabilized_alerts,
            'buffer_size': len(self.prediction_buffer),
            'consecutive_suspicious': self.consecutive_suspicious_count
        }


# Example usage
if __name__ == "__main__":
    # Test the engine
    class_names = ['Emergency', 'Robbery', 'Trespassing', 'Violence', 'Weaponized']
    
    engine = SuspiciousLogicEngine(
        class_names=class_names,
        confidence_threshold=0.65,
        high_confidence_threshold=0.85,
        temporal_buffer_size=5,
        stabilization_required=3
    )
    
    # Simulate some predictions
    print("Testing SuspiciousLogicEngine...")
    
    # Test 1: Low confidence prediction (should be filtered)
    result1 = engine.process_prediction(
        predicted_class='Robbery',
        confidence=0.60,
        all_probabilities={'Emergency': 0.1, 'Robbery': 0.6, 'Trespassing': 0.1, 'Violence': 0.1, 'Weaponized': 0.1},
        predicted_index=1
    )
    print(f"Test 1 - Low confidence Robbery: {result1.alert_level.value}, Class: {result1.stabilized_class}")
    
    # Test 2: High confidence suspicious (should trigger after stabilization)
    for i in range(5):
        result2 = engine.process_prediction(
            predicted_class='Violence',
            confidence=0.90,
            all_probabilities={'Emergency': 0.05, 'Robbery': 0.05, 'Trespassing': 0.05, 'Violence': 0.90, 'Weaponized': 0.05},
            predicted_index=3
        )
        print(f"Test 2 - Iteration {i+1}: {result2.alert_level.value}, Stable: {result2.is_stable}, Count: {result2.vote_count}")
    
    print("\nStatistics:", engine.get_statistics())

