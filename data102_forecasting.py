"""
Data 102 Enhanced Forecasting Module
=====================================
Applies statistical rigor from Berkeley's Data 102 course to forecasting:
1. Benjamini-Hochberg (Multiple Testing Correction)
2. Causal Inference (DAG Analysis)
3. Concentration Inequalities (Calibration)
4. Loss Function Optimization
"""

import re
from typing import List, Dict, Tuple, Optional
import math
from dataclasses import dataclass


@dataclass
class Signal:
    """A piece of evidence/information from research"""
    content: str
    source: str
    confidence: float  # 0-1, how confident are we in this signal
    p_value: float  # probability this is noise/false discovery
    

@dataclass
class CausalAnalysis:
    """Result of causal inference analysis"""
    treatment: str  # X (the claimed cause)
    outcome: str  # Y (the predicted effect)
    confounders: List[str]  # Z (variables causing both X and Y)
    is_causal: bool  # True if X->Y is likely causal, False if spurious
    dag_description: str  # Text description of the causal graph


class BenjaminiHochbergFilter:
    """
    Multiple Testing Correction
    
    Problem: If you search for 100 "reasons SPY will go up," 5 will look 
    significant by pure chance.
    
    Solution: Only keep signals that survive statistical threshold.
    """
    
    @staticmethod
    def filter_signals(signals: List[Signal], alpha: float = 0.05) -> List[Signal]:
        """
        Apply Benjamini-Hochberg procedure to control False Discovery Rate.
        
        Args:
            signals: List of Signal objects with p_values
            alpha: FDR threshold (default 0.05 = 5% false discovery rate)
            
        Returns:
            List of signals that survive the threshold
        """
        if not signals:
            return []
        
        # Sort by p-value (ascending)
        sorted_signals = sorted(signals, key=lambda x: x.p_value)
        m = len(sorted_signals)
        
        verified_signals = []
        for i, signal in enumerate(sorted_signals):
            # Benjamini-Hochberg threshold
            threshold = ((i + 1) / m) * alpha
            
            if signal.p_value <= threshold:
                verified_signals.append(signal)
            else:
                # Once we fail, all subsequent fail (they're sorted)
                break
                
        return verified_signals
    
    @staticmethod
    def assign_p_values(signals: List[Signal], 
                       total_sources_checked: int) -> List[Signal]:
        """
        Estimate p-values for signals based on source diversity.
        
        Higher p-value (more likely to be noise) if:
        - Only one source mentions it
        - Low confidence signal
        - Many sources checked but only few mention it
        """
        for signal in signals:
            # Simple heuristic: p_value based on source uniqueness
            # If 1 out of 10 sources mention it: high p-value (likely noise)
            # If 8 out of 10 sources mention it: low p-value (likely real)
            
            # This is a simplified model - in production you'd want more sophistication
            source_mentions = 1  # Simplified: assume each signal is one source
            p_value = 1.0 - (source_mentions / max(total_sources_checked, 1))
            
            # Adjust by confidence
            p_value = p_value * (1.0 - signal.confidence)
            
            signal.p_value = max(0.01, min(0.99, p_value))  # Clamp to reasonable range
            
        return signals


class CausalInferenceAgent:
    """
    Causal DAG Analysis
    
    Problem: Correlation ≠ Causation. Is X causing Y, or is confounder Z 
    causing both?
    
    Solution: Force LLM to identify causal structure before forecasting.
    """
    
    @staticmethod
    async def analyze_causality(question: str, 
                               research: str, 
                               llm) -> CausalAnalysis:
        """
        Have LLM identify the causal structure.
        
        Returns:
            CausalAnalysis with DAG structure
        """
        prompt = f"""
You are a causal inference expert using Directed Acyclic Graphs (DAGs).

QUESTION: {question}

RESEARCH: {research}

Analyze the causal structure:

1. TREATMENT (X): What is the main variable being discussed as a "cause"?
2. OUTCOME (Y): What is the predicted effect/outcome?
3. CONFOUNDERS (Z): What variables might be causing BOTH X and Y (making the correlation spurious)?
4. CAUSAL JUDGMENT: Is X truly causing Y, or is this correlation driven by confounders?

Respond in this format:
TREATMENT: [variable]
OUTCOME: [variable]
CONFOUNDERS: [list variables, or "none"]
IS_CAUSAL: [YES/NO]
REASONING: [explain the DAG structure]

Example:
Question: "Will S&P 500 rise because of 2026 Tax Act?"
TREATMENT: 2026 Tax Act passage
OUTCOME: S&P 500 increase
CONFOUNDERS: General economic optimism, Fed monetary policy
IS_CAUSAL: NO
REASONING: The Tax Act and market rise are both driven by underlying economic strength. 
The act itself may not cause the rise - they're parallel effects of the same confounder.
"""
        
        response = await llm.invoke(prompt)
        
        # Parse response
        treatment = CausalInferenceAgent._extract_field(response, "TREATMENT")
        outcome = CausalInferenceAgent._extract_field(response, "OUTCOME")
        confounders_str = CausalInferenceAgent._extract_field(response, "CONFOUNDERS")
        is_causal_str = CausalInferenceAgent._extract_field(response, "IS_CAUSAL")
        reasoning = CausalInferenceAgent._extract_field(response, "REASONING")
        
        confounders = [c.strip() for c in confounders_str.split(",")] if confounders_str.lower() != "none" else []
        is_causal = "yes" in is_causal_str.lower()
        
        return CausalAnalysis(
            treatment=treatment or "Unknown",
            outcome=outcome or "Unknown",
            confounders=confounders,
            is_causal=is_causal,
            dag_description=reasoning or response
        )
    
    @staticmethod
    def _extract_field(text: str, field_name: str) -> str:
        """Extract field value from structured response"""
        pattern = rf"{field_name}:\s*(.+?)(?:\n|$)"
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else ""


class ConcentrationCalibrator:
    """
    Hoeffding's Inequality for Confidence Calibration
    
    Problem: If you only found 3 articles, your confidence should be lower
    than if you found 100 articles.
    
    Solution: Shrink extreme predictions toward 50% based on sample size.
    """
    
    @staticmethod
    def calibrate_prediction(raw_prediction: float, 
                            sample_size: int,
                            min_sample_for_confidence: int = 20) -> float:
        """
        Shrink prediction toward 50% (uninformed prior) if sample size is small.
        
        Uses Hoeffding's inequality logic:
        P(|sample_mean - true_mean| ≥ ε) ≤ 2*exp(-2*n*ε²)
        
        Args:
            raw_prediction: Initial forecast (0-1)
            sample_size: Number of sources/evidence pieces
            min_sample_for_confidence: Sample size needed for full confidence
            
        Returns:
            Calibrated prediction shrunk toward 50%
        """
        if sample_size >= min_sample_for_confidence:
            return raw_prediction
        
        # Calculate shrinkage factor based on sample size
        # Smaller sample = more shrinkage toward 50%
        shrinkage_factor = sample_size / min_sample_for_confidence
        
        # Shrink toward 0.5 (uninformed prior)
        calibrated = 0.5 + (raw_prediction - 0.5) * shrinkage_factor
        
        return calibrated
    
    @staticmethod
    def get_confidence_interval(prediction: float, 
                               sample_size: int,
                               confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval using Hoeffding bound.
        
        Returns:
            (lower_bound, upper_bound) for the prediction
        """
        if sample_size <= 1:
            return (0.0, 1.0)  # Maximum uncertainty
        
        # Hoeffding bound: ε for confidence level
        # P(|X̄ - μ| ≥ ε) ≤ 2*exp(-2*n*ε²)
        # Solve for ε: ε = sqrt(-ln(α/2) / (2*n))
        alpha = 1 - confidence_level
        epsilon = math.sqrt(-math.log(alpha / 2) / (2 * sample_size))
        
        lower = max(0.0, prediction - epsilon)
        upper = min(1.0, prediction + epsilon)
        
        return (lower, upper)


class MetaculusLossOptimizer:
    """
    Loss Function Optimization for Logarithmic Scoring
    
    Problem: Metaculus uses log score. Being "wrong and confident" (99% on 
    a No) is catastrophic.
    
    Solution: Adjust predictions to minimize expected log loss.
    """
    
    @staticmethod
    def calculate_log_score(prediction: float, outcome: bool) -> float:
        """
        Calculate Metaculus log score.
        
        Score = ln(p) if Yes, ln(1-p) if No
        Lower (more negative) is worse.
        """
        p = max(0.01, min(0.99, prediction))  # Clamp to avoid log(0)
        
        if outcome:  # Question resolved Yes
            return math.log(p)
        else:  # Question resolved No
            return math.log(1 - p)
    
    @staticmethod
    def risk_averse_adjustment(prediction: float, 
                              evidence_strength: float) -> float:
        """
        Pull back from extreme predictions unless evidence is overwhelming.
        
        Args:
            prediction: Raw prediction (0-1)
            evidence_strength: How strong is the evidence (0-1)
                1.0 = overwhelming, causal, verified
                0.5 = moderate evidence
                0.0 = weak/speculative
                
        Returns:
            Adjusted prediction that's more conservative
        """
        # Only allow extreme predictions if evidence is very strong
        if evidence_strength < 0.8:
            # Shrink away from extremes
            if prediction > 0.5:
                # Pull back from high confidence
                max_allowed = 0.5 + (evidence_strength * 0.45)  # Max 0.95 if evidence=1.0
                prediction = min(prediction, max_allowed)
            else:
                # Pull back from low confidence
                min_allowed = 0.5 - (evidence_strength * 0.45)  # Min 0.05 if evidence=1.0
                prediction = max(prediction, min_allowed)
        
        return prediction
    
    @staticmethod
    def expected_log_loss(prediction: float, 
                         estimated_true_prob: float) -> float:
        """
        Calculate expected log loss given your estimate of true probability.
        
        Useful for: "Should I predict 70% or 60%? Which minimizes expected loss?"
        """
        # Expected loss = p*log(pred) + (1-p)*log(1-pred)
        # where p is the true probability
        p_true = estimated_true_prob
        pred = max(0.01, min(0.99, prediction))
        
        loss = p_true * math.log(pred) + (1 - p_true) * math.log(1 - pred)
        return loss


class EnhancedForecaster:
    """
    Combines all Data 102 techniques into a unified forecasting pipeline.
    """
    
    def __init__(self, llm):
        self.llm = llm
        self.bh_filter = BenjaminiHochbergFilter()
        self.causal_agent = CausalInferenceAgent()
        self.calibrator = ConcentrationCalibrator()
        self.loss_optimizer = MetaculusLossOptimizer()
    
    async def enhanced_forecast(self,
                               question: str,
                               research: str,
                               raw_prediction: float,
                               signals: Optional[List[Signal]] = None,
                               sample_size: int = 10) -> Dict:
        """
        Apply all Data 102 enhancements to a forecast.
        
        Returns:
            Dict with:
                - adjusted_prediction: Final calibrated forecast
                - causal_analysis: DAG analysis results
                - verified_signals: Signals that survived BH filter
                - confidence_interval: Uncertainty bounds
                - reasoning: Explanation of adjustments
        """
        # Step 1: Multiple Testing Correction (if signals provided)
        verified_signals = []
        if signals:
            signals = self.bh_filter.assign_p_values(signals, sample_size)
            verified_signals = self.bh_filter.filter_signals(signals, alpha=0.05)
        
        # Step 2: Causal Inference
        causal_analysis = await self.causal_agent.analyze_causality(
            question, research, self.llm
        )
        
        # Step 3: Calculate evidence strength
        evidence_strength = 0.5  # Default moderate
        
        if causal_analysis.is_causal and len(verified_signals) >= 3:
            evidence_strength = 0.9  # Strong: causal + multiple verified signals
        elif causal_analysis.is_causal or len(verified_signals) >= 3:
            evidence_strength = 0.7  # Moderate-strong
        elif len(causal_analysis.confounders) > 0:
            evidence_strength = 0.3  # Weak: confounding present
        
        # Step 4: Loss Function Optimization
        risk_adjusted = self.loss_optimizer.risk_averse_adjustment(
            raw_prediction, evidence_strength
        )
        
        # Step 5: Concentration Inequality Calibration
        final_prediction = self.calibrator.calibrate_prediction(
            risk_adjusted, 
            sample_size
        )
        
        # Step 6: Confidence Interval
        conf_interval = self.calibrator.get_confidence_interval(
            final_prediction, sample_size
        )
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            raw_prediction, 
            final_prediction,
            causal_analysis,
            len(verified_signals),
            sample_size,
            evidence_strength
        )
        
        return {
            "adjusted_prediction": final_prediction,
            "raw_prediction": raw_prediction,
            "causal_analysis": causal_analysis,
            "verified_signals": verified_signals,
            "confidence_interval": conf_interval,
            "evidence_strength": evidence_strength,
            "reasoning": reasoning
        }
    
    def _generate_reasoning(self, raw: float, adjusted: float, 
                           causal: CausalAnalysis, num_signals: int,
                           sample_size: int, evidence_strength: float) -> str:
        """Generate explanation of adjustments"""
        reasoning = f"""
📊 DATA 102 ENHANCEMENTS APPLIED:

Raw LLM Prediction: {raw:.1%}
Final Adjusted Prediction: {adjusted:.1%}

🔬 CAUSAL ANALYSIS:
Treatment: {causal.treatment}
Outcome: {causal.outcome}
Confounders: {', '.join(causal.confounders) if causal.confounders else 'None identified'}
Causal Relationship: {'YES - Treatment likely causes outcome' if causal.is_causal else 'NO - Correlation may be spurious'}

📈 MULTIPLE TESTING:
Verified Signals: {num_signals}
(Signals that survived Benjamini-Hochberg correction)

📉 CALIBRATION:
Sample Size: {sample_size} sources
Evidence Strength: {evidence_strength:.1%}
Adjustment: {'Conservative due to small sample/weak causality' if adjusted < raw else 'Maintained confidence with strong evidence'}

🎯 RISK MANAGEMENT:
Optimized for Metaculus log score to avoid overconfidence penalty.
"""
        return reasoning.strip()


# Example usage
if __name__ == "__main__":
    # Demo the functions
    signals = [
        Signal("Fed signals rate hold", "Reuters", 0.8, 0.0),
        Signal("Markets rally on tech earnings", "Bloomberg", 0.7, 0.0),
        Signal("Unverified rumor about policy change", "Random blog", 0.3, 0.0),
    ]
    
    bh = BenjaminiHochbergFilter()
    signals = bh.assign_p_values(signals, total_sources_checked=10)
    verified = bh.filter_signals(signals)
    
    print(f"Original signals: {len(signals)}")
    print(f"Verified signals: {len(verified)}")
    
    # Demo calibration
    calibrator = ConcentrationCalibrator()
    raw_pred = 0.85
    calibrated = calibrator.calibrate_prediction(raw_pred, sample_size=5)
    print(f"\nRaw prediction: {raw_pred:.1%}")
    print(f"Calibrated (n=5): {calibrated:.1%}")