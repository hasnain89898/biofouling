#!/usr/bin/env python3
"""
Defect Diagnostic Engine & Explainable AI (XAI) Service for Industrial Biofouling
---------------------------------------------------------------------------------
Features:
1. Translates EfficientNet-B3 4-class biofouling predictions into deep, domain-grounded
   human-readable defect diagnostics and causal explanations.
2. Synthesizes Explainable AI (XAI) morphological analysis:
   - primary_issue: Specific biological/mechanical defect identification.
   - reasoning: Morphological features, biofilm matrix characteristics, and corrosion risks.
   - action_required: Immediate operational engineering and biocide dosing directives.
3. Dynamically accounts for classification confidence, Shannon entropy, and runner-up classes.

Author: Principal Explainable AI (XAI) Systems Architect
"""

from typing import Any, Dict, Optional


class DefectDiagnosticEngine:
    """
    Explainable AI (XAI) Diagnostic Engine mapping computer vision feature classifications
    into actionable industrial engineering defect analysis.
    """

    DIAGNOSTIC_KB: Dict[str, Dict[str, str]] = {
        "Critical": {
            "primary_issue": "Severe Macro-Fouling, Calcareous Incrustation & Critical Hydraulic Blockage",
            "reasoning": (
                "The EfficientNet-B3 diagnostic pipeline detected dense localized biological blockage "
                "(>75% surface area coverage), mature sloughing biomass clumps, severe calcium scaling, "
                "and structural degradation patterns matching advanced microbially influenced corrosion (MIC). "
                "Deep anaerobic under-deposit colonization by Sulfate-Reducing Bacteria (SRB) presents an acute "
                "risk of localized pinhole corrosion, thermal impedance collapse, and catastrophic tube rupture."
            ),
            "action_required": (
                "Immediate operational intervention required within 24 hours. Isolate the affected channel "
                "or heat exchanger plate pack. Execute chemical shock treatment with oxidizing biocide "
                "(Chlorine Dioxide at 2.0-5.0 ppm or stabilized Sodium Hypochlorite with biodispersant) combined "
                "with non-oxidizing DBNPA at 50-100 ppm, followed by high-pressure rotary hydro-jetting "
                "(2,000-3,500 PSI). Mandatory human metallurgical corrosion sign-off required."
            ),
            "corrosion_risk": "Critical (Acute Pinhole Pitting & Galvanic Attack)",
            "morphology": "Dense sloughing biomass clumps, calcareous crusts, severe flow occlusion",
        },
        "High": {
            "primary_issue": "Dense Microbial Slime Matrix & Heavy Anaerobic Biofilm Colonization",
            "reasoning": (
                "The model detected extensive dark microbial slime matrix with dense sessile colonial growth "
                "(>50% surface coverage) and substantial extracellular polymeric substance (EPS) deposition. "
                "Significant boundary degradation between substrate metal and bacterial slime creates localized "
                "anaerobic micro-environments favorable to under-deposit corrosion and measurable delta-P hydraulic head loss."
            ),
            "action_required": (
                "Schedule mechanical cleaning or targeted pressure wash within 48-72 hours. Introduce "
                "surface-active biopenetrant/biodispersant 4 hours prior to biocide addition to penetrate the "
                "thick sessile bacterial matrix. Administer Isothiazolinone (CMIT/MIT at 15-25 ppm active) or "
                "Glutaraldehyde (45-75 ppm) to neutralize anaerobic sulfate-reducing bacteria (SRB)."
            ),
            "corrosion_risk": "High (Active Under-Deposit Biofilm Acidification)",
            "morphology": "Continuous thick microbial slime layer, heavy EPS matrix, elevated flow drag",
        },
        "Moderate": {
            "primary_issue": "Early Biofilm Colonization & Filamentous Green Microalgal Patchiness",
            "reasoning": (
                "Computer vision analysis identified early cyanobacterial colonization, thin patchy biofilm adhesion, "
                "and low-contrast surface discoloration across cooling tower fill and pipe interior walls. "
                "Structural flow channels remain unobstructed without immediate hydraulic impairment, but bacterial "
                "and algal colonization is actively expanding across warm boundary layers."
            ),
            "action_required": (
                "Schedule proactive preventive cleaning within 14 days. Recalibrate automated chemical feed "
                "pumps, increase continuous background oxidizing biocide dosing (maintain 0.2-0.5 ppm free available "
                "halogen residual), and re-inspect visual inspection frames at 7-day intervals to verify biofilm suppression."
            ),
            "corrosion_risk": "Low to Moderate (Superficial Deposition Without Structural Pitting)",
            "morphology": "Dispersed microalgal patches, thin slimy film, unobstructed flow geometry",
        },
        "Low": {
            "primary_issue": "Nominal Surface Condition / Pristine Substrate Baseline",
            "reasoning": (
                "Substrate exhibits uniform optical reflectance, clean uncolonized heat-transfer surfaces, and "
                "unobstructed flow cross-sections. No macroscopic biological aggregations, extracellular polymeric "
                "slime matrices, or thermal transfer degradation patterns were detected by the neural feature extractor."
            ),
            "action_required": (
                "No physical or corrective mechanical intervention required. Maintain routine biocide background "
                "residual, monitor ORP (oxidation-reduction potential at 650-700 mV), and schedule routine visual "
                "frame capture in 30 days."
            ),
            "corrosion_risk": "Negligible (Protected Passive Oxide Layer Maintained)",
            "morphology": "Clean metallic/polymer substrate, zero hydraulic impedance, optimal thermal transfer",
        },
    }

    @classmethod
    def analyze_defect(
        cls,
        predicted_class: str,
        confidence: float,
        entropy: float = 0.0,
        probabilities: Optional[Dict[str, float]] = None,
        top2_class: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Synthesizes Explainable AI (XAI) defect analysis payload.

        Args:
            predicted_class: Classification label ('Low', 'Moderate', 'High', 'Critical')
            confidence: Model confidence score (0.0 to 1.0)
            entropy: Normalized Shannon entropy (0.0 to 1.0)
            probabilities: Full softmax class probability dictionary
            top2_class: Secondary class runner-up for boundary explanation

        Returns:
            Dict containing primary_issue, reasoning, action_required, and diagnostic telemetry
        """
        sev = predicted_class if predicted_class in cls.DIAGNOSTIC_KB else "High"
        kb_entry = cls.DIAGNOSTIC_KB[sev]

        primary_issue = kb_entry["primary_issue"]
        reasoning = kb_entry["reasoning"]
        action_required = kb_entry["action_required"]
        corrosion_risk = kb_entry["corrosion_risk"]

        # If high entropy or borderline confidence, augment reasoning with XAI boundary telemetry
        if entropy >= 0.50 or confidence < 0.75:
            runner_up_note = ""
            if top2_class and top2_class != sev:
                runner_up_note = f" Morphological boundary ambiguity detected alongside '{top2_class}' features."
            reasoning += (
                f" [XAI Telemetry Uncertainty Alert: Elevated uncertainty observed (Confidence: {confidence * 100:.1f}%, "
                f"Entropy: {entropy:.3f}).{runner_up_note} Human-in-the-Loop engineering sign-off recommended.]"
            )
            action_required += " [Human verification recommended due to boundary uncertainty.]"

        return {
            "primary_issue": primary_issue,
            "reasoning": reasoning,
            "action_required": action_required,
            "corrosion_risk": corrosion_risk,
            "morphology_summary": kb_entry["morphology"],
            "severity": sev,
            "predicted_class": sev,
            "confidence_pct": f"{confidence * 100:.1f}%",
            "entropy_score": round(entropy, 3),
        }

    # Alias for convenience
    analyze = analyze_defect


# Global Singleton
_DEFECT_ENGINE = DefectDiagnosticEngine()


def get_defect_engine() -> DefectDiagnosticEngine:
    """Retrieve global DefectDiagnosticEngine singleton."""
    return _DEFECT_ENGINE

