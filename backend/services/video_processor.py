import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional
import onnxruntime as ort

from backend.services.guardrail import get_guardrail
from backend.services.inference import ONNXInferenceTTAWrapper
from backend.services.defect_engine import get_defect_engine

class IndustrialVideoProcessor:
    """
    Processes video feeds from industrial drones or ROVs, applying robust
    real-world augmentations to handle challenging lighting and underwater
    conditions, and runs the extracted frames through the ONNX inference pipeline.
    """
    def __init__(
        self,
        session: ort.InferenceSession,
        val_transforms: Any,
        input_name: str,
        output_name: str,
        idx_to_label: Dict[int, str]
    ):
        self.session = session
        self.val_transforms = val_transforms
        self.input_name = input_name
        self.output_name = output_name
        self.idx_to_label = idx_to_label
        self.guardrail = get_guardrail()

    def apply_robust_augmentations(self, frame: np.ndarray) -> np.ndarray:
        """
        Applies aggressive real-world augmentations (histogram equalization,
        contrast adjustment, noise filtering) to handle dark, wet, or blurry frames.
        Args:
            frame: RGB numpy array
        Returns:
            Enhanced RGB numpy array
        """
        # Convert to LAB for Contrast Limited Adaptive Histogram Equalization (CLAHE)
        # which improves local contrast without blowing out the global image.
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge channels and convert back to RGB
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        
        # Apply Bilateral Filter for noise removal. It's highly effective for
        # underwater/industrial camera noise while preserving sharp edges.
        denoised = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)
        
        return denoised

    def process_video(
        self,
        video_path: str,
        fps_sample_rate: float = 1.0,
        threshold: float = 0.75
    ) -> Dict[str, Any]:
        """
        Processes an MP4/AVI video, extracts frames at `fps_sample_rate`,
        runs domain validation, applies TTA preprocessing, and aggregates the results.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0  # fallback in case FPS cannot be detected
            
        frame_interval = max(1, int(round(fps / fps_sample_rate)))
        
        timeline = []
        critical_alerts = []
        max_severity_idx = 0
        max_severity_label = self.idx_to_label.get(0, "Low")
        
        frame_idx = 0
        success = True
        
        while success:
            success, frame_bgr = cap.read()
            if not success:
                break
                
            if frame_idx % frame_interval == 0:
                timestamp = frame_idx / fps
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                
                # 1. Industrial Domain Guardrail
                gr_result = self.guardrail.validate(frame_rgb)
                
                if gr_result.is_valid:
                    # 2. Robust Test-Time Augmentations for Real-World Feeds
                    aug_frame = self.apply_robust_augmentations(frame_rgb)
                    
                    # 3. & 4. ONNX Inference with TTA
                    tta_wrapper = ONNXInferenceTTAWrapper(
                        session=self.session,
                        val_transforms=self.val_transforms,
                        input_name=self.input_name,
                        output_name=self.output_name,
                    )
                    consensus_probs, entropy, individual_probs = tta_wrapper.predict_tta(aug_frame)
                    probs_np = consensus_probs
                    
                    pred_idx = int(np.argmax(probs_np))
                    pred_class = self.idx_to_label[pred_idx]
                    confidence = float(probs_np[pred_idx])
                    
                    defect_engine = get_defect_engine()
                    frame_defect = defect_engine.analyze_defect(
                        predicted_class=pred_class,
                        confidence=confidence,
                        entropy=entropy,
                    )
                    
                    # Update max severity
                    if pred_idx > max_severity_idx:
                        max_severity_idx = pred_idx
                        max_severity_label = pred_class
                        
                    # Register alerts for High/Critical regions
                    if pred_class in ["High", "Critical"]:
                        critical_alerts.append({
                            "timestamp": round(timestamp, 2),
                            "severity": pred_class,
                            "confidence": round(confidence, 4),
                            "primary_issue": frame_defect.get("primary_issue"),
                        })
                    
                    timeline.append({
                        "timestamp": round(timestamp, 2),
                        "frame_index": frame_idx,
                        "valid_domain": True,
                        "predicted_class": pred_class,
                        "confidence": round(confidence, 4),
                        "gating_status": "AUTOMATED_DECISION" if confidence >= threshold else "HUMAN_EXPERT_REVIEW_REQUIRED",
                        "tta_applied": True,
                        "tta_consensus_score": round(confidence, 4),
                        "entropy": round(entropy, 4),
                        "defect_analysis": frame_defect,
                    })
                else:
                    timeline.append({
                        "timestamp": round(timestamp, 2),
                        "frame_index": frame_idx,
                        "valid_domain": False,
                        "rejection_reason": gr_result.rejection_reason
                    })
            
            frame_idx += 1
            
        cap.release()
        
        overall_defect = get_defect_engine().analyze_defect(
            predicted_class=max_severity_label,
            confidence=0.85,
            entropy=0.25,
        )
        
        return {
            "max_severity_detected": max_severity_label,
            "critical_alerts": critical_alerts,
            "frame_by_frame_results": timeline,
            "total_frames_processed": len(timeline),
            "video_fps": round(fps, 2),
            "defect_analysis": overall_defect,
        }
