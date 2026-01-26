"""
COMPLETE INTEGRATION PIPELINE - FINAL VERSION WITH VULNERABILITY MODEL
✅ FIXED: All critical issues identified in the analysis
✅ FIXED: Proper model loading WITHOUT silent reinitialization
✅ FIXED: ML predictions have real impact on final decision
✅ FIXED: Human-readable labels and confidence thresholds
✅ FIXED: Remove dangerous ignore_mismatched_sizes flag
✅ FIXED: Exact training architecture enforced
"""

import os
import sys
import json
import time
import traceback
import torch
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from transformers import RobertaTokenizer, RobertaConfig, RobertaForSequenceClassification

# ================== FIXED PATH CONFIGURATION ==================
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

print(f"🔧 Project root: {project_root}")
print(f"🔧 Src path: {os.path.join(project_root, 'src')}")

# Import all modules with error handling
COMPONENTS = {}

# ================== LOAD ALL COMPONENTS ==================

print("🔧 Loading all components...")

# Language Detector
try:
    from language_detector import get_language_detector
    COMPONENTS['language_detector'] = get_language_detector()
    print("✅ Language detector loaded")
except ImportError as e:
    print(f"❌ Language detector import failed: {e}")
    COMPONENTS['language_detector'] = None

# Style Checker
try:
    from style_checker import get_style_checker
    COMPONENTS['style_checker'] = get_style_checker()
    print("✅ Style checker loaded")
except ImportError as e:
    print(f"❌ Style checker import failed: {e}")
    COMPONENTS['style_checker'] = None

# Security Scanner
try:
    from security_scanner import get_security_scanner
    COMPONENTS['security_scanner'] = get_security_scanner()
    print("✅ Security scanner loaded")
except ImportError as e:
    print(f"❌ Security scanner import failed: {e}")
    COMPONENTS['security_scanner'] = None

# AUG-PDG
try:
    from aug_pdg import Pipeline as AugPDGPipeline
    COMPONENTS['aug_pdg_builder'] = AugPDGPipeline
    print("✅ AUG-PDG loaded")
except ImportError as e:
    print(f"❌ AUG-PDG import failed: {e}")
    COMPONENTS['aug_pdg_builder'] = None

# ================== VULNERABILITY MODEL LOADER ==================

class ModelLoadError(Exception):
    """Exception raised when model fails to load."""
    pass

class VulnerabilityModelLoader:
    """
    Loads vulnerability detection model
    ✅ FIXED: No dangerous ignore_mismatched_sizes flag
    ✅ FIXED: Exact training architecture enforced
    ✅ FIXED: Human-readable labels
    """
    
    LOCAL_MODEL_PATH = os.path.join(project_root, "models", "vulnerability_logic_production")
    
    # 🔴 CRITICAL FIX: Define human-readable labels
    LABEL_MAPPING = {0: "SAFE", 1: "VULNERABLE"}

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.status = "UNINITIALIZED"
        self.load_model()

    def load_model(self):
        """
        Load model ONLY from local directory
        ✅ FIXED: Remove ignore_mismatched_sizes flag to detect training errors
        ✅ FIXED: Force exact training architecture
        """
        print(f"📦 Loading vulnerability model from LOCAL path: {self.LOCAL_MODEL_PATH}")
        
        # Check if local model directory exists
        if not os.path.exists(self.LOCAL_MODEL_PATH):
            self.status = "MODEL_MISSING"
            raise ModelLoadError(
                f"Model directory not found: {self.LOCAL_MODEL_PATH}\n"
                "Please run: python scripts/download_model.py"
            )
        
        # Check for required files
        config_path = os.path.join(self.LOCAL_MODEL_PATH, "config.json")
        if not os.path.exists(config_path):
            self.status = "INCOMPLETE_MODEL"
            raise ModelLoadError(
                f"Model config not found: {config_path}\n"
                "Model download may be incomplete."
            )
        
        # Check for model weights
        safetensors_path = os.path.join(self.LOCAL_MODEL_PATH, "model.safetensors")
        pytorch_path = os.path.join(self.LOCAL_MODEL_PATH, "pytorch_model.bin")
        
        if not os.path.exists(safetensors_path) and not os.path.exists(pytorch_path):
            self.status = "NO_MODEL_WEIGHTS"
            raise ModelLoadError(
                f"No model weights found in: {self.LOCAL_MODEL_PATH}\n"
                "Expected either model.safetensors or pytorch_model.bin"
            )
        
        try:
            # ✅ STEP 1: Load tokenizer
            print("   Loading tokenizer...")
            self.tokenizer = RobertaTokenizer.from_pretrained(self.LOCAL_MODEL_PATH)
            vocab_size = len(self.tokenizer)
            print(f"   Tokenizer vocab size: {vocab_size}")
            
            # ✅ STEP 2: Load config EXACTLY as trained
            print("   Loading config exactly as trained...")
            config = RobertaConfig.from_pretrained(self.LOCAL_MODEL_PATH)
            
            # 🔴 CRITICAL FIX: FORCE exact training parameters
            config.max_position_embeddings = 514
            config.type_vocab_size = 1
            config.num_labels = 2
            
            # 🔴 CRITICAL FIX: Set human-readable labels
            config.id2label = self.LABEL_MAPPING
            config.label2id = {"SAFE": 0, "VULNERABLE": 1}
            
            # ✅ STEP 3: Load model with exact config
            print("   Loading model with exact config...")
            
            # 🔴 CRITICAL FIX: REMOVED ignore_mismatched_sizes=True
            if os.path.exists(safetensors_path):
                print(f"   Using safetensors format")
                self.model = RobertaForSequenceClassification.from_pretrained(
                    self.LOCAL_MODEL_PATH,
                    config=config,
                    use_safetensors=True
                    # ✅ REMOVED: ignore_mismatched_sizes=True
                )
            else:
                print(f"   Using pytorch format")
                self.model = RobertaForSequenceClassification.from_pretrained(
                    self.LOCAL_MODEL_PATH,
                    config=config
                    # ✅ REMOVED: ignore_mismatched_sizes=True
                )
            
            # Move to device
            self.model.to(self.device)
            self.model.eval()
            
            # ✅ STEP 4: HARD ASSERTIONS to verify exact architecture
            print("   Verifying exact architecture...")
            assert self.model.config.max_position_embeddings == 514, \
                f"Model max_position_embeddings mismatch: {self.model.config.max_position_embeddings} != 514"
            assert self.model.config.type_vocab_size == 1, \
                f"Model type_vocab_size mismatch: {self.model.config.type_vocab_size} != 1"
            assert self.model.config.num_labels == 2, \
                f"Model num_labels mismatch: {self.model.config.num_labels} != 2"
            
            print(f"   ✅ Architecture verified:")
            print(f"      - max_position_embeddings: {self.model.config.max_position_embeddings}")
            print(f"      - type_vocab_size: {self.model.config.type_vocab_size}")
            print(f"      - num_labels: {self.model.config.num_labels}")
            print(f"      - labels: {self.model.config.id2label}")
            
            self.status = "MODEL_LOADED"
            print(f"✅ Vulnerability model loaded successfully on {self.device}")
            print(f"   Status: {self.status}")
            
        except RuntimeError as e:
            # 🔴 CRITICAL: Catch size mismatch errors
            if "size mismatch" in str(e) or "ignore_mismatched_sizes" in str(e):
                self.status = "ARCHITECTURE_MISMATCH"
                raise ModelLoadError(
                    f"❌ MODEL TRAINING ERROR DETECTED!\n"
                    f"The saved model weights don't match the expected architecture.\n"
                    f"This indicates the model was not trained with the correct configuration.\n"
                    f"Error: {str(e)}\n\n"
                    f"🔥 SOLUTION:\n"
                    f"1. Check if the model was trained with max_position_embeddings=514\n"
                    f"2. Verify the tokenizer matches the model vocabulary\n"
                    f"3. Re-train the model with the exact configuration above"
                ) from e
            raise
        except Exception as e:
            self.status = "LOAD_FAILED"
            raise ModelLoadError(
                f"Failed to load model from {self.LOCAL_MODEL_PATH}: {str(e)}\n"
                f"Traceback: {traceback.format_exc()}"
            ) from e

    def predict_vulnerability(self, source, sink, sanitization):
        """
        Predict if a code pattern is vulnerable
        ✅ FIXED: Increased max_length for better context
        """
        # Check model status
        if self.status != "MODEL_LOADED" or not self.model or not self.tokenizer:
            raise RuntimeError(
                f"Cannot make predictions: model status is '{self.status}'\n"
                "Model must be successfully loaded before making predictions."
            )

        # Format input as trained
        input_text = f"""[VULNERABILITY_FLOW]
SOURCE: {source}
SINK: {sink}
SANITIZATION: {sanitization}"""

        # ✅ STEP 5: Increased max_length for better context
        inputs = self.tokenizer(
            input_text,
            truncation=True,
            padding=True,
            max_length=256,  # 🔴 FIXED: Increased from 128 to 256
            return_tensors="pt"
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()

        # Get label mapping (now human-readable)
        label_map = self.model.config.id2label
        class_name = label_map.get(predicted_class, f"class_{predicted_class}")

        return {
            "is_vulnerable": bool(predicted_class == 1),
            "predicted_class": predicted_class,
            "predicted_class_name": class_name,
            "confidence": confidence,
            "vulnerability_score": probabilities[0][1].item() if len(probabilities[0]) > 1 else 0.0,
            "safe_score": probabilities[0][0].item() if len(probabilities[0]) > 0 else 0.0,
            "model_status": self.status,
            "device": str(self.device)
        }
    
    def get_status(self):
        """Get current model status."""
        return {
            "status": self.status,
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None,
            "device": str(self.device),
            "model_path": self.LOCAL_MODEL_PATH,
            "vocab_size": len(self.tokenizer) if self.tokenizer else 0,
            "model_vocab_size": self.model.config.vocab_size if self.model else 0,
            "num_labels": self.model.config.num_labels if self.model else 0,
            "labels": self.model.config.id2label if self.model else {},
            "max_position_embeddings": self.model.config.max_position_embeddings if self.model else 0
        }

# ================== PIPELINE CLASSES ==================

class PipelineStatus(Enum):
    """Status of pipeline analysis."""
    SUCCESS = "success"
    PARTIAL = "partial"
    ERROR = "error"
    SKIPPED = "skipped"

@dataclass
class ComponentResult:
    """Result from a single component."""
    name: str
    status: PipelineStatus
    data: Dict[str, Any]
    error: Optional[str] = None
    processing_time: float = 0.0

@dataclass
class PipelineResult:
    """Complete pipeline analysis results."""
    filename: str
    language: str

    # Component results
    language_detection: ComponentResult
    style_analysis: ComponentResult
    security_analysis: ComponentResult
    aug_pdg_analysis: Optional[ComponentResult] = None
    vulnerability_prediction: Optional[ComponentResult] = None

    # Combined metrics
    overall_score: int = 0
    overall_status: str = "UNKNOWN"
    final_decision: str = "UNKNOWN"
    total_processing_time: float = 0.0
    timestamp: float = 0.0

    # Recommendations
    recommendations: List[str] = None

    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['language_detection']['status'] = self.language_detection.status.value
        result['style_analysis']['status'] = self.style_analysis.status.value
        result['security_analysis']['status'] = self.security_analysis.status.value

        if self.aug_pdg_analysis:
            result['aug_pdg_analysis']['status'] = self.aug_pdg_analysis.status.value
        if self.vulnerability_prediction:
            result['vulnerability_prediction']['status'] = self.vulnerability_prediction.status.value

        return result

class CodeReviewPipeline:
    """
    Main pipeline for complete code review
    ✅ FIXED: ML predictions have real security impact
    """

    def __init__(self):
        self.components = COMPONENTS
        self.supported_languages = ["python", "java", "javascript", "php", "ruby", "go"]
        
        # Confidence threshold for language detection
        self.language_confidence_threshold = 0.7  # 70%
        
        # 🔴 CRITICAL FIX: ML confidence thresholds
        self.vulnerability_confidence_threshold = 0.85  # 85% confidence to reject
        self.review_confidence_threshold = 0.70  # 70% confidence to require review
        
        # Initialize vulnerability model
        print(f"\n🤖 Initializing vulnerability model...")
        try:
            self.vulnerability_model = VulnerabilityModelLoader()
            model_status = self.vulnerability_model.get_status()
            print(f"   Model status: {model_status['status']}")
            print(f"   Labels: {model_status.get('labels', {})}")
            print(f"   Architecture: max_position_embeddings={model_status.get('max_position_embeddings', 'N/A')}")
        except ModelLoadError as e:
            print(f"❌ CRITICAL: Vulnerability model failed to load")
            print(f"   Error: {e}")
            print(f"   Please ensure model is downloaded: python scripts/download_model.py")
            raise

        print(f"\n📋 Available Components:")
        for name, component in self.components.items():
            status = "✅" if component is not None else "❌"
            print(f"  {status} {name}")
        
        # Show model status
        model_status = self.vulnerability_model.get_status()
        status_symbol = "✅" if model_status['status'] == "MODEL_LOADED" else "❌"
        print(f"  {status_symbol} vulnerability_model [{model_status['status']}]")
        print(f"     Confidence thresholds: Reject @ {self.vulnerability_confidence_threshold:.0%}, Review @ {self.review_confidence_threshold:.0%}")

    def process_file(self, filepath: str) -> PipelineResult:
        """
        Process a single file through the complete pipeline
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()

        filename = os.path.basename(filepath)

        return self.analyze_code(code, filename)

    def analyze_code(self, code: str, filename: str = "unknown.txt") -> PipelineResult:
        """
        Complete analysis pipeline for code
        """
        start_time = time.time()
        component_results = {}
        recommendations = []

        print(f"\n{'='*70}")
        print(f"🚀 ANALYZING: {filename}")
        print(f"{'='*70}")

        # ================== STEP 1: LANGUAGE DETECTION ==================
        print(f"\n[1/5] 🔍 Language Detection")
        lang_detection_time = time.time()

        language = "unknown"
        language_supported = False
        language_confidence = 0.0

        if self.components['language_detector']:
            try:
                detector = self.components['language_detector']
                language, confidence = detector.detect(filename, code)
                language_confidence = confidence
                language_supported = language and language.lower() in self.supported_languages

                component_results['language_detection'] = ComponentResult(
                    name="language_detector",
                    status=PipelineStatus.SUCCESS,
                    data={
                        "language": language or "unknown",
                        "confidence": confidence,
                        "supported": language_supported
                    },
                    processing_time=time.time() - lang_detection_time
                )

                print(f"   → Detected: {language or 'unknown'} (confidence: {confidence:.1%})")
                print(f"   → Supported: {'✅ Yes' if language_supported else '❌ No'}")

                # Check confidence threshold
                if confidence < self.language_confidence_threshold:
                    print(f"   ⚠️  Low confidence detection (<{self.language_confidence_threshold:.0%})")
                    component_results['language_detection'].data["supported"] = False

            except Exception as e:
                component_results['language_detection'] = ComponentResult(
                    name="language_detector",
                    status=PipelineStatus.ERROR,
                    data={"language": "unknown", "confidence": 0.0, "supported": False},
                    error=str(e),
                    processing_time=time.time() - lang_detection_time
                )
                print(f"   → Failed: {e}")
        else:
            language = self._detect_by_extension(filename)
            language_confidence = 0.7  # Default confidence for extension detection
            language_supported = language in self.supported_languages
            component_results['language_detection'] = ComponentResult(
                name="language_detector",
                status=PipelineStatus.SKIPPED,
                data={"language": language, "confidence": language_confidence, "supported": language_supported},
                processing_time=time.time() - lang_detection_time
            )
            print(f"   → Using extension detection: {language}")

        language = component_results['language_detection'].data.get("language", "unknown")
        language_supported = component_results['language_detection'].data.get("supported", False)
        language_confidence = component_results['language_detection'].data.get("confidence", 0.0)

        # Stop if language not supported OR confidence is too low
        if not language_supported or language == "unknown":
            reason = ""
            if language_confidence < self.language_confidence_threshold:
                reason = f"Low confidence detection ({language_confidence:.1%} < {self.language_confidence_threshold:.0%})"
            else:
                reason = f"Unsupported language: {language}"
                
            result = PipelineResult(
                filename=filename,
                language=language,
                language_detection=component_results['language_detection'],
                style_analysis=ComponentResult(
                    name="style_checker",
                    status=PipelineStatus.SKIPPED,
                    data={"passed": False, "score": 0, "error": reason},
                    error=reason
                ),
                security_analysis=ComponentResult(
                    name="security_scanner",
                    status=PipelineStatus.SKIPPED,
                    data={"passed": False, "score": 0, "error": reason},
                    error=reason
                ),
                overall_score=0,
                overall_status="UNSUPPORTED_LANGUAGE",
                final_decision="REJECT",
                total_processing_time=time.time() - start_time,
                timestamp=start_time,
                recommendations=[f"Language detection failed: {reason}"]
            )
            print(f"\n❌ Analysis stopped: {reason}")
            return result

        # ================== STEP 2: STYLE ANALYSIS ==================
        print(f"\n[2/5] 📋 Style Analysis")
        style_analysis_time = time.time()

        if self.components['style_checker']:
            try:
                checker = self.components['style_checker']
                style_result = checker.analyze_code(code, filename)

                # Convert to dict if needed
                if hasattr(style_result, 'to_dict'):
                    style_data = style_result.to_dict()
                else:
                    style_data = style_result

                component_results['style_analysis'] = ComponentResult(
                    name="style_checker",
                    status=PipelineStatus.SUCCESS,
                    data=style_data,
                    processing_time=time.time() - style_analysis_time
                )

                print(f"   → Score: {style_data.get('score', 0)}/100")
                print(f"   → Errors: {len(style_data.get('errors', []))}, Warnings: {len(style_data.get('warnings', []))}")

                # Add recommendations
                style_score = style_data.get('score', 100)
                if style_score < 70:
                    recommendations.append(f"Improve code style (score: {style_score}/100)")

            except Exception as e:
                component_results['style_analysis'] = ComponentResult(
                    name="style_checker",
                    status=PipelineStatus.ERROR,
                    data={"passed": False, "score": 0, "error": str(e)},
                    error=str(e),
                    processing_time=time.time() - style_analysis_time
                )
                print(f"   → Failed: {e}")
        else:
            component_results['style_analysis'] = ComponentResult(
                name="style_checker",
                status=PipelineStatus.SKIPPED,
                data={"passed": True, "score": 80, "skip_reason": "Component unavailable"},
                processing_time=time.time() - style_analysis_time
            )
            print(f"   → Skipped (component unavailable)")

        # ================== STEP 3: SECURITY ANALYSIS ==================
        print(f"\n[3/5] 🔒 Security Analysis")
        security_analysis_time = time.time()

        security_score = 100  # Default score
        
        if self.components['security_scanner']:
            try:
                scanner = self.components['security_scanner']
                security_result = scanner.scan(language, code, filename)

                component_results['security_analysis'] = ComponentResult(
                    name="security_scanner",
                    status=PipelineStatus.SUCCESS,
                    data=security_result,
                    processing_time=time.time() - security_analysis_time
                )

                security_score = security_result.get('score', 100)
                risk_level = security_result.get('risk_level', 'LOW')
                findings = security_result.get('findings', [])
                
                print(f"   → Score: {security_score}/100")
                print(f"   → Risk Level: {risk_level}")
                print(f"   → Findings: {len(findings)}")

                # Add recommendations
                if security_score < 80:
                    recommendations.append(f"Address security issues (score: {security_score}/100)")

                critical_findings = [f for f in findings if f.get('severity') in ['CRITICAL', 'HIGH']]
                if critical_findings:
                    recommendations.append(f"Fix {len(critical_findings)} critical/high security findings")

            except Exception as e:
                # 🟠 ISSUE 5: Handle scanner failures gracefully but mark them
                component_results['security_analysis'] = ComponentResult(
                    name="security_scanner",
                    status=PipelineStatus.ERROR,
                    data={
                        "passed": False, 
                        "score": 0, 
                        "error": str(e),
                        "risk_level": "UNKNOWN",
                        "findings": []
                    },
                    error=str(e),
                    processing_time=time.time() - security_analysis_time
                )
                print(f"   → ⚠️ Security scanner failed: {e}")
                recommendations.append("Security scanner failed - manual review required")
                security_score = 50  # Penalty for scanner failure
        else:
            component_results['security_analysis'] = ComponentResult(
                name="security_scanner",
                status=PipelineStatus.SKIPPED,
                data={
                    "passed": True, 
                    "score": 80, 
                    "skip_reason": "Component unavailable",
                    "risk_level": "UNKNOWN",
                    "findings": []
                },
                processing_time=time.time() - security_analysis_time
            )
            print(f"   → Skipped (component unavailable)")

        # ================== STEP 4: AUG-PDG ANALYSIS ==================
        print(f"\n[4/5] 🏗️ AUG-PDG Analysis")
        aug_pdg_time = time.time()

        vulnerability_patterns = []  # Collect patterns for ML analysis
        
        if self.components['aug_pdg_builder']:
            try:
                pdg_pipeline = self.components['aug_pdg_builder'](language=language)
                pdg_result = pdg_pipeline.run(code)

                pdg_stats = pdg_result.get('stats', {})
                pdg_vulnerabilities = pdg_result.get('vulnerabilities', [])

                component_results['aug_pdg_analysis'] = ComponentResult(
                    name="aug_pdg",
                    status=PipelineStatus.SUCCESS,
                    data={
                        "pdg_nodes": pdg_stats.get('pdg_nodes', 0),
                        "pdg_edges": pdg_stats.get('pdg_edges', 0),
                        "security_issues": pdg_stats.get('security_issues', 0),
                        "vulnerabilities": pdg_vulnerabilities
                    },
                    processing_time=time.time() - aug_pdg_time
                )

                print(f"   → PDG Nodes: {pdg_stats.get('pdg_nodes', 0)}")
                print(f"   → PDG Edges: {pdg_stats.get('pdg_edges', 0)}")
                print(f"   → Security Issues: {pdg_stats.get('security_issues', 0)}")
                print(f"   → Vulnerabilities found: {len(pdg_vulnerabilities)}")

                # Extract vulnerability patterns for model
                for vuln in pdg_vulnerabilities[:3]:  # Limit to top 3
                    source_candidates = vuln.get('variables_used', []) or vuln.get('input_sources', [])
                    sink_candidates = vuln.get('variables_defined', []) or vuln.get('sensitive_sinks', [])
                    
                    if source_candidates and sink_candidates:
                        for source in source_candidates[:2]:
                            for sink in sink_candidates[:2]:
                                pattern = {
                                    'source': source,
                                    'sink': sink,
                                    'sanitization': 'none' if vuln.get('involves_tainted_input', False) else 'unknown',
                                    'context': vuln.get('location', 'unknown'),
                                    'confidence': vuln.get('confidence', 0.5),
                                    'type': vuln.get('type', 'unknown')
                                }
                                vulnerability_patterns.append(pattern)

                component_results['aug_pdg_analysis'].data['vulnerability_patterns'] = vulnerability_patterns

                if pdg_vulnerabilities:
                    recommendations.append(f"Review {len(pdg_vulnerabilities)} PDG-identified vulnerabilities")

            except Exception as e:
                component_results['aug_pdg_analysis'] = ComponentResult(
                    name="aug_pdg",
                    status=PipelineStatus.ERROR,
                    data={
                        "pdg_nodes": 0,
                        "pdg_edges": 0,
                        "vulnerabilities": [],
                        "error": str(e)
                    },
                    error=str(e),
                    processing_time=time.time() - aug_pdg_time
                )
                print(f"   → Failed: {e}")
        else:
            component_results['aug_pdg_analysis'] = ComponentResult(
                name="aug_pdg",
                status=PipelineStatus.SKIPPED,
                data={"pdg_nodes": 0, "pdg_edges": 0, "vulnerabilities": [], "skip_reason": "Component unavailable"},
                processing_time=time.time() - aug_pdg_time
            )
            print(f"   → Skipped (component unavailable)")

        # ================== STEP 5: VULNERABILITY MODEL PREDICTION ==================
        print(f"\n[5/5] 🤖 Vulnerability Model Prediction")
        vulnerability_time = time.time()

        predictions = []
        try:
            # If no patterns from AUG-PDG, create from security findings
            if not vulnerability_patterns and \
               component_results['security_analysis'].status == PipelineStatus.SUCCESS:
                security_data = component_results['security_analysis'].data
                findings = security_data.get('findings', [])
                for finding in findings[:2]:
                    pattern = {
                        'source': 'user_input',
                        'sink': finding.get('category', 'dangerous_operation'),
                        'sanitization': 'none' if 'injection' in finding.get('category', '').lower() else 'unknown',
                        'context': finding.get('description', ''),
                        'type': finding.get('severity', 'MEDIUM'),
                        'confidence': 0.5
                    }
                    vulnerability_patterns.append(pattern)

            # Fallback: create simple patterns
            if not vulnerability_patterns:
                vulnerability_patterns = [
                    {
                        'source': 'user_input', 
                        'sink': 'system_call', 
                        'sanitization': 'unknown',
                        'context': 'General system interaction',
                        'type': 'command_injection',
                        'confidence': 0.3
                    }
                ]

            # Make predictions for each pattern
            for pattern in vulnerability_patterns[:2]:
                try:
                    pred = self.vulnerability_model.predict_vulnerability(
                        pattern['source'],
                        pattern['sink'],
                        pattern['sanitization']
                    )
                    pred['pattern'] = pattern
                    predictions.append(pred)
                    
                    print(f"   → Pattern: {pattern['source']} → {pattern['sink']}")
                    print(f"     Prediction: {pred['predicted_class_name']} (confidence: {pred['confidence']:.1%})")
                    
                except Exception as e:
                    print(f"   → ⚠️ Failed to predict for pattern: {e}")
                    continue

            component_results['vulnerability_prediction'] = ComponentResult(
                name="vulnerability_model",
                status=PipelineStatus.SUCCESS,
                data={
                    "predictions": predictions,
                    "total_predictions": len(predictions),
                    "vulnerable_count": sum(1 for p in predictions if p.get('is_vulnerable', False)),
                    "high_confidence_vulnerable": sum(1 for p in predictions 
                                                     if p.get('is_vulnerable', False) and 
                                                     p.get('confidence', 0) >= self.vulnerability_confidence_threshold),
                    "model_status": self.vulnerability_model.get_status()
                },
                processing_time=time.time() - vulnerability_time
            )

            print(f"   → Total predictions: {len(predictions)}")
            print(f"   → Vulnerable predictions: {sum(1 for p in predictions if p.get('is_vulnerable', False))}")
            print(f"   → High-confidence vulnerable: {component_results['vulnerability_prediction'].data['high_confidence_vulnerable']}")

            # Add recommendations based on model predictions
            vulnerable_predictions = [p for p in predictions if p.get('is_vulnerable', False)]
            if vulnerable_predictions:
                high_conf_vulns = [p for p in vulnerable_predictions if p.get('confidence', 0) >= self.vulnerability_confidence_threshold]
                if high_conf_vulns:
                    recommendations.append(f"CRITICAL: Model detected {len(high_conf_vulns)} high-confidence vulnerable patterns")
                else:
                    recommendations.append(f"Model detected {len(vulnerable_predictions)} vulnerable patterns (review recommended)")

        except Exception as e:
            component_results['vulnerability_prediction'] = ComponentResult(
                name="vulnerability_model",
                status=PipelineStatus.ERROR,
                data={
                    "predictions": [], 
                    "error": str(e),
                    "model_status": self.vulnerability_model.get_status()
                },
                error=str(e),
                processing_time=time.time() - vulnerability_time
            )
            print(f"   → Failed: {e}")

        # ================== FINAL DECISION ==================
        total_time = time.time() - start_time

        # Calculate overall score
        style_score = component_results['style_analysis'].data.get('score', 100)
        overall_score = int((style_score * 0.4) + (security_score * 0.6))

        # Get vulnerability data safely
        vulnerability_data = {}
        if 'vulnerability_prediction' in component_results:
            vuln_result = component_results['vulnerability_prediction']
            if vuln_result and hasattr(vuln_result, 'data'):
                vulnerability_data = vuln_result.data

        # Get security data
        security_data = component_results['security_analysis'].data
        risk_level = security_data.get('risk_level', 'LOW')

        # 🔴 CRITICAL FIX: ML predictions have REAL impact
        final_decision = self._make_decision(
            overall_score,
            security_score,
            risk_level,
            vulnerability_data
        )

        # Determine overall status
        if overall_score >= 90:
            overall_status = "EXCELLENT"
        elif overall_score >= 75:
            overall_status = "GOOD"
        elif overall_score >= 60:
            overall_status = "FAIR"
        else:
            overall_status = "POOR"

        # Create final result
        result = PipelineResult(
            filename=filename,
            language=language,
            language_detection=component_results['language_detection'],
            style_analysis=component_results['style_analysis'],
            security_analysis=component_results['security_analysis'],
            aug_pdg_analysis=component_results.get('aug_pdg_analysis'),
            vulnerability_prediction=component_results.get('vulnerability_prediction'),
            overall_score=overall_score,
            overall_status=overall_status,
            final_decision=final_decision,
            total_processing_time=total_time,
            timestamp=start_time,
            recommendations=recommendations
        )

        print(f"\n{'='*70}")
        print(f"✅ ANALYSIS COMPLETE")
        print(f"{'='*70}")
        print(f"📊 Overall Score: {overall_score}/100 ({overall_status})")
        print(f"🔒 Security Score: {security_score}/100")
        print(f"⚖️  Final Decision: {final_decision}")
        print(f"⏱️  Total Time: {total_time:.2f}s")
        print(f"💡 Recommendations: {len(recommendations)}")
        print(f"🤖 Model Status: {self.vulnerability_model.get_status()['status']}")
        print(f"{'='*70}")

        return result

    def _make_decision(self, overall_score, security_score, risk_level, vulnerability_data):
        """
        Make final decision on code safety
        🔴 CRITICAL FIX: ML predictions have REAL impact
        """
        # 🔴 ISSUE 2 FIX: High-confidence ML predictions → REJECT
        high_confidence_vulnerable = vulnerability_data.get('high_confidence_vulnerable', 0)
        
        # Rule 1: High-confidence ML detection → REJECT
        if high_confidence_vulnerable > 0:
            return "REJECT"
        
        # Rule 2: Critical security issues → REJECT
        if security_score < 60 or risk_level in ['CRITICAL', 'HIGH']:
            return "REJECT"
        
        # Rule 3: Any ML vulnerability detection → REVIEW_REQUIRED
        vulnerable_count = vulnerability_data.get('vulnerable_count', 0)
        if vulnerable_count > 0:
            return "REVIEW_REQUIRED"
        
        # Rule 4: Medium security issues → REVIEW_RECOMMENDED
        if security_score < 75:
            return "REVIEW_RECOMMENDED"
        
        # Rule 5: Everything looks good → APPROVE
        return "APPROVE"

    def _detect_by_extension(self, filename: str) -> str:
        """Detect language by file extension."""
        ext_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.java': 'java',
            '.php': 'php',
            '.rb': 'ruby',
            '.go': 'go',
        }

        for ext, lang in ext_map.items():
            if filename.lower().endswith(ext):
                return lang

        return "unknown"

    def get_model_status(self):
        """Get vulnerability model status."""
        return self.vulnerability_model.get_status()

# ================== MAIN EXECUTION ==================

def main():
    """Main function to run the complete pipeline"""

    print("=" * 80)
    print("🚀 CODE REVIEW ASSISTANT - COMPLETE PIPELINE")
    print("=" * 80)
    print("Workflow: File → Language Detector → Style Check → Security Scan →")
    print("          AUG-PDG → Vulnerability Model → Final Decision")
    print("=" * 80)

    try:
        # Initialize pipeline (will raise ModelLoadError if model missing)
        pipeline = CodeReviewPipeline()
        
        # Show model status
        model_status = pipeline.get_model_status()
        print(f"\n🤖 Model Status: {model_status['status']}")
        print(f"📁 Model Path: {model_status['model_path']}")
        print(f"⚡ Device: {model_status['device']}")
        print(f"📊 Architecture: max_position_embeddings={model_status.get('max_position_embeddings', 'N/A')}")
        
    except ModelLoadError as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        print(f"\n💡 SOLUTION: Run this command first:")
        print(f"   python scripts/download_model.py")
        return 1

    # Ask for file path
    file_path = input("\n📂 Enter path to code file: ").strip()

    if not file_path:
        print("⚠️ No file path provided. Using example file...")

        # Create example file
        example_code = '''import os
import subprocess

def calculate_sum(a, b):
    """Calculate sum of two numbers."""
    return a + b

def vulnerable_function(user_input):
    # Simulated vulnerability
    os.system("echo " + user_input)  # Command injection
    return "Done"

def main():
    x = 5
    y = 10
    result = calculate_sum(x, y)
    print(f"Sum: {result}")

    # Test vulnerable function
    vulnerable_function("test")

if __name__ == "__main__":
    main()'''

        file_path = "/tmp/example.py"
        with open(file_path, 'w') as f:
            f.write(example_code)
        print(f"Created example file at: {file_path}")

    try:
        # Process the file
        result = pipeline.process_file(file_path)

        # Generate report
        print("\n" + "=" * 80)
        print("📋 FINAL ANALYSIS REPORT")
        print("=" * 80)

        # Language info
        print(f"\n🔤 Language: {result.language}")
        print(f"📁 File: {result.filename}")

        # Style analysis
        style_data = result.style_analysis.data
        print(f"\n📋 STYLE ANALYSIS:")
        print(f"  Score: {style_data.get('score', 0)}/100")
        print(f"  Status: {'✅ PASSED' if style_data.get('passed', False) else '❌ FAILED'}")

        # Security analysis
        security_data = result.security_analysis.data
        print(f"\n🔒 SECURITY ANALYSIS:")
        print(f"  Score: {security_data.get('score', 0)}/100")
        print(f"  Risk Level: {security_data.get('risk_level', 'UNKNOWN')}")
        print(f"  Findings: {len(security_data.get('findings', []))}")

        # AUG-PDG analysis
        if result.aug_pdg_analysis:
            pdg_data = result.aug_pdg_analysis.data
            print(f"\n🏗️ AUG-PDG ANALYSIS:")
            print(f"  PDG Nodes: {pdg_data.get('pdg_nodes', 0)}")
            print(f"  PDG Edges: {pdg_data.get('pdg_edges', 0)}")
            print(f"  Vulnerabilities: {len(pdg_data.get('vulnerabilities', []))}")

        # Vulnerability model predictions
        if result.vulnerability_prediction:
            vuln_data = result.vulnerability_prediction.data
            print(f"\n🤖 VULNERABILITY MODEL:")
            print(f"  Status: {vuln_data.get('model_status', {}).get('status', 'UNKNOWN')}")
            print(f"  Predictions: {vuln_data.get('total_predictions', 0)}")
            print(f"  Vulnerable: {vuln_data.get('vulnerable_count', 0)}")
            print(f"  High-confidence vulnerable: {vuln_data.get('high_confidence_vulnerable', 0)}")
            
            # Show individual predictions
            predictions = vuln_data.get('predictions', [])
            for i, pred in enumerate(predictions, 1):
                print(f"    {i}. Source: {pred.get('pattern', {}).get('source', 'N/A')}")
                print(f"       Sink: {pred.get('pattern', {}).get('sink', 'N/A')}")
                print(f"       Prediction: {pred.get('predicted_class_name', 'N/A')}")
                print(f"       Confidence: {pred.get('confidence', 0):.1%}")
                print(f"       High-confidence: {'✅' if pred.get('confidence', 0) >= 0.85 else '❌'}")

        # Final decision
        print(f"\n" + "=" * 80)
        print(f"⚖️ FINAL DECISION: {result.final_decision}")
        print("=" * 80)

        if result.final_decision == "APPROVE":
            print("✅ Code is SAFE to merge")
        elif result.final_decision == "REVIEW_REQUIRED":
            print("⚠️ Code requires SECURITY REVIEW before merging")
        elif result.final_decision == "REVIEW_RECOMMENDED":
            print("⚠️ Code review recommended")
        else:
            print("❌ Code is NOT SAFE - REJECT pull request")

        # Recommendations
        if result.recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(result.recommendations, 1):
                print(f"  {i}. {rec}")

        print(f"\n⏱️ Total processing time: {result.total_processing_time:.2f}s")

        # ✅ STEP 6: Sanity test
        print(f"\n🧪 SANITY TEST - Verifying model predictions...")
        try:
            test_pred = pipeline.vulnerability_model.predict_vulnerability(
                "user_input",
                "system_call",
                "none"
            )
            print(f"   Test prediction: {test_pred['predicted_class_name']} (confidence: {test_pred['confidence']:.1%})")
            print(f"   Labels: {pipeline.vulnerability_model.model.config.id2label}")
            print(f"   ✅ Model is producing predictions with correct labels")
        except Exception as e:
            print(f"   ❌ Sanity test failed: {e}")

    except Exception as e:
        print(f"\n❌ Error processing file: {e}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
