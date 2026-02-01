"""
Layer 3: LLM Explainer

Generates natural language explanations for stock analysis using local LLMs.
Supports:
- Qwen2.5-3B: Good balance of quality and resource usage
- Qwen2.5-7B: Better explanations, needs more memory
- Phi-3-mini: Microsoft's efficient small model

Uses llama-cpp-python for efficient CPU/GPU inference with GGUF models.
"""

import json
import logging
import time
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .base import (
    ExplainerOutput,
    MLLayerBase,
    ModelConfig,
    ModelStatus,
    ModelType,
)

logger = logging.getLogger(__name__)


@dataclass
class AnalysisContext:
    """Context for generating explanations."""
    symbol: str
    company_name: str
    signal: str  # BUY/HOLD/AVOID/SELL
    confidence: float
    
    # Scores from analysis engines
    governance_score: float
    financial_score: float
    valuation_score: float
    market_score: float
    composite_score: float
    
    # Key metrics
    metrics: Dict[str, Any]
    
    # Forecast data (optional)
    forecast_trend: str = "neutral"
    forecast_strength: float = 0.0
    price_prediction_30d: Optional[float] = None
    
    # Red flags
    red_flags: List[str] = None
    warnings: List[str] = None
    
    # Signal probabilities from classifier
    signal_probabilities: Optional[Dict[str, float]] = None
    
    def to_prompt_context(self) -> str:
        """Convert to a context string for the LLM prompt."""
        lines = [
            f"## Stock Analysis: {self.symbol} ({self.company_name})",
            "",
            f"**Recommendation:** {self.signal} (Confidence: {self.confidence:.0%})",
            f"**Composite Score:** {self.composite_score:.1f}/100",
            "",
            "### Dimension Scores:",
            f"- Governance & Legacy: {self.governance_score:.1f}/100",
            f"- Financial Trajectory: {self.financial_score:.1f}/100",
            f"- Valuation Context: {self.valuation_score:.1f}/100",
            f"- Market Behaviour: {self.market_score:.1f}/100",
            "",
        ]
        
        if self.forecast_trend != "neutral":
            lines.extend([
                "### Price Forecast:",
                f"- Trend: {self.forecast_trend.capitalize()} (Strength: {self.forecast_strength:.0%})",
            ])
            if self.price_prediction_30d:
                lines.append(f"- 30-Day Price Target: ₹{self.price_prediction_30d:.2f}")
            lines.append("")
        
        if self.metrics:
            lines.append("### Key Metrics:")
            for key, value in self.metrics.items():
                if isinstance(value, float):
                    lines.append(f"- {key}: {value:.2f}")
                else:
                    lines.append(f"- {key}: {value}")
            lines.append("")
        
        if self.red_flags:
            lines.append("### Red Flags:")
            for flag in self.red_flags:
                lines.append(f"- ⚠️ {flag}")
            lines.append("")
        
        if self.warnings:
            lines.append("### Warnings:")
            for warning in self.warnings:
                lines.append(f"- {warning}")
            lines.append("")
        
        if self.signal_probabilities:
            lines.append("### Signal Probabilities:")
            for sig, prob in sorted(self.signal_probabilities.items(), 
                                   key=lambda x: x[1], reverse=True):
                lines.append(f"- {sig}: {prob:.0%}")
        
        return "\n".join(lines)


class AnalysisExplainer(MLLayerBase):
    """Base class for LLM-based analysis explainers."""
    
    @property
    def layer_name(self) -> str:
        return "explainer"
    
    @abstractmethod
    def explain(
        self,
        context: AnalysisContext,
        explanation_type: str = "full"  # "full", "summary", "risks", "opportunities"
    ) -> ExplainerOutput:
        """
        Generate explanation for stock analysis.
        
        Args:
            context: Analysis context with all scores and metrics
            explanation_type: Type of explanation to generate
        """
        pass
    
    def predict(self, *args, **kwargs) -> ExplainerOutput:
        """Wrapper for explain method."""
        return self.explain(*args, **kwargs)
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for the LLM."""
        return """You are an expert Indian equity analyst assistant. Your role is to explain stock analysis results in clear, actionable language.

Key guidelines:
1. Be objective and balanced - present both opportunities and risks
2. Use simple language that retail investors can understand
3. Reference specific metrics and scores to support your points
4. Focus on the "why" behind the recommendation
5. Avoid generic statements - be specific to this stock
6. Consider the Indian market context (NSE/BSE, sector dynamics)
7. Keep explanations concise but comprehensive

Format your response as:
1. Executive Summary (2-3 sentences)
2. Key Strengths (bullet points)
3. Key Risks (bullet points)  
4. Recommendation Rationale (1 paragraph)"""
    
    def _build_user_prompt(self, context: AnalysisContext, explanation_type: str) -> str:
        """Build user prompt for the LLM."""
        context_str = context.to_prompt_context()
        
        if explanation_type == "summary":
            instruction = "Provide a brief 2-3 sentence summary of this analysis."
        elif explanation_type == "risks":
            instruction = "Focus only on the risks and concerns for this stock."
        elif explanation_type == "opportunities":
            instruction = "Focus only on the opportunities and positive aspects."
        else:  # full
            instruction = "Provide a comprehensive explanation of this analysis."
        
        return f"""{context_str}

---
Task: {instruction}
Explain why the recommendation is {context.signal} with {context.confidence:.0%} confidence."""


class QwenExplainer(AnalysisExplainer):
    """
    Qwen2.5-based explainer using llama-cpp-python.
    
    Uses GGUF format for efficient inference on consumer hardware.
    Supports both CPU and GPU (CUDA/Metal) acceleration.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._llm = None
    
    def load_model(self) -> bool:
        """Load Qwen model using llama-cpp-python."""
        if self._llm is not None:
            return True
        
        try:
            self._status = ModelStatus.LOADING
            
            # Import llama-cpp-python
            try:
                from llama_cpp import Llama
            except ImportError:
                self._logger.error(
                    "llama-cpp-python not installed. "
                    "Run: pip install llama-cpp-python"
                )
                self._status = ModelStatus.ERROR
                return False
            
            # Find GGUF file
            model_path = self.config.model_path
            gguf_files = list(model_path.glob("*.gguf"))
            
            if not gguf_files:
                self._logger.error(f"No GGUF files found in {model_path}")
                self._status = ModelStatus.ERROR
                return False
            
            # Prefer Q4_K_M for balance of quality and speed
            gguf_file = None
            for preferred in ["q4_k_m", "q4_0", "q8_0", "f16"]:
                for f in gguf_files:
                    if preferred in f.name.lower():
                        gguf_file = f
                        break
                if gguf_file:
                    break
            
            if not gguf_file:
                gguf_file = gguf_files[0]  # Use first available
            
            self._logger.info(f"Loading model from {gguf_file}")
            
            # Determine GPU layers
            n_gpu_layers = 0
            if self.config.device == "cuda":
                n_gpu_layers = -1  # All layers on GPU
            elif self.config.device == "mps":
                n_gpu_layers = -1  # All layers on Metal
            
            # Load model
            self._llm = Llama(
                model_path=str(gguf_file),
                n_ctx=self.config.context_length,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
            )
            
            self._model = self._llm
            self._status = ModelStatus.READY
            self._logger.info(f"Qwen model loaded successfully on {self.config.device}")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to load Qwen model: {e}")
            self._status = ModelStatus.ERROR
            return False
    
    def unload_model(self) -> None:
        """Unload model from memory."""
        if self._llm is not None:
            del self._llm
            self._llm = None
        if self._model is not None:
            del self._model
            self._model = None
        
        import gc
        gc.collect()
        
        self._status = ModelStatus.DOWNLOADED
    
    def explain(
        self,
        context: AnalysisContext,
        explanation_type: str = "full"
    ) -> ExplainerOutput:
        """Generate explanation using Qwen model."""
        start_time = time.time()
        
        if not self.is_ready:
            if not self.load_model():
                return ExplainerOutput(
                    success=False,
                    layer_name=self.layer_name,
                    model_used=self.config.model_type.value,
                    inference_time_ms=0,
                    device_used=self.config.device,
                    error="Failed to load model"
                )
        
        try:
            # Build prompts
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(context, explanation_type)
            
            # Format for Qwen chat template
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Generate response
            response = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=1024,
                temperature=0.7,
                top_p=0.9,
                stop=["</s>", "<|im_end|>", "<|endoftext|>"]
            )
            
            # Extract generated text
            generated_text = response['choices'][0]['message']['content'].strip()
            
            # Parse the response
            parsed = self._parse_response(generated_text, context)
            
            inference_time = (time.time() - start_time) * 1000
            
            return ExplainerOutput(
                success=True,
                layer_name=self.layer_name,
                model_used=self.config.model_type.value,
                inference_time_ms=inference_time,
                device_used=self.config.device,
                summary=parsed["summary"],
                detailed_analysis=generated_text,
                key_points=parsed["key_points"],
                risk_factors=parsed["risks"],
                opportunities=parsed["opportunities"],
                recommendation_rationale=parsed["rationale"],
                raw_output=response,
                metadata={
                    "tokens_generated": response['usage']['completion_tokens'],
                    "total_tokens": response['usage']['total_tokens'],
                    "explanation_type": explanation_type,
                }
            )
            
        except Exception as e:
            self._logger.error(f"Explanation generation failed: {e}")
            return ExplainerOutput(
                success=False,
                layer_name=self.layer_name,
                model_used=self.config.model_type.value,
                inference_time_ms=(time.time() - start_time) * 1000,
                device_used=self.config.device,
                error=str(e)
            )
    
    def _parse_response(
        self, 
        text: str, 
        context: AnalysisContext
    ) -> Dict[str, Any]:
        """Parse LLM response into structured components."""
        result = {
            "summary": "",
            "key_points": [],
            "risks": [],
            "opportunities": [],
            "rationale": ""
        }
        
        lines = text.split('\n')
        current_section = None
        section_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect section headers
            lower_line = line.lower()
            if any(x in lower_line for x in ["executive summary", "summary:"]):
                if current_section and section_content:
                    result[current_section] = self._process_section(current_section, section_content)
                current_section = "summary"
                section_content = []
            elif any(x in lower_line for x in ["key strength", "strength", "positive"]):
                if current_section and section_content:
                    result[current_section] = self._process_section(current_section, section_content)
                current_section = "opportunities"
                section_content = []
            elif any(x in lower_line for x in ["key risk", "risk", "concern", "weakness"]):
                if current_section and section_content:
                    result[current_section] = self._process_section(current_section, section_content)
                current_section = "risks"
                section_content = []
            elif any(x in lower_line for x in ["recommendation", "rationale", "conclusion"]):
                if current_section and section_content:
                    result[current_section] = self._process_section(current_section, section_content)
                current_section = "rationale"
                section_content = []
            elif current_section:
                section_content.append(line)
        
        # Process last section
        if current_section and section_content:
            result[current_section] = self._process_section(current_section, section_content)
        
        # If no structured parsing worked, use the whole text
        if not result["summary"] and text:
            # Take first 2-3 sentences as summary
            sentences = text.split('.')
            result["summary"] = '. '.join(sentences[:3]).strip()
            if result["summary"] and not result["summary"].endswith('.'):
                result["summary"] += '.'
        
        # Combine key points from opportunities
        result["key_points"] = result["opportunities"][:3] if result["opportunities"] else []
        
        return result
    
    def _process_section(
        self, 
        section_type: str, 
        content: List[str]
    ) -> Any:
        """Process section content based on type."""
        if section_type in ["summary", "rationale"]:
            return ' '.join(content)
        else:
            # Extract bullet points
            points = []
            for line in content:
                # Remove bullet markers
                clean = line.lstrip('•-*123456789. ')
                if clean and len(clean) > 5:
                    points.append(clean)
            return points


class RuleBasedExplainer(AnalysisExplainer):
    """
    Fallback explainer using rule-based templates.
    
    Used when no LLM is available. Provides basic explanations
    based on score thresholds and metrics.
    """
    
    def load_model(self) -> bool:
        """No model to load for rule-based explainer."""
        self._status = ModelStatus.READY
        return True
    
    def unload_model(self) -> None:
        """Nothing to unload."""
        pass
    
    def explain(
        self,
        context: AnalysisContext,
        explanation_type: str = "full"
    ) -> ExplainerOutput:
        """Generate rule-based explanation."""
        start_time = time.time()
        
        try:
            # Generate summary
            summary = self._generate_summary(context)
            
            # Generate key points
            key_points = self._extract_key_points(context)
            
            # Identify risks
            risks = self._identify_risks(context)
            
            # Identify opportunities
            opportunities = self._identify_opportunities(context)
            
            # Generate rationale
            rationale = self._generate_rationale(context)
            
            # Combine for detailed analysis
            detailed = self._format_detailed_analysis(
                summary, key_points, risks, opportunities, rationale
            )
            
            inference_time = (time.time() - start_time) * 1000
            
            return ExplainerOutput(
                success=True,
                layer_name=self.layer_name,
                model_used="rule-based",
                inference_time_ms=inference_time,
                device_used="cpu",
                summary=summary,
                detailed_analysis=detailed,
                key_points=key_points,
                risk_factors=risks,
                opportunities=opportunities,
                recommendation_rationale=rationale,
            )
            
        except Exception as e:
            self._logger.error(f"Rule-based explanation failed: {e}")
            return ExplainerOutput(
                success=False,
                layer_name=self.layer_name,
                model_used="rule-based",
                inference_time_ms=(time.time() - start_time) * 1000,
                device_used="cpu",
                error=str(e)
            )
    
    def _generate_summary(self, context: AnalysisContext) -> str:
        """Generate a summary based on scores."""
        signal_descriptions = {
            "BUY": "shows strong investment potential",
            "HOLD": "presents a balanced risk-reward profile",
            "AVOID": "carries significant concerns",
            "SELL": "exhibits warning signs requiring attention"
        }
        
        desc = signal_descriptions.get(context.signal, "requires further analysis")
        
        # Identify strongest dimension
        dimensions = {
            "governance": context.governance_score,
            "financial trajectory": context.financial_score,
            "valuation": context.valuation_score,
            "market behaviour": context.market_score
        }
        
        strongest = max(dimensions, key=dimensions.get)
        weakest = min(dimensions, key=dimensions.get)
        
        summary = (
            f"{context.symbol} {desc} with a composite score of {context.composite_score:.0f}/100. "
            f"The stock's {strongest} ({dimensions[strongest]:.0f}/100) is its strongest dimension, "
            f"while {weakest} ({dimensions[weakest]:.0f}/100) needs attention."
        )
        
        return summary
    
    def _extract_key_points(self, context: AnalysisContext) -> List[str]:
        """Extract key points from metrics."""
        points = []
        
        metrics = context.metrics or {}
        
        if metrics.get("revenue_cagr_5y", 0) > 15:
            points.append(f"Strong revenue growth of {metrics['revenue_cagr_5y']:.1f}% CAGR over 5 years")
        
        if metrics.get("roce", 0) > 15:
            points.append(f"Healthy return on capital employed at {metrics['roce']:.1f}%")
        
        if metrics.get("promoter_holding", 0) > 50:
            points.append(f"High promoter confidence with {metrics['promoter_holding']:.1f}% stake")
        
        if context.forecast_trend == "bullish":
            points.append(f"Positive price momentum with {context.forecast_strength:.0%} trend strength")
        
        if not points:
            points.append(f"Composite analysis score: {context.composite_score:.0f}/100")
        
        return points[:5]
    
    def _identify_risks(self, context: AnalysisContext) -> List[str]:
        """Identify risk factors."""
        risks = list(context.red_flags or [])
        
        if context.governance_score < 50:
            risks.append("Below-average governance score indicates management concerns")
        
        if context.valuation_score < 40:
            risks.append("Stretched valuations relative to historical norms")
        
        if context.market_score < 40:
            risks.append("High volatility and poor market behaviour patterns")
        
        metrics = context.metrics or {}
        if metrics.get("debt_to_equity", 0) > 1.5:
            risks.append(f"High leverage with debt-to-equity of {metrics['debt_to_equity']:.2f}")
        
        if metrics.get("pledge_ratio", 0) > 20:
            risks.append(f"Elevated promoter pledge ratio at {metrics['pledge_ratio']:.1f}%")
        
        return risks[:5]
    
    def _identify_opportunities(self, context: AnalysisContext) -> List[str]:
        """Identify opportunities."""
        opportunities = []
        
        if context.financial_score > 70:
            opportunities.append("Strong financial trajectory supports growth")
        
        if context.valuation_score > 70:
            opportunities.append("Attractive valuations offer good entry point")
        
        if context.forecast_trend == "bullish" and context.forecast_strength > 0.5:
            opportunities.append("Strong positive price momentum")
        
        metrics = context.metrics or {}
        if metrics.get("fcf_yield", 0) > 5:
            opportunities.append(f"Healthy free cash flow yield of {metrics['fcf_yield']:.1f}%")
        
        if metrics.get("dividend_yield", 0) > 2:
            opportunities.append(f"Attractive dividend yield of {metrics['dividend_yield']:.1f}%")
        
        return opportunities[:5]
    
    def _generate_rationale(self, context: AnalysisContext) -> str:
        """Generate recommendation rationale."""
        signal = context.signal
        confidence = context.confidence
        
        if signal == "BUY":
            return (
                f"The {signal} recommendation with {confidence:.0%} confidence is based on "
                f"the stock's strong composite score of {context.composite_score:.0f}/100. "
                f"The combination of solid fundamentals and reasonable valuations suggests "
                f"potential for capital appreciation."
            )
        elif signal == "HOLD":
            return (
                f"The {signal} recommendation reflects a balanced view. While the stock has "
                f"some positive attributes (score: {context.composite_score:.0f}/100), there are "
                f"also areas of concern that warrant monitoring before adding to positions."
            )
        elif signal == "AVOID":
            return (
                f"The {signal} recommendation is driven by concerns identified in our analysis. "
                f"With a composite score of {context.composite_score:.0f}/100, the risk-reward "
                f"profile does not favor new investments at current levels."
            )
        else:  # SELL
            return (
                f"The {signal} recommendation indicates significant concerns. "
                f"The composite score of {context.composite_score:.0f}/100, combined with "
                f"identified red flags, suggests reducing exposure to manage risk."
            )
    
    def _format_detailed_analysis(
        self,
        summary: str,
        key_points: List[str],
        risks: List[str],
        opportunities: List[str],
        rationale: str
    ) -> str:
        """Format all components into detailed analysis."""
        sections = [
            "## Executive Summary",
            summary,
            "",
            "## Key Strengths",
        ]
        
        for point in opportunities:
            sections.append(f"• {point}")
        
        sections.extend(["", "## Risk Factors"])
        for risk in risks:
            sections.append(f"• {risk}")
        
        sections.extend([
            "",
            "## Recommendation Rationale",
            rationale
        ])
        
        return "\n".join(sections)


def create_explainer(
    model_type: ModelType,
    model_path: Path,
    device: str = "auto"
) -> AnalysisExplainer:
    """
    Factory function to create the appropriate explainer.
    
    Args:
        model_type: QWEN_3B, QWEN_7B, PHI3_MINI, or None for rule-based
        model_path: Path to model files
        device: "auto", "cpu", "cuda", or "mps"
    """
    config = ModelConfig(
        model_type=model_type,
        model_path=model_path,
        device=device,
        context_length=2048,
    )
    
    if model_type in [ModelType.QWEN_3B, ModelType.QWEN_7B]:
        return QwenExplainer(config)
    elif model_type == ModelType.PHI3_MINI:
        # Phi-3 uses same interface as Qwen
        return QwenExplainer(config)
    else:
        # Fallback to rule-based
        return RuleBasedExplainer(config)
