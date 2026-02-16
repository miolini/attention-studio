from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
from loguru import logger

from attention_studio.core.feature_extractor import FeatureExtractor


@dataclass
class AgentConfig:
    api_key: str = ""
    base_url: str = "https://api.minimax.chat/v1"
    model: str = "minimax-m2.5"
    temperature: float = 0.7
    max_tokens: int = 1024


@dataclass
class AgentResponse:
    content: str
    model: str
    usage: dict[str, int]
    confidence: float = 0.0


class BaseAgent:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.client = httpx.AsyncClient(
            base_url=config.base_url,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )
        self.history: list[dict[str, str]] = []

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AgentResponse:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
        }

        try:
            response = await self.client.post("/text/chatcompletion_v2", json=payload)
            response.raise_for_status()
            data = response.json()

            return AgentResponse(
                content=data["choices"][0]["message"]["content"],
                model=data.get("model", self.config.model),
                usage=data.get("usage", {}),
            )
        except httpx.HTTPError as e:
            logger.error(f"HTTP error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error: {e}")
            raise

    def add_to_history(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

    def clear_history(self):
        self.history = []


class FeatureAgent(BaseAgent):
    SYSTEM_PROMPT = """You are an expert in mechanistic interpretability of neural networks.
Your task is to analyze and interpret features discovered in a language model's hidden states.
Focus on the semantic meaning of features based on their activation patterns and top contexts."""

    def __init__(self, config: AgentConfig):
        super().__init__(config)

    async def interpret_feature(
        self,
        feature_extractor: FeatureExtractor,
        feature_idx: int,
        prompt: str,
        top_contexts: int = 10,
    ) -> AgentResponse:
        contexts = feature_extractor.get_top_contexts(prompt, feature_idx, top_contexts)

        context_str = "\n".join([
            f"Position {c['position']}: '{c['token']}' (activation: {c['activation']:.4f})"
            for c in contexts[:5]
        ])

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"""Analyze feature {feature_idx} in layer {feature_extractor.layer_idx}.

Top contexts from the prompt:
{context_str}

Based on this information, what semantic meaning does this feature likely represent?
Provide a brief interpretation (1-2 sentences)."""},
        ]

        response = await self.chat(messages)
        self.add_to_history("user", messages[1]["content"])
        self.add_to_history("assistant", response.content)

        return response


class CircuitAgent(BaseAgent):
    SYSTEM_PROMPT = """You are an expert in circuit discovery and mechanistic interpretability.
Your task is to identify known circuit patterns (induction heads, copy mechanisms, IOI, etc.)
based on feature patterns and graph structures in language models."""

    KNOWN_PATTERNS = {
        "induction": "Induction heads attend to previous tokens that are similar to current tokens, enabling in-context learning.",
        "copy": "Copy heads copy information from previous positions to current position.",
        "skip": "Skip heads bypass certain layers or attention patterns.",
        "ioi": "Indirect Object Identification - attending to subject tokens to predict objects.",
    }

    def __init__(self, config: AgentConfig):
        super().__init__(config)

    async def find_patterns(
        self,
        feature_info: list[dict[str, Any]],
        graph_stats: dict[str, Any],
    ) -> AgentResponse:
        patterns_str = "\n".join([
            f"- Feature {f['idx']}: activation={f['activation']:.4f}, layer={f['layer']}"
            for f in feature_info[:20]
        ])

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"""Analyze the following feature information to identify known circuit patterns:

Feature summary:
{patterns_str}

Graph statistics:
- Nodes: {graph_stats.get('num_nodes', 0)}
- Edges: {graph_stats.get('num_edges', 0)}
- Density: {graph_stats.get('density', 0):.4f}

Known patterns to look for:
{chr(10).join([f"- {k}: {v}" for k, v in self.KNOWN_PATTERNS.items()])}

Which patterns do you identify? Provide your analysis."""},
        ]

        response = await self.chat(messages)
        self.add_to_history("user", messages[1]["content"])
        self.add_to_history("assistant", response.content)

        return response


class VerifierAgent(BaseAgent):
    SYSTEM_PROMPT = """You are an expert in causal interventions and verification of interpretability hypotheses.
Your task is to design and analyze ablation experiments to verify feature/circuit functionality."""

    def __init__(self, config: AgentConfig):
        super().__init__(config)

    async def verify_hypothesis(
        self,
        hypothesis: str,
        feature_info: dict[str, Any],
    ) -> AgentResponse:
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"""Design an ablation experiment to verify this hypothesis:

Hypothesis: {hypothesis}

Feature info:
Layer: {feature_info.get('layer', 'N/A')}
Activation: {feature_info.get('activation', 0):.4f}

Describe:
1. How to ablate this feature
2. Expected effect on model behavior
3. How to measure success
4. Confidence score (0-1)"""},
        ]

        response = await self.chat(messages, temperature=0.3)
        self.add_to_history("user", messages[1]["content"])
        self.add_to_history("assistant", response.content)

        return response


class AgentManager:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.feature_agent = FeatureAgent(config)
        self.circuit_agent = CircuitAgent(config)
        self.verifier_agent = VerifierAgent(config)

    async def close(self):
        await self.feature_agent.client.aclose()
        await self.circuit_agent.client.aclose()
        await self.verifier_agent.client.aclose()
