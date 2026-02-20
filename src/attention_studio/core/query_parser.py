from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


class QueryType(Enum):
    FIND_INDUCTION = "find_induction"
    FIND_COPY = "find_copy"
    FIND_CIRCUITS = "find_circuits"
    EXTRACT_FEATURES = "extract_features"
    BUILD_GRAPH = "build_graph"
    ANALYZE_LAYER = "analyze_layer"
    SHOW_TOP_FEATURES = "show_top_features"
    COMPARE_PROMPTS = "compare_prompts"
    ABLATE_FEATURE = "ablate_feature"
    PATCH_ACTIVATION = "patch_activation"
    UNKNOWN = "unknown"


@dataclass
class ParsedQuery:
    query_type: QueryType
    parameters: dict[str, Any]
    original_text: str
    confidence: float


class NaturalLanguageParser:
    PATTERNS = {
        QueryType.FIND_INDUCTION: [
            r"find induction (heads?|circuits?)",
            r"detect induction",
            r"show induction",
            r"induction (heads?|patterns?)",
        ],
        QueryType.FIND_COPY: [
            r"find cop(y|ying) (heads?|circuits?)",
            r"detect cop(y|ying)",
            r"show cop(y|ying)",
        ],
        QueryType.FIND_CIRCUITS: [
            r"find (all )?circuits",
            r"detect circuits",
            r"show circuits",
            r"what circuits",
            r"analyze circuits",
        ],
        QueryType.EXTRACT_FEATURES: [
            r"extract features",
            r"get features",
            r"show features",
            r"analyze features",
            r"top features",
        ],
        QueryType.BUILD_GRAPH: [
            r"build (attribution )?graph",
            r"create graph",
            r"show graph",
            r"visualize (the )?graph",
        ],
        QueryType.ANALYZE_LAYER: [
            r"analyze layer (\d+)",
            r"layer (\d+) (analysis|features)",
            r"what('s| is) in layer (\d+)",
        ],
        QueryType.SHOW_TOP_FEATURES: [
            r"top (\d+) features",
            r"show top (\d+)",
            r"best (\d+) features",
            r"most active features",
        ],
        QueryType.COMPARE_PROMPTS: [
            r"compare (.+?) and (.+)",
            r"comparison between (.+?) and (.+)",
            r"diff(erence)? between (.+?) and (.+)",
        ],
        QueryType.ABLATE_FEATURE: [
            r"ablate feature (\d+)",
            r"remove feature (\d+)",
            r"disable feature (\d+)",
            r"what if feature (\d+) (is|was) (removed|disabled|zero)",
        ],
        QueryType.PATCH_ACTIVATION: [
            r"patch (activations? )?from (.+?) to (.+)",
            r"swap (activations? )?from (.+?) to (.+)",
        ],
    }

    LAYER_PATTERN = re.compile(r"layer\s*(\d+)", re.IGNORECASE)
    FEATURE_PATTERN = re.compile(r"feature\s*(\d+)", re.IGNORECASE)
    NUMBER_PATTERN = re.compile(r"\b(\d+)\b")
    TOP_K_PATTERN = re.compile(r"top\s*(\d+)", re.IGNORECASE)

    def __init__(self):
        self._compiled_patterns = {}
        for query_type, patterns in self.PATTERNS.items():
            self._compiled_patterns[query_type] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

    def parse(self, text: str) -> ParsedQuery:
        text_lower = text.lower().strip()

        best_match = None
        best_confidence = 0.0

        for query_type, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                match = pattern.search(text_lower)
                if match:
                    confidence = len(match.group()) / len(text_lower)
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_match = (query_type, match)

        if best_match:
            query_type, match = best_match
            params = self._extract_parameters(text, query_type, match)
            return ParsedQuery(
                query_type=query_type,
                parameters=params,
                original_text=text,
                confidence=best_confidence,
            )

        return ParsedQuery(
            query_type=QueryType.UNKNOWN,
            parameters={"text": text},
            original_text=text,
            confidence=0.0,
        )

    def _extract_parameters(self, text: str, query_type: QueryType, match: re.Match) -> dict[str, Any]:
        params = {}

        layer_match = self.LAYER_PATTERN.search(text)
        if layer_match:
            params["layer"] = int(layer_match.group(1))

        feature_match = self.FEATURE_PATTERN.search(text)
        if feature_match:
            params["feature_idx"] = int(feature_match.group(1))

        top_k_match = self.TOP_K_PATTERN.search(text)
        if top_k_match:
            params["top_k"] = int(top_k_match.group(1))

        if query_type == QueryType.ANALYZE_LAYER:
            for pattern in self._compiled_patterns[query_type]:
                m = pattern.search(text)
                if m:
                    for g in m.groups():
                        if g and g.isdigit():
                            params["layer"] = int(g)
                            break

        if query_type == QueryType.SHOW_TOP_FEATURES:
            for pattern in self._compiled_patterns[query_type]:
                m = pattern.search(text)
                if m:
                    for g in m.groups():
                        if g and g.isdigit():
                            params["top_k"] = int(g)
                            break

        if query_type == QueryType.COMPARE_PROMPTS:
            groups = match.groups()
            if len(groups) >= 2:
                params["prompt_a"] = groups[0].strip()
                params["prompt_b"] = groups[1].strip()

        if query_type == QueryType.PATCH_ACTIVATION:
            groups = match.groups()
            non_none_groups = [g for g in groups if g]
            if len(non_none_groups) >= 2:
                params["source_prompt"] = non_none_groups[-2].strip()
                params["target_prompt"] = non_none_groups[-1].strip()

        if query_type == QueryType.ABLATE_FEATURE:
            for pattern in self._compiled_patterns[query_type]:
                m = pattern.search(text)
                if m:
                    for g in m.groups():
                        if g and g.isdigit():
                            params["feature_idx"] = int(g)
                            break

        return params


class QueryExecutor:
    def __init__(
        self,
        model_manager: Any,
        trainer: Any,
        feature_extractor: Any = None,
    ):
        self.model_manager = model_manager
        self.trainer = trainer
        self.feature_extractor = feature_extractor
        self.parser = NaturalLanguageParser()

    def execute(self, query_text: str, prompt: str = "") -> dict[str, Any]:
        parsed = self.parser.parse(query_text)

        result = {
            "query_type": parsed.query_type.value,
            "parameters": parsed.parameters,
            "confidence": parsed.confidence,
            "success": False,
            "data": None,
            "error": None,
        }

        try:
            if parsed.query_type == QueryType.FIND_CIRCUITS:
                result["data"] = self._find_circuits(prompt)
                result["success"] = True

            elif parsed.query_type == QueryType.EXTRACT_FEATURES:
                layer = parsed.parameters.get("layer", 0)
                top_k = parsed.parameters.get("top_k", 20)
                result["data"] = self._extract_features(prompt, layer, top_k)
                result["success"] = True

            elif parsed.query_type == QueryType.BUILD_GRAPH:
                result["data"] = self._build_graph(prompt)
                result["success"] = True

            elif parsed.query_type == QueryType.ANALYZE_LAYER:
                layer = parsed.parameters.get("layer", 0)
                result["data"] = self._analyze_layer(layer, prompt)
                result["success"] = True

            elif parsed.query_type == QueryType.SHOW_TOP_FEATURES:
                top_k = parsed.parameters.get("top_k", 10)
                result["data"] = self._show_top_features(prompt, top_k)
                result["success"] = True

            elif parsed.query_type == QueryType.ABLATE_FEATURE:
                feature_idx = parsed.parameters.get("feature_idx", 0)
                result["data"] = self._ablate_feature(feature_idx, prompt)
                result["success"] = True

            elif parsed.query_type == QueryType.UNKNOWN:
                result["error"] = "Could not understand query"

        except Exception as e:
            result["error"] = str(e)

        return result

    def _find_circuits(self, prompt: str) -> dict[str, Any]:
        if not self.trainer or not self.trainer.transcoders:
            raise RuntimeError("CRM not built")

        from attention_studio.core.feature_extractor import GlobalCircuitAnalyzer

        analyzer = GlobalCircuitAnalyzer(
            self.model_manager,
            self.trainer.transcoders,
            getattr(self.trainer, 'lorsas', None),
            self.trainer.layer_indices,
        )
        return analyzer.analyze_all_circuits(prompt)

    def _extract_features(self, prompt: str, layer: int, top_k: int) -> list[dict]:
        if not self.trainer or not self.trainer.transcoders:
            raise RuntimeError("CRM not built")

        transcoder = self.trainer.get_transcoder(layer)
        if not transcoder:
            raise RuntimeError(f"No transcoder for layer {layer}")

        from attention_studio.core.feature_extractor import FeatureExtractor

        extractor = FeatureExtractor(self.model_manager, transcoder, layer)
        features = extractor.extract_features(prompt, top_k)
        return [{"idx": f.idx, "layer": f.layer, "activation": f.activation, "norm": f.norm} for f in features]

    def _build_graph(self, prompt: str) -> dict[str, Any]:
        if not self.trainer or not self.trainer.transcoders:
            raise RuntimeError("CRM not built")

        from attention_studio.core.attribution_graph import AttributionGraphBuilder

        builder = AttributionGraphBuilder(
            self.model_manager,
            self.trainer.transcoders,
            getattr(self.trainer, 'lorsas', None),
            self.trainer.layer_indices,
        )
        graph = builder.build_complete_attribution_graph(prompt)
        return {
            "num_nodes": len(graph.nodes),
            "num_edges": len(graph.edges),
        }

    def _analyze_layer(self, layer: int, prompt: str) -> dict[str, Any]:
        features = self._extract_features(prompt, layer, 50)
        return {
            "layer": layer,
            "features": features,
            "total_features": len(features),
        }

    def _show_top_features(self, prompt: str, top_k: int) -> list[dict]:
        all_features = []
        for layer in self.trainer.layer_indices:
            features = self._extract_features(prompt, layer, top_k)
            all_features.extend(features)
        all_features.sort(key=lambda x: abs(x.get("activation", 0)), reverse=True)
        return all_features[:top_k]

    def _ablate_feature(self, feature_idx: int, prompt: str) -> dict[str, Any]:
        if not self.feature_extractor:
            raise RuntimeError("No feature extractor available")
        return self.feature_extractor.ablate_feature(prompt, feature_idx)
