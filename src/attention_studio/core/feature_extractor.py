from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
import torch


@dataclass
class FeatureInfo:
    idx: int
    layer: int
    activation: float
    norm: float
    top_contexts: list[dict[str, Any]] = None


@dataclass
class DLAResult:
    feature_logits: torch.Tensor
    feature_contributions: dict[int, float]


@dataclass
class CircuitInfo:
    source_feature: int
    target_feature: int
    source_layer: int
    target_layer: int
    circuit_type: str
    strength: float


@dataclass
class GlobalCircuit:
    name: str
    circuit_type: str
    features: list[tuple[int, int]]
    strength: float


class FeatureExtractor:
    def __init__(self, model_manager: Any, transcoder: Any, layer_idx: int):
        self.model_manager = model_manager
        self.transcoder = transcoder
        self.layer_idx = layer_idx

    def extract_features(
        self,
        prompt: str,
        top_k: int = 100,
    ) -> list[FeatureInfo]:
        if self.model_manager is None or not self.model_manager.is_loaded:
            raise RuntimeError("Model not loaded")

        tokenizer = self.model_manager.tokenizer
        model = self.model_manager.model

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states[self.layer_idx]

        _, features = self.transcoder(hidden_states)

        features = features.squeeze(0)
        feature_activations = features.mean(dim=0)

        top_indices = torch.topk(feature_activations, min(top_k, len(feature_activations)))

        feature_infos = []
        for _i, (idx, act) in enumerate(zip(top_indices.indices, top_indices.values, strict=True)):
            feature_idx_val = idx.item()
            norm = torch.norm(self.transcoder.decoder.weight[:, feature_idx_val]).item()
            feature_infos.append(FeatureInfo(
                idx=feature_idx_val,
                layer=self.layer_idx,
                activation=act.item(),
                norm=norm,
            ))

        return feature_infos

    def get_top_contexts(
        self,
        prompt: str,
        feature_idx: int,
        k: int = 10,
    ) -> list[dict[str, Any]]:
        tokenizer = self.model_manager.tokenizer
        model = self.model_manager.model

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states[self.layer_idx]

        _, features = self.transcoder(hidden_states)

        feature_values = features[0, :, feature_idx]
        top_positions = torch.topk(feature_values, min(k, len(feature_values)))

        token_strs = tokenizer.convert_ids_to_tokens(input_ids[0])

        contexts = []
        for pos, val in zip(top_positions.indices, top_positions.values, strict=True):
            contexts.append({
                "position": pos.item(),
                "token": token_strs[pos.item()],
                "activation": val.item(),
            })

        return contexts

    def compute_logit_lens(
        self,
        prompt: str,
        layer_idx: int | None = None,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        tokenizer = self.model_manager.tokenizer
        model = self.model_manager.model
        layer = layer_idx or self.layer_idx

        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        results = []

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

            if hasattr(model, 'lm_head') and model.lm_head is not None:
                unembed = model.lm_head
            elif hasattr(model, 'transformer') and hasattr(model.transformer, 'ln_f'):

                def unembed_fn(x):
                    if hasattr(model, 'lm_head'):
                        return model.lm_head(model.transformer.ln_f(x))
                    return None

                unembed = unembed_fn
            else:
                unembed = None

            if unembed is None:
                if hasattr(model, 'get_output_embeddings'):
                    output_embeddings = model.get_output_embeddings()
                    if output_embeddings is not None:
                        unembed = output_embeddings.weight
                    else:
                        return results
                else:
                    return results

            if not callable(unembed):
                unembed_weight = unembed
                if len(unembed_weight.shape) == 2:

                    def unembed_fn(x):
                        return torch.matmul(x, unembed_weight.t())

                    unembed = unembed_fn

        for pos in range(input_ids.shape[1]):
            if attention_mask[0, pos].item() == 0:
                continue

            hidden_state = outputs.hidden_states[layer][0, pos:pos+1]

            if callable(unembed):
                logits = unembed(hidden_state)
            else:
                continue

            probs = torch.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probs[0], min(top_k, probs.shape[-1]))

            token_preds = []
            for prob, idx in zip(top_probs, top_indices, strict=True):
                token_str = tokenizer.decode(idx.item())
                token_preds.append({
                    "token": token_str,
                    "probability": prob.item(),
                    "token_id": idx.item(),
                })

            results.append({
                "position": pos,
                "input_token": tokenizer.decode(input_ids[0, pos].item()),
                "predictions": token_preds,
            })

        return results

    def compute_dla(
        self,
        prompt: str,
        layer_idx: int | None = None,
    ) -> DLAResult:
        tokenizer = self.model_manager.tokenizer
        model = self.model_manager.model
        layer = layer_idx or self.layer_idx

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True,
            )

            hidden_states = outputs.hidden_states[layer]

        feature_activations = hidden_states @ self.transcoder.encoder.weight.T

        feature_logits = feature_activations @ self.transcoder.decoder.weight @ model.lm_head.weight

        last_token_logits = feature_logits[0, -1, :]

        contributions = {}
        for i in range(last_token_logits.shape[0]):
            if last_token_logits[i] > 0:
                contributions[i] = last_token_logits[i].item()

        return DLAResult(
            feature_logits=feature_logits,
            feature_contributions=contributions,
        )

    def ablate_feature(
        self,
        prompt: str,
        feature_idx: int,
        layer_idx: int | None = None,
        ablation_value: float = 0.0,
    ) -> dict[str, Any]:
        tokenizer = self.model_manager.tokenizer
        model = self.model_manager.model
        layer = layer_idx or self.layer_idx

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        with torch.no_grad():
            outputs_original = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

            original_logits = outputs_original.logits
            original_last_token_logits = original_logits[0, -1, :]
            original_top = torch.topk(torch.softmax(original_last_token_logits, dim=-1), 10)

        def ablation_forward_hook(module, inp, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            _, features = self.transcoder(hidden_states)
            features_ablated = features.clone()
            features_ablated[0, :, feature_idx] = ablation_value

            reconstructed = self.transcoder.decoder(features_ablated.view(-1, features_ablated.shape[-1]))
            reconstructed = reconstructed.view(hidden_states.shape)

            if isinstance(output, tuple):
                return (reconstructed,) + output[1:]
            return reconstructed

        layer_modules = None
        for name, module in model.named_modules():
            if f".h.{layer}" in name or f".layers.{layer}" in name:
                layer_modules = module
                break

        if layer_modules is None:
            return {
                "error": f"Could not find layer {layer} in model",
                "original_top_tokens": [],
                "ablated_top_tokens": [],
                "logit_diff": {},
            }

        hook = None
        try:
            hook = layer_modules.register_forward_hook(ablation_forward_hook)

            with torch.no_grad():
                outputs_ablated = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )

                ablated_logits = outputs_ablated.logits
                ablated_last_token_logits = ablated_logits[0, -1, :]
                ablated_top = torch.topk(torch.softmax(ablated_last_token_logits, dim=-1), 10)
        finally:
            if hook is not None:
                hook.remove()

        original_tokens = [
            {"token": tokenizer.decode(idx.item()), "prob": prob.item()}
            for prob, idx in zip(original_top.values, original_top.indices, strict=True)
        ]

        ablated_tokens = [
            {"token": tokenizer.decode(idx.item()), "prob": prob.item()}
            for prob, idx in zip(ablated_top.values, ablated_top.indices, strict=True)
        ]

        logit_diff = (ablated_last_token_logits - original_last_token_logits).abs()
        top_changed = torch.topk(logit_diff, 10)

        changed_tokens = [
            {"token": tokenizer.decode(idx.item()), "diff": diff.item()}
            for diff, idx in zip(top_changed.values, top_changed.indices, strict=True)
        ]

        return {
            "feature_idx": feature_idx,
            "layer": layer,
            "original_top_tokens": original_tokens,
            "ablated_top_tokens": ablated_tokens,
            "logit_diff": changed_tokens,
        }

    def patch_activation(
        self,
        source_prompt: str,
        target_prompt: str,
        layer_idx: int | None = None,
        positions: list[int] | None = None,
    ) -> dict[str, Any]:
        tokenizer = self.model_manager.tokenizer
        model = self.model_manager.model
        layer = layer_idx or self.layer_idx

        source_inputs = tokenizer(source_prompt, return_tensors="pt")
        source_input_ids = source_inputs["input_ids"].to(model.device)

        target_inputs = tokenizer(target_prompt, return_tensors="pt")
        target_input_ids = target_inputs["input_ids"].to(model.device)

        with torch.no_grad():
            source_outputs = model(
                input_ids=source_input_ids,
                output_hidden_states=True,
                return_dict=True,
            )

            target_outputs = model(
                input_ids=target_input_ids,
                output_hidden_states=True,
                return_dict=True,
            )

            source_hidden = source_outputs.hidden_states[layer]
            target_hidden = target_outputs.hidden_states[layer]

            original_target_logits = target_outputs.logits
            original_top = torch.topk(torch.softmax(original_target_logits[0, -1, :], dim=-1), 10)

        patched_hidden = target_hidden.clone()
        source_seq_len = source_hidden.shape[1]
        target_seq_len = target_hidden.shape[1]

        if positions is None:
            min_len = min(source_seq_len, target_seq_len)
            positions = list(range(min_len))

        for pos in positions:
            if pos < source_seq_len and pos < target_seq_len:
                patched_hidden[0, pos, :] = source_hidden[0, pos, :]

        layer_module = None
        for name, module in model.named_modules():
            if f".h.{layer}" in name or f".layers.{layer}" in name:
                layer_module = module
                break

        if layer_module is None:
            return {
                "error": f"Could not find layer {layer} in model",
                "original_top_tokens": [],
                "patched_top_tokens": [],
            }

        captured_patched = None

        def capture_hook(module, inp, output):
            nonlocal captured_patched
            captured_patched = output[0].clone() if isinstance(output, tuple) else output.clone()

        hook = layer_module.register_forward_hook(capture_hook)

        try:
            with torch.no_grad():
                _ = model(
                    input_ids=target_input_ids,
                    output_hidden_states=True,
                    return_dict=True,
                )

            if captured_patched is not None:
                captured_patched[:, :min(patched_hidden.shape[1], captured_patched.shape[1]), :] = patched_hidden[:, :min(patched_hidden.shape[1], captured_patched.shape[1]), :]

            with torch.no_grad():
                patched_outputs = model(
                    input_ids=target_input_ids,
                    output_hidden_states=True,
                    return_dict=True,
                )

                patched_logits = patched_outputs.logits
                patched_top = torch.topk(torch.softmax(patched_logits[0, -1, :], dim=-1), 10)
        finally:
            hook.remove()

        original_tokens = [
            {"token": tokenizer.decode(idx.item()), "prob": prob.item()}
            for prob, idx in zip(original_top.values, original_top.indices, strict=True)
        ]

        patched_tokens = [
            {"token": tokenizer.decode(idx.item()), "prob": prob.item()}
            for prob, idx in zip(patched_top.values, patched_top.indices, strict=True)
        ]

        return {
            "layer": layer,
            "positions": positions,
            "source_prompt": source_prompt,
            "target_prompt": target_prompt,
            "original_top_tokens": original_tokens,
            "patched_top_tokens": patched_tokens,
        }

    def compute_feature_importance(
        self,
        prompt: str,
        layer_idx: int | None = None,
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        tokenizer = self.model_manager.tokenizer
        model = self.model_manager.model
        layer = layer_idx or self.layer_idx

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True,
            )

            hidden_states = outputs.hidden_states[layer]
            _, features = self.transcoder(hidden_states)

        feature_acts = features[0].mean(dim=0)
        top_features = torch.topk(feature_acts.abs(), min(top_k, len(feature_acts)))

        importance = []
        for idx, act in zip(top_features.indices, top_features.values, strict=True):
            idx_val = idx.item()
            ablation_result = self.ablate_feature(prompt, idx_val, layer)
            logit_change = sum(d["diff"] for d in ablation_result.get("logit_diff", []))

            importance.append({
                "feature_idx": idx_val,
                "activation": act.item(),
                "logit_impact": logit_change,
            })

        importance.sort(key=lambda x: abs(x["logit_impact"]), reverse=True)
        return importance

    def cluster_features(
        self,
        prompts: list[str],
        n_clusters: int = 10,
        layer_idx: int | None = None,
    ) -> dict[str, Any]:
        try:
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA
        except ImportError:
            return {"error": "scikit-learn not installed", "clusters": []}

        layer = layer_idx or self.layer_idx
        all_activations = []

        for prompt in prompts:
            tokenizer = self.model_manager.tokenizer
            model = self.model_manager.model

            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(model.device)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden_states = outputs.hidden_states[layer]

            _, features = self.transcoder(hidden_states)
            features = features.squeeze(0)

            mean_activations = features.mean(dim=0).cpu().numpy()
            all_activations.append(mean_activations)

        if not all_activations:
            return {"error": "No activations computed", "clusters": []}

        activation_matrix = np.stack(all_activations, axis=0)

        if len(prompts) >= n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(activation_matrix.T)
        else:
            cluster_labels = np.zeros(activation_matrix.shape[1], dtype=int)

        pca = PCA(n_components=min(2, activation_matrix.shape[0]))
        coords = pca.fit_transform(activation_matrix.T)

        decoder_weights = self.transcoder.decoder.weight.detach().cpu().numpy()
        if decoder_weights.shape[0] > 2:
            weight_pca = PCA(n_components=min(2, min(decoder_weights.shape)))
            _weight_coords = weight_pca.fit_transform(decoder_weights)
        else:
            _weight_coords = np.zeros((decoder_weights.shape[0], 2))

        clusters = {}
        for feat_idx in range(activation_matrix.shape[1]):
            label = int(cluster_labels[feat_idx])
            if label not in clusters:
                clusters[label] = {
                    "features": [],
                    "mean_activation": 0.0,
                    "coords": [],
                }
            clusters[label]["features"].append(feat_idx)
            clusters[label]["mean_activation"] += float(activation_matrix[:, feat_idx].mean())
            clusters[label]["coords"].append({
                "pca_x": float(coords[feat_idx, 0]) if feat_idx < len(coords) else 0.0,
                "pca_y": float(coords[feat_idx, 1]) if feat_idx < len(coords) and coords.shape[1] > 1 else 0.0,
            })

        for label in clusters:
            clusters[label]["mean_activation"] /= len(clusters[label]["features"])

        return {
            "n_clusters": len(clusters),
            "clusters": clusters,
            "pca_explained_variance": pca.explained_variance_ratio_.tolist() if hasattr(pca, 'explained_variance_ratio_') else [],
        }


class GraphBuilder:
    def __init__(
        self,
        model_manager: Any,
        transcoders: list[Any],
        layer_indices: list[int],
    ):
        self.model_manager = model_manager
        self.transcoders = transcoders
        self.layer_indices = layer_indices

    def build_attribution_graph(
        self,
        prompt: str,
        threshold: float = 0.01,
    ) -> nx.DiGraph:
        tokenizer = self.model_manager.tokenizer
        model = self.model_manager.model

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states_list = outputs.hidden_states

        graph = nx.DiGraph()

        for layer_idx in self.layer_indices:
            hidden_states = hidden_states_list[layer_idx]
            tc_idx = self.layer_indices.index(layer_idx)
            transcoder = self.transcoders[tc_idx]

            _, features = transcoder(hidden_states)

            features = features.squeeze(0)
            feature_activations = features.mean(dim=0)

            for feat_idx, act in enumerate(feature_activations):
                if act.item() > threshold:
                    node_id = f"L{layer_idx}_F{feat_idx}"
                    graph.add_node(
                        node_id,
                        layer=layer_idx,
                        feature=feat_idx,
                        activation=act.item(),
                    )

            virtual_weights = transcoder.decoder.weight

            for i in range(min(100, virtual_weights.shape[0])):
                if feature_activations[i].item() > threshold:
                    for j in range(min(100, virtual_weights.shape[1])):
                        weight = virtual_weights[i, j].item()
                        if abs(weight) > threshold:
                            src = f"L{layer_idx}_F{i}"
                            dst = f"L{layer_idx}_F{j}"
                            if graph.has_edge(src, dst):
                                graph[src][dst]["weight"] += weight
                            else:
                                graph.add_edge(src, dst, weight=weight)

        self._add_layer_edges(graph)

        return graph

    def _add_layer_edges(self, graph: nx.DiGraph):
        for i in range(len(self.layer_indices) - 1):
            src_layer = self.layer_indices[i]
            dst_layer = self.layer_indices[i + 1]

            for feat in range(min(100, self.transcoders[i].config.dictionary_size)):
                src_node = f"L{src_layer}_F{feat}"
                dst_node = f"L{dst_layer}_F{feat}"

                if src_node in graph.nodes and dst_node in graph.nodes:
                    graph.add_edge(src_node, dst_node, weight=1.0, type="layer")

    def find_paths(
        self,
        graph: nx.DiGraph,
        source: str,
        target: str,
        max_length: int = 10,
    ) -> list[list[str]]:
        try:
            paths = list(nx.all_simple_paths(
                graph,
                source,
                target,
                cutoff=max_length,
            ))
            return paths
        except nx.NetworkXNoPath:
            return []

    def extract_subgraph(
        self,
        graph: nx.DiGraph,
        node: str,
        depth: int = 2,
    ) -> nx.DiGraph:
        ancestors = set()
        descendants = set()

        if depth > 0:
            ancestors = nx.ancestors(graph, node)
            descendants = nx.descendants(graph, node)

        nodes = {node} | ancestors | descendants

        return graph.subgraph(nodes).copy()

    def get_graph_stats(self, graph: nx.DiGraph) -> dict[str, Any]:
        return {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "density": nx.density(graph),
            "is_directed": graph.is_directed(),
        }


class GlobalCircuitAnalyzer:
    def __init__(
        self,
        model_manager: Any,
        transcoders: list[Any],
        lorsas: list[Any] | None = None,
        layer_indices: list[int] | None = None,
    ):
        self.model_manager = model_manager
        self.transcoders = transcoders
        self.lorsas = lorsas
        self.layer_indices = layer_indices or list(range(len(transcoders)))

    def compute_feature_circuits(
        self,
        layer_idx: int,
        feature_idx: int,
    ) -> dict[str, Any]:
        if layer_idx not in self.layer_indices:
            raise ValueError(f"Layer {layer_idx} not in layer_indices")

        tc_idx = self.layer_indices.index(layer_idx)
        transcoder = self.transcoders[tc_idx]

        decoder_weight = transcoder.decoder.weight
        encoder_weight = transcoder.encoder.weight

        feature_vec = decoder_weight[feature_idx]
        input_vec = encoder_weight[:, feature_idx]

        qk_circuit = None
        ov_circuit = None
        if self.lorsas and tc_idx < len(self.lorsas):
            lorsa = self.lorsas[tc_idx]
            qk_circuit = {
                "W_Q": lorsa.W_Q.weight,
                "W_K": lorsa.W_K.weight,
            }
            ov_circuit = {
                "w_V": lorsa.sparse_W_V,
                "w_O": lorsa.sparse_W_O,
            }

        return {
            "layer_idx": layer_idx,
            "feature_idx": feature_idx,
            "decoder_vec": feature_vec,
            "encoder_vec": input_vec,
            "qk_circuit": qk_circuit,
            "ov_circuit": ov_circuit,
            "norm": torch.norm(feature_vec).item(),
            "encoder_norm": torch.norm(input_vec).item(),
        }

    def find_induction_circuit(
        self,
        prompt: str,
        layer_idx: int,
        threshold: float = 0.1,
    ) -> list[GlobalCircuit]:
        tokenizer = self.model_manager.tokenizer
        model = self.model_manager.model

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)

        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        induction_features = []

        for i in range(1, len(tokens) - 1):
            if tokens[i] == tokens[i + 1]:
                for tc_idx, layer in enumerate(self.layer_indices):
                    if layer != layer_idx:
                        continue

                    transcoder = self.transcoders[tc_idx]
                    with torch.no_grad():
                        outputs = model(
                            input_ids=input_ids,
                            output_hidden_states=True,
                            return_dict=True,
                        )
                        hidden_states = outputs.hidden_states[layer]
                        _, features = transcoder(hidden_states)

                    features = features.squeeze(0)
                    feature_activations = features[i].abs()

                    top_features = torch.topk(feature_activations, min(10, len(feature_activations)))
                    for feat_idx, act in zip(top_features.indices, top_features.values, strict=True):
                        if act.item() > threshold:
                            induction_features.append((
                                layer,
                                feat_idx.item(),
                                act.item(),
                            ))

        if not induction_features:
            return []

        unique_features = {}
        for layer, feat, act in induction_features:
            key = (layer, feat)
            if key not in unique_features or unique_features[key] < act:
                unique_features[key] = act

        sorted_features = sorted(unique_features.items(), key=lambda x: x[1], reverse=True)[:20]

        return [
            GlobalCircuit(
                name="Induction",
                circuit_type="induction",
                features=[(layer, feat) for (layer, feat), _ in sorted_features],
                strength=sum(act for _, act in sorted_features) / len(sorted_features) if sorted_features else 0.0,
            )
        ]

    def find_attention_circuits(
        self,
        prompt: str,
        layer_idx: int,
        head_idx: int | None = None,
    ) -> list[dict[str, Any]]:
        if not self.lorsas:
            return []

        tokenizer = self.model_manager.tokenizer
        model = self.model_manager.model

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True,
            )

        if layer_idx not in self.layer_indices or not self.lorsas:
            return []

        tc_idx = self.layer_indices.index(layer_idx)
        if tc_idx >= len(self.lorsas):
            return []

        lorsa = self.lorsas[tc_idx]

        hidden_states = outputs.hidden_states[layer_idx]
        batch, seq_len, hidden_dim = hidden_states.shape

        Q = lorsa.W_Q(hidden_states)  # noqa: N806
        K = lorsa.W_K(hidden_states)  # noqa: N806
        V = lorsa.W_V(hidden_states)  # noqa: N806

        Q = Q.view(batch, seq_len, lorsa.num_heads, lorsa.head_dim).transpose(1, 2)  # noqa: N806
        K = K.view(batch, seq_len, lorsa.num_heads, lorsa.head_dim).transpose(1, 2)  # noqa: N806
        V = V.view(batch, seq_len, lorsa.num_heads, lorsa.head_dim).transpose(1, 2)  # noqa: N806

        if lorsa.config.qk_layernorm:
            Q = lorsa.q_layernorm(Q)  # noqa: N806
            K = lorsa.k_layernorm(K)  # noqa: N806

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (lorsa.head_dim ** 0.5)
        attn_probs = torch.softmax(scores, dim=-1)

        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        circuits = []
        num_heads = lorsa.num_heads if head_idx is None else 1
        head_range = range(num_heads) if head_idx is None else [head_idx]

        for h in head_range:
            attn_pattern = attn_probs[0, h]

            max_attn, max_pos = attn_pattern.max(dim=1)

            for pos in range(seq_len):
                if max_attn[pos].item() > 0.3:
                    circuits.append({
                        "layer": layer_idx,
                        "head": h,
                        "from_pos": pos,
                        "from_token": tokens[pos],
                        "to_pos": max_pos[pos].item(),
                        "to_token": tokens[max_pos[pos].item()],
                        "strength": max_attn[pos].item(),
                    })

        return circuits

    def find_copy_circuits(
        self,
        prompt: str,
        layer_idx: int,
    ) -> list[GlobalCircuit]:
        tokenizer = self.model_manager.tokenizer
        model = self.model_manager.model

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True,
            )

        copy_features = []

        for tc_idx, layer in enumerate(self.layer_indices):
            if layer != layer_idx:
                continue

            transcoder = self.transcoders[tc_idx]
            hidden_states = outputs.hidden_states[layer]
            _, features = transcoder(hidden_states)

            features = features.squeeze(0)

            for pos in range(1, len(tokens) - 1):
                if tokens[pos] == tokens[pos - 1]:
                    feature_activations = features[pos].abs()
                    top_feat = feature_activations.argmax().item()
                    copy_features.append((layer, top_feat, feature_activations[top_feat].item()))

        if not copy_features:
            return []

        unique_features = {}
        for layer, feat, act in copy_features:
            key = (layer, feat)
            if key not in unique_features or unique_features[key] < act:
                unique_features[key] = act

        sorted_features = sorted(unique_features.items(), key=lambda x: x[1], reverse=True)[:15]

        return [
            GlobalCircuit(
                name="Copying",
                circuit_type="copy",
                features=[(layer, feat) for (layer, feat), _ in sorted_features],
                strength=sum(act for _, act in sorted_features) / len(sorted_features) if sorted_features else 0.0,
            )
        ]

    def find_s_inhibition_circuit(
        self,
        prompt: str,
        layer_idx: int,
        threshold: float = 0.1,
    ) -> list[GlobalCircuit]:
        if not self.lorsas:
            return []

        tokenizer = self.model_manager.tokenizer
        model = self.model_manager.model

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True,
            )

        tc_idx = self.layer_indices.index(layer_idx)
        if tc_idx >= len(self.lorsas):
            return []

        lorsa = self.lorsas[tc_idx]
        hidden_states = outputs.hidden_states[layer_idx]
        batch, seq_len, hidden_dim = hidden_states.shape

        Q = lorsa.W_Q(hidden_states)  # noqa: N806
        K = lorsa.W_K(hidden_states)  # noqa: N806

        Q = Q.view(batch, seq_len, lorsa.num_heads, lorsa.head_dim).transpose(1, 2)  # noqa: N806
        K = K.view(batch, seq_len, lorsa.num_heads, lorsa.head_dim).transpose(1, 2)  # noqa: N806

        if lorsa.config.qk_layernorm:
            Q = lorsa.q_layernorm(Q)  # noqa: N806
            K = lorsa.k_layernorm(K)  # noqa: N806

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (lorsa.head_dim ** 0.5)
        attn_probs = torch.softmax(scores, dim=-1)

        s_inhibition_features = []
        special_tokens = {"[SEP]", "[PAD]", ".", ",", "!", "?", ";", ":"}

        for h in range(lorsa.num_heads):
            attn_pattern = attn_probs[0, h]

            for pos in range(1, seq_len - 1):
                current_token = tokens[pos]
                if current_token in special_tokens or current_token.startswith("Ċ"):
                    prev_attn = attn_pattern[pos, pos - 1].item()
                    if prev_attn > threshold:
                        s_inhibition_features.append((layer_idx, h, prev_attn))

        if not s_inhibition_features:
            return []

        unique_features = {}
        for layer, feat, act in s_inhibition_features:
            key = (layer, feat)
            if key not in unique_features or unique_features[key] < act:
                unique_features[key] = act

        sorted_features = sorted(unique_features.items(), key=lambda x: x[1], reverse=True)[:15]

        return [
            GlobalCircuit(
                name="S-Inhibition",
                circuit_type="s_inhibition",
                features=[(layer, feat) for (layer, feat), _ in sorted_features],
                strength=sum(act for _, act in sorted_features) / len(sorted_features) if sorted_features else 0.0,
            )
        ]

    def find_name_copying_circuit(
        self,
        prompt: str,
        layer_idx: int,
        threshold: float = 0.1,
    ) -> list[GlobalCircuit]:
        if not self.lorsas:
            return []

        tokenizer = self.model_manager.tokenizer
        model = self.model_manager.model

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True,
            )

        tc_idx = self.layer_indices.index(layer_idx)
        if tc_idx >= len(self.lorsas):
            return []

        lorsa = self.lorsas[tc_idx]
        hidden_states = outputs.hidden_states[layer_idx]
        batch, seq_len, hidden_dim = hidden_states.shape

        Q = lorsa.W_Q(hidden_states)  # noqa: N806
        K = lorsa.W_K(hidden_states)  # noqa: N806

        Q = Q.view(batch, seq_len, lorsa.num_heads, lorsa.head_dim).transpose(1, 2)  # noqa: N806
        K = K.view(batch, seq_len, lorsa.num_heads, lorsa.head_dim).transpose(1, 2)  # noqa: N806

        if lorsa.config.qk_layernorm:
            Q = lorsa.q_layernorm(Q)  # noqa: N806
            K = lorsa.k_layernorm(K)  # noqa: N806

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (lorsa.head_dim ** 0.5)
        attn_probs = torch.softmax(scores, dim=-1)

        name_copying_features = []
        name_positions = set()
        name_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

        for i, token in enumerate(tokens):
            if token and (token[0] in name_chars or token.startswith("Ċ") and len(token) > 1):
                name_positions.add(i)

        for h in range(lorsa.num_heads):
            attn_pattern = attn_probs[0, h]

            for pos in range(2, seq_len):
                if pos not in name_positions:
                    continue

                for prev_pos in range(pos - 1):
                    if prev_pos in name_positions:
                        attn_score = attn_pattern[pos, prev_pos].item()
                        if attn_score > threshold:
                            name_copying_features.append((layer_idx, h, attn_score))

        if not name_copying_features:
            return []

        unique_features = {}
        for layer, feat, act in name_copying_features:
            key = (layer, feat)
            if key not in unique_features or unique_features[key] < act:
                unique_features[key] = act

        sorted_features = sorted(unique_features.items(), key=lambda x: x[1], reverse=True)[:15]

        return [
            GlobalCircuit(
                name="Name Copying",
                circuit_type="name_copying",
                features=[(layer, feat) for (layer, feat), _ in sorted_features],
                strength=sum(act for _, act in sorted_features) / len(sorted_features) if sorted_features else 0.0,
            )
        ]

    def find_greater_than_circuit(
        self,
        prompt: str,
        layer_idx: int,
        threshold: float = 0.1,
    ) -> list[GlobalCircuit]:
        if not self.lorsas:
            return []

        tokenizer = self.model_manager.tokenizer
        model = self.model_manager.model

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True,
            )

        tc_idx = self.layer_indices.index(layer_idx)
        if tc_idx >= len(self.lorsas):
            return []

        lorsa = self.lorsas[tc_idx]
        hidden_states = outputs.hidden_states[layer_idx]
        batch, seq_len, hidden_dim = hidden_states.shape

        Q = lorsa.W_Q(hidden_states)  # noqa: N806
        K = lorsa.W_K(hidden_states)  # noqa: N806

        Q = Q.view(batch, seq_len, lorsa.num_heads, lorsa.head_dim).transpose(1, 2)  # noqa: N806
        K = K.view(batch, seq_len, lorsa.num_heads, lorsa.head_dim).transpose(1, 2)  # noqa: N806

        if lorsa.config.qk_layernorm:
            Q = lorsa.q_layernorm(Q)  # noqa: N806
            K = lorsa.k_layernorm(K)  # noqa: N806

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (lorsa.head_dim ** 0.5)
        attn_probs = torch.softmax(scores, dim=-1)

        greater_than_features = []
        comparison_tokens = {"more", "less", "greater", "bigger", "smaller", "larger", "higher", "lower", "better", "worse", "older", "younger"}

        for i, token in enumerate(tokens):
            token_lower = token.lower().replace("Ċ", "").replace("Ġ", "")
            if token_lower in comparison_tokens:
                for h in range(lorsa.num_heads):
                    attn_pattern = attn_probs[0, h]
                    for pos in range(i + 1, min(i + 5, seq_len)):
                        attn_score = attn_pattern[pos, i].item()
                        if attn_score > threshold:
                            greater_than_features.append((layer_idx, h, attn_score))

        if not greater_than_features:
            return []

        unique_features = {}
        for layer, feat, act in greater_than_features:
            key = (layer, feat)
            if key not in unique_features or unique_features[key] < act:
                unique_features[key] = act

        sorted_features = sorted(unique_features.items(), key=lambda x: x[1], reverse=True)[:15]

        return [
            GlobalCircuit(
                name="Greater-Than",
                circuit_type="greater_than",
                features=[(layer, feat) for (layer, feat), _ in sorted_features],
                strength=sum(act for _, act in sorted_features) / len(sorted_features) if sorted_features else 0.0,
            )
        ]

    def find_position_circuit(
        self,
        prompt: str,
        layer_idx: int,
        threshold: float = 0.15,
    ) -> list[GlobalCircuit]:
        if not self.lorsas:
            return []

        tokenizer = self.model_manager.tokenizer
        model = self.model_manager.model

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True,
            )

        tc_idx = self.layer_indices.index(layer_idx)
        if tc_idx >= len(self.lorsas):
            return []

        lorsa = self.lorsas[tc_idx]
        hidden_states = outputs.hidden_states[layer_idx]
        batch, seq_len, hidden_dim = hidden_states.shape

        Q = lorsa.W_Q(hidden_states)  # noqa: N806
        K = lorsa.W_K(hidden_states)  # noqa: N806

        Q = Q.view(batch, seq_len, lorsa.num_heads, lorsa.head_dim).transpose(1, 2)  # noqa: N806
        K = K.view(batch, seq_len, lorsa.num_heads, lorsa.head_dim).transpose(1, 2)  # noqa: N806

        if lorsa.config.qk_layernorm:
            Q = lorsa.q_layernorm(Q)  # noqa: N806
            K = lorsa.k_layernorm(K)  # noqa: N806

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (lorsa.head_dim ** 0.5)
        attn_probs = torch.softmax(scores, dim=-1)

        position_features = []

        for h in range(lorsa.num_heads):
            attn_pattern = attn_probs[0, h]

            diag_score = 0.0
            diag_count = 0
            for pos in range(1, seq_len - 1):
                prev_pos_attn = attn_pattern[pos, pos - 1].item()
                diag_score += prev_pos_attn
                diag_count += 1

            avg_diag = diag_score / max(diag_count, 1)

            if avg_diag > threshold:
                position_features.append((layer_idx, h, avg_diag))

            for pos in range(1, seq_len):
                first_token_attn = attn_pattern[pos, 0].item()
                if first_token_attn > threshold:
                    position_features.append((layer_idx, h, first_token_attn))

        if not position_features:
            return []

        unique_features = {}
        for layer, feat, act in position_features:
            key = (layer, feat)
            if key not in unique_features or unique_features[key] < act:
                unique_features[key] = act

        sorted_features = sorted(unique_features.items(), key=lambda x: x[1], reverse=True)[:15]

        return [
            GlobalCircuit(
                name="Position",
                circuit_type="position",
                features=[(layer, feat) for (layer, feat), _ in sorted_features],
                strength=sum(act for _, act in sorted_features) / len(sorted_features) if sorted_features else 0.0,
            )
        ]

    def find_token_composition_circuit(
        self,
        prompt: str,
        layer_idx: int,
        threshold: float = 0.1,
    ) -> list[GlobalCircuit]:
        tokenizer = self.model_manager.tokenizer
        model = self.model_manager.model

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        composition_features = []

        for tc_idx, layer in enumerate(self.layer_indices):
            if layer != layer_idx:
                continue

            transcoder = self.transcoders[tc_idx]
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden_states = outputs.hidden_states[layer]
                _, features = transcoder(hidden_states)

            features = features.squeeze(0)

            for pos in range(2, len(tokens) - 1):
                if tokens[pos] == tokens[pos - 1] + tokens[pos - 2]:
                    feature_activations = features[pos].abs()
                    top_feat = feature_activations.argmax().item()
                    composition_features.append((layer, top_feat, feature_activations[top_feat].item()))

        if not composition_features:
            return []

        unique_features = {}
        for layer, feat, act in composition_features:
            key = (layer, feat)
            if key not in unique_features or unique_features[key] < act:
                unique_features[key] = act

        sorted_features = sorted(unique_features.items(), key=lambda x: x[1], reverse=True)[:15]

        return [
            GlobalCircuit(
                name="Token Composition",
                circuit_type="token_composition",
                features=[(layer, feat) for (layer, feat), _ in sorted_features],
                strength=sum(act for _, act in sorted_features) / len(sorted_features) if sorted_features else 0.0,
            )
        ]

    def analyze_all_circuits(
        self,
        prompt: str,
    ) -> dict[str, list[GlobalCircuit]]:
        all_circuits = {}

        for layer_idx in self.layer_indices:
            induction = self.find_induction_circuit(prompt, layer_idx)
            if induction:
                all_circuits["induction"] = induction

            copy = self.find_copy_circuits(prompt, layer_idx)
            if copy:
                all_circuits["copy"] = copy

            if self.lorsas:
                attn_circuits = self.find_attention_circuits(prompt, layer_idx)
                if attn_circuits:
                    all_circuits[f"attention_layer_{layer_idx}"] = [
                        GlobalCircuit(
                            name=f"Attention L{layer_idx}",
                            circuit_type="attention",
                            features=[(layer_idx, 0)],
                            strength=sum(c["strength"] for c in attn_circuits) / len(attn_circuits),
                        )
                    ]

                s_inhibition = self.find_s_inhibition_circuit(prompt, layer_idx)
                if s_inhibition:
                    all_circuits["s_inhibition"] = s_inhibition

                name_copying = self.find_name_copying_circuit(prompt, layer_idx)
                if name_copying:
                    all_circuits["name_copying"] = name_copying

                greater_than = self.find_greater_than_circuit(prompt, layer_idx)
                if greater_than:
                    all_circuits["greater_than"] = greater_than

                position = self.find_position_circuit(prompt, layer_idx)
                if position:
                    all_circuits["position"] = position

            composition = self.find_token_composition_circuit(prompt, layer_idx)
            if composition:
                all_circuits["token_composition"] = composition

        return all_circuits
