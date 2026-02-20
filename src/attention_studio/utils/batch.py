from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class BatchResult:
    prompt: str
    features: list[dict[str, Any]]
    circuits: dict[str, list[Any]]
    graph_stats: dict[str, int] | None
    error: str | None = None


class BatchProcessor:
    def __init__(
        self,
        model_manager: Any,
        trainer: Any,
        max_workers: int = 4,
    ):
        self.model_manager = model_manager
        self.trainer = trainer
        self.max_workers = max_workers
        self._cancel_flag = False

    def process_prompts(
        self,
        prompts: list[str],
        layer_indices: list[int] | None = None,
        extract_features: bool = True,
        find_circuits: bool = True,
        build_graphs: bool = False,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> list[BatchResult]:
        results = []
        self._cancel_flag = False

        total = len(prompts)
        completed = 0

        def process_single(prompt: str) -> BatchResult:
            if self._cancel_flag:
                return BatchResult(
                    prompt=prompt,
                    features=[],
                    circuits={},
                    graph_stats=None,
                    error="Cancelled",
                )

            features = []
            circuits = {}
            graph_stats = None

            try:
                if extract_features and self.trainer and self.trainer.transcoders:
                    from attention_studio.core.feature_extractor import FeatureExtractor

                    layers = layer_indices or self.trainer.layer_indices
                    for layer_idx in layers:
                        transcoder = self.trainer.get_transcoder(layer_idx)
                        if transcoder:
                            extractor = FeatureExtractor(
                                self.model_manager, transcoder, layer_idx
                            )
                            layer_features = extractor.extract_features(prompt, top_k=20)
                            for feat in layer_features:
                                feat.layer = layer_idx
                            features.extend([{
                                "idx": f.idx,
                                "layer": f.layer,
                                "activation": f.activation,
                                "norm": f.norm,
                            } for f in layer_features])

                if find_circuits and self.trainer and self.trainer.transcoders:
                    from attention_studio.core.feature_extractor import GlobalCircuitAnalyzer

                    analyzer = GlobalCircuitAnalyzer(
                        self.model_manager,
                        self.trainer.transcoders,
                        self.trainer.lorsas if hasattr(self.trainer, 'lorsas') else None,
                        self.trainer.layer_indices,
                    )
                    circuits = analyzer.analyze_all_circuits(prompt)

                if build_graphs and self.trainer and self.trainer.transcoders:
                    from attention_studio.core.attribution_graph import AttributionGraphBuilder

                    builder = AttributionGraphBuilder(
                        self.model_manager,
                        self.trainer.transcoders,
                        self.trainer.lorsas if hasattr(self.trainer, 'lorsas') else None,
                        self.trainer.layer_indices,
                    )
                    graph = builder.build_complete_attribution_graph(prompt)
                    graph_stats = {
                        "num_nodes": len(graph.nodes),
                        "num_edges": len(graph.edges),
                    }

            except Exception as e:
                return BatchResult(
                    prompt=prompt,
                    features=features,
                    circuits=circuits,
                    graph_stats=graph_stats,
                    error=str(e),
                )

            return BatchResult(
                prompt=prompt,
                features=features,
                circuits=circuits,
                graph_stats=graph_stats,
            )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_prompt = {
                executor.submit(process_single, prompt): prompt
                for prompt in prompts
            }

            for future in as_completed(future_to_prompt):
                if self._cancel_flag:
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

                prompt = future_to_prompt[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append(BatchResult(
                        prompt=prompt,
                        features=[],
                        circuits={},
                        graph_stats=None,
                        error=str(e),
                    ))

                completed += 1
                if progress_callback:
                    progress_callback(completed, total, prompt)

        return results

    def cancel(self):
        self._cancel_flag = True


class CheckpointManager:
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def list_checkpoints(self, model_name: str | None = None) -> list[dict[str, Any]]:
        checkpoints = []

        for ckpt_file in self.checkpoint_dir.rglob("*.pt"):
            try:
                relative = ckpt_file.relative_to(self.checkpoint_dir)
                parts = relative.parts

                if len(parts) >= 2:
                    ckpt_model = parts[0]
                    layer_str = parts[1].replace("layer_", "")
                    layer_idx = int(layer_str) if layer_str.isdigit() else None
                    epoch = None
                    if "epoch_" in ckpt_file.stem:
                        epoch_str = ckpt_file.stem.split("epoch_")[-1]
                        epoch = int(epoch_str) if epoch_str.isdigit() else None
                else:
                    ckpt_model = "unknown"
                    layer_idx = None
                    epoch = None

                if model_name and ckpt_model != model_name:
                    continue

                checkpoints.append({
                    "path": str(ckpt_file),
                    "model": ckpt_model,
                    "layer": layer_idx,
                    "epoch": epoch,
                    "size_mb": ckpt_file.stat().st_size / (1024 * 1024),
                    "modified": ckpt_file.stat().st_mtime,
                })
            except (ValueError, OSError):
                continue

        checkpoints.sort(key=lambda x: x.get("modified", 0), reverse=True)
        return checkpoints

    def load_checkpoint_info(self, checkpoint_path: Path) -> dict[str, Any] | None:
        import torch

        try:
            data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            return {
                "keys": list(data.keys()) if isinstance(data, dict) else [],
                "num_keys": len(data) if isinstance(data, dict) else 0,
            }
        except Exception:
            return None

    def delete_checkpoint(self, checkpoint_path: Path) -> bool:
        try:
            checkpoint_path.unlink()
            return True
        except OSError:
            return False

    def get_checkpoint_stats(self, model_name: str) -> dict[str, Any]:
        checkpoints = self.list_checkpoints(model_name)
        total_size = sum(c.get("size_mb", 0) for c in checkpoints)
        layers = {c.get("layer") for c in checkpoints if c.get("layer") is not None}

        return {
            "total_checkpoints": len(checkpoints),
            "total_size_mb": total_size,
            "layers_with_checkpoints": sorted(layers),
        }
