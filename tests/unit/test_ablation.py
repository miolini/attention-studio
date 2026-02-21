import pytest
from attention_studio.core.ablation import (
    AblationResult,
    AblationStudy,
    FeatureAblator,
    LayerAblator,
    AblationStudyManager,
)


class TestAblationResult:
    def test_ablation_result_creation(self):
        result = AblationResult(
            original_score=0.9,
            ablated_score=0.5,
            change=0.4,
            change_percent=44.4,
            feature_indices=[0, 1, 2],
            layer_idx=5,
        )
        assert result.original_score == 0.9
        assert result.change == 0.4


class TestAblationStudy:
    def test_ablation_study_creation(self):
        study = AblationStudy(name="test_study")
        assert study.name == "test_study"
        assert len(study.results) == 0


class TestAblationStudyManager:
    def test_create_study(self):
        manager = AblationStudyManager()
        study = manager.create_study("test", {"description": "test study"})
        assert study is not None
        assert study.name == "test"

    def test_add_result(self):
        manager = AblationStudyManager()
        manager.create_study("test")
        result = AblationResult(
            original_score=0.9,
            ablated_score=0.5,
            change=0.4,
            change_percent=44.4,
            feature_indices=[0, 1],
            layer_idx=5,
        )
        manager.add_result("test", result)
        assert len(manager.get_study("test").results) == 1

    def test_get_study(self):
        manager = AblationStudyManager()
        manager.create_study("test")
        study = manager.get_study("test")
        assert study is not None
        assert study.name == "test"

    def test_compare_ablations(self):
        manager = AblationStudyManager()
        manager.create_study("test")
        
        result1 = AblationResult(0.9, 0.5, 0.4, 44.4, [0, 1], 5)
        result2 = AblationResult(0.9, 0.7, 0.2, 22.2, [2, 3], 5)
        
        manager.add_result("test", result1)
        manager.add_result("test", result2)
        
        comparison = manager.compare_ablations("test")
        assert "most_impactful" in comparison
        assert comparison["total_ablations"] == 2

    def test_rank_features(self):
        manager = AblationStudyManager()
        manager.create_study("test")
        
        result1 = AblationResult(0.9, 0.5, 0.4, 44.4, [0, 1], 5)
        result2 = AblationResult(0.9, 0.7, 0.2, 22.2, [1, 2], 5)
        
        manager.add_result("test", result1)
        manager.add_result("test", result2)
        
        ranked = manager.rank_features("test")
        assert isinstance(ranked, list)

    def test_get_nonexistent_study(self):
        manager = AblationStudyManager()
        study = manager.get_study("nonexistent")
        assert study is None

    def test_add_result_to_nonexistent_study(self):
        manager = AblationStudyManager()
        result = AblationResult(0.9, 0.5, 0.4, 44.4, [0, 1], 5)
        with pytest.raises(ValueError):
            manager.add_result("nonexistent", result)
