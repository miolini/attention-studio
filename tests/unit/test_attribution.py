import pytest
import torch
import torch.nn as nn


class TestAttributionModuleExists:
    def test_import_gradient_attribution(self):
        from attention_studio.core.attribution import GradientAttribution
        assert GradientAttribution is not None

    def test_import_attention_attribution(self):
        from attention_studio.core.attribution import AttentionAttribution
        assert AttentionAttribution is not None

    def test_import_feature_attribution(self):
        from attention_studio.core.attribution import FeatureAttribution
        assert FeatureAttribution is not None

    def test_import_neuron_level_attribution(self):
        from attention_studio.core.attribution import NeuronLevelAttribution
        assert NeuronLevelAttribution is not None

    def test_import_feature_interaction_attribution(self):
        from attention_studio.core.attribution import FeatureInteractionAttribution
        assert FeatureInteractionAttribution is not None


class TestAttributionClasses:
    def test_feature_attribution_class(self):
        from attention_studio.core.attribution import FeatureAttribution
        attr = FeatureAttribution(model=None)
        assert attr is not None

    def test_neuron_level_attribution_static(self):
        from attention_studio.core.attribution import NeuronLevelAttribution
        assert hasattr(NeuronLevelAttribution, 'attribute_to_neurons')

    def test_feature_interaction_static(self):
        from attention_studio.core.attribution import FeatureInteractionAttribution
        assert hasattr(FeatureInteractionAttribution, 'compute_feature_interactions')
