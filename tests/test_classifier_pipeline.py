"""
Tests for EMG Classifier Pipeline Accuracy

This module tests the accuracy and consistency of the saved classifier pipeline,
including the XGBoost model, PCA transformation, and StandardScaler objects.

Tests verify:
- Artifact loading (model, PCA, scaler, event dictionary)
- Prediction consistency across multiple loads
- Transform consistency for PCA and scaler
- Complete end-to-end pipeline functionality
- Error handling for missing artifacts
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import xgboost as xgb
from pickle import load
import json
import tempfile
import shutil

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ClassifierHandler import ClassifierHandler, get_paths
from preprocessing import generate_final_features


class TestArtifactLoading:
    """Test suite for verifying artifact loading functionality"""

    @pytest.fixture(scope="class")
    def paths(self):
        """Get artifact and model paths"""
        artifact_dir, model_save_dir = get_paths()
        return artifact_dir, model_save_dir

    @pytest.fixture(scope="class")
    def dummy_env_data(self, tmp_path_factory):
        """Create dummy EMG envelope data for testing"""
        # Create a temporary directory for test data
        tmp_dir = tmp_path_factory.mktemp("test_data")
        env_path = os.path.join(tmp_dir, "test_env.npy")
        
        # Create dummy EMG data: (n_tastes=2, n_trials=3, n_timepoints=7000)
        # This simulates EMG envelope data with realistic dimensions
        dummy_env = np.random.rand(2, 3, 7000) * 100
        np.save(env_path, dummy_env)
        
        return env_path

    def test_xgb_model_exists(self, paths):
        """Verify XGBoost model file exists"""
        artifact_dir, model_save_dir = paths
        model_path = os.path.join(model_save_dir, 'xgb_model.json')
        assert os.path.exists(model_path), f"XGBoost model not found at {model_path}"

    def test_pca_object_exists(self, paths):
        """Verify PCA object file exists"""
        artifact_dir, model_save_dir = paths
        pca_path = os.path.join(artifact_dir, 'pca_obj.pkl')
        assert os.path.exists(pca_path), f"PCA object not found at {pca_path}"

    def test_scaler_object_exists(self, paths):
        """Verify scaler object file exists"""
        artifact_dir, model_save_dir = paths
        scaler_path = os.path.join(artifact_dir, 'scale_obj.pkl')
        assert os.path.exists(scaler_path), f"Scaler object not found at {scaler_path}"

    def test_event_dict_exists(self, paths):
        """Verify event code dictionary file exists"""
        artifact_dir, model_save_dir = paths
        event_dict_path = os.path.join(artifact_dir, 'event_code_dict.json')
        assert os.path.exists(event_dict_path), f"Event dictionary not found at {event_dict_path}"

    def test_load_xgb_model(self, paths, dummy_env_data):
        """Verify XGBoost model loads correctly and has expected properties"""
        artifact_dir, model_save_dir = paths
        
        handler = ClassifierHandler(
            model_dir=model_save_dir,
            output_dir=artifact_dir,
            env_path=dummy_env_data
        )
        
        clf = handler.load_model()
        
        # Verify model type
        assert isinstance(clf, xgb.XGBClassifier), "Loaded model is not an XGBClassifier"
        
        # Verify model has expected number of classes (3: no movement, gape, MTMs)
        assert hasattr(clf, 'n_classes_'), "Model does not have n_classes_ attribute"
        assert clf.n_classes_ == 3, f"Expected 3 classes, got {clf.n_classes_}"

    def test_load_pca_object(self, paths):
        """Verify PCA object loads correctly and has expected properties"""
        artifact_dir, model_save_dir = paths
        pca_path = os.path.join(artifact_dir, 'pca_obj.pkl')
        
        with open(pca_path, 'rb') as f:
            pca_obj = load(f)
        
        # Verify PCA has 3 components
        assert pca_obj.n_components == 3, f"Expected 3 PCA components, got {pca_obj.n_components}"
        
        # Verify PCA is fitted (has components_)
        assert hasattr(pca_obj, 'components_'), "PCA object is not fitted"
        assert pca_obj.components_.shape[0] == 3, "PCA components shape mismatch"

    def test_load_scaler_object(self, paths):
        """Verify scaler object loads correctly and has expected properties"""
        artifact_dir, model_save_dir = paths
        scaler_path = os.path.join(artifact_dir, 'scale_obj.pkl')
        
        with open(scaler_path, 'rb') as f:
            scaler_obj = load(f)
        
        # Verify scaler is fitted (has mean_ and scale_)
        assert hasattr(scaler_obj, 'mean_'), "Scaler object is not fitted (missing mean_)"
        assert hasattr(scaler_obj, 'scale_'), "Scaler object is not fitted (missing scale_)"
        
        # Verify scaler has 8 features (5 original + 3 PCA)
        assert len(scaler_obj.mean_) == 8, f"Expected 8 features, got {len(scaler_obj.mean_)}"

    def test_load_event_dict(self, paths, dummy_env_data):
        """Verify event dictionary loads correctly and has expected mappings"""
        artifact_dir, model_save_dir = paths
        
        handler = ClassifierHandler(
            model_dir=model_save_dir,
            output_dir=artifact_dir,
            env_path=dummy_env_data
        )
        
        event_dict = handler.load_event_types()
        
        # Verify expected keys exist
        expected_keys = {"gape", "MTMs", "no movement"}
        assert set(event_dict.keys()) == expected_keys, \
            f"Event dictionary keys mismatch. Expected {expected_keys}, got {set(event_dict.keys())}"
        
        # Verify expected values
        assert event_dict["no movement"] == 0, "Incorrect code for 'no movement'"
        assert event_dict["gape"] == 1, "Incorrect code for 'gape'"
        assert event_dict["MTMs"] == 2, "Incorrect code for 'MTMs'"


class TestPredictionConsistency:
    """Test suite for verifying prediction consistency"""

    @pytest.fixture(scope="class")
    def paths(self):
        """Get artifact and model paths"""
        artifact_dir, model_save_dir = get_paths()
        return artifact_dir, model_save_dir

    @pytest.fixture(scope="class")
    def dummy_env_data(self, tmp_path_factory):
        """Create dummy EMG envelope data for testing"""
        tmp_dir = tmp_path_factory.mktemp("test_data")
        env_path = os.path.join(tmp_dir, "test_env.npy")
        
        # Create dummy EMG data with realistic structure
        # Use fixed seed for reproducibility
        np.random.seed(42)
        dummy_env = np.random.rand(2, 3, 7000) * 100
        np.save(env_path, dummy_env)
        
        return env_path

    def test_model_prediction_consistency(self, paths, dummy_env_data):
        """Verify model produces consistent predictions across multiple loads"""
        artifact_dir, model_save_dir = paths
        
        # Create dummy features for testing
        np.random.seed(42)
        X_test = np.random.rand(10, 8)  # 10 samples, 8 features
        
        # Load model twice and make predictions
        handler1 = ClassifierHandler(
            model_dir=model_save_dir,
            output_dir=artifact_dir,
            env_path=dummy_env_data
        )
        clf1 = handler1.load_model()
        pred1 = clf1.predict(X_test)
        proba1 = clf1.predict_proba(X_test)
        
        handler2 = ClassifierHandler(
            model_dir=model_save_dir,
            output_dir=artifact_dir,
            env_path=dummy_env_data
        )
        clf2 = handler2.load_model()
        pred2 = clf2.predict(X_test)
        proba2 = clf2.predict_proba(X_test)
        
        # Verify predictions are identical
        np.testing.assert_array_equal(pred1, pred2, 
            err_msg="Model predictions differ across loads")
        
        # Verify probabilities are identical
        np.testing.assert_allclose(proba1, proba2, rtol=1e-10,
            err_msg="Model prediction probabilities differ across loads")

    def test_pca_transform_consistency(self, paths):
        """Verify PCA produces consistent transformations across multiple loads"""
        artifact_dir, model_save_dir = paths
        pca_path = os.path.join(artifact_dir, 'pca_obj.pkl')
        
        # Create dummy segment data (100 timepoints per segment)
        np.random.seed(42)
        X_test = np.random.rand(10, 100)
        
        # Load PCA twice and transform
        with open(pca_path, 'rb') as f:
            pca1 = load(f)
        transform1 = pca1.transform(X_test)[:, :3]
        
        with open(pca_path, 'rb') as f:
            pca2 = load(f)
        transform2 = pca2.transform(X_test)[:, :3]
        
        # Verify transforms are identical
        np.testing.assert_allclose(transform1, transform2, rtol=1e-10,
            err_msg="PCA transforms differ across loads")

    def test_scaler_transform_consistency(self, paths):
        """Verify scaler produces consistent transformations across multiple loads"""
        artifact_dir, model_save_dir = paths
        scaler_path = os.path.join(artifact_dir, 'scale_obj.pkl')
        
        # Create dummy feature data (8 features)
        np.random.seed(42)
        X_test = np.random.rand(10, 8)
        
        # Load scaler twice and transform
        with open(scaler_path, 'rb') as f:
            scaler1 = load(f)
        transform1 = scaler1.transform(X_test)
        
        with open(scaler_path, 'rb') as f:
            scaler2 = load(f)
        transform2 = scaler2.transform(X_test)
        
        # Verify transforms are identical
        np.testing.assert_allclose(transform1, transform2, rtol=1e-10,
            err_msg="Scaler transforms differ across loads")


class TestEndToEndPipeline:
    """Test suite for complete pipeline functionality"""

    @pytest.fixture(scope="class")
    def paths(self):
        """Get artifact and model paths"""
        artifact_dir, model_save_dir = get_paths()
        return artifact_dir, model_save_dir

    @pytest.fixture(scope="class")
    def dummy_env_data(self, tmp_path_factory):
        """Create dummy EMG envelope data for testing"""
        tmp_dir = tmp_path_factory.mktemp("test_data")
        env_path = os.path.join(tmp_dir, "test_env.npy")
        
        # Create dummy EMG data with realistic structure
        np.random.seed(42)
        dummy_env = np.random.rand(2, 3, 7000) * 100
        np.save(env_path, dummy_env)
        
        return env_path

    def test_complete_pipeline_runs(self, paths, dummy_env_data):
        """Verify complete inference pipeline executes without errors"""
        artifact_dir, model_save_dir = paths
        
        handler = ClassifierHandler(
            model_dir=model_save_dir,
            output_dir=artifact_dir,
            env_path=dummy_env_data
        )
        
        # Run complete pipeline
        y_pred, segment_frame, feature_names = handler.parse_and_predict()
        
        # Verify outputs exist and have expected structure
        assert y_pred is not None, "Predictions are None"
        assert segment_frame is not None, "Segment frame is None"
        assert feature_names is not None, "Feature names are None"
        
        # Verify predictions are valid class indices
        assert all(p in [0, 1, 2] for p in y_pred), \
            "Predictions contain invalid class indices"

    def test_segment_frame_structure(self, paths, dummy_env_data):
        """Verify segment frame has expected columns and structure"""
        artifact_dir, model_save_dir = paths
        
        handler = ClassifierHandler(
            model_dir=model_save_dir,
            output_dir=artifact_dir,
            env_path=dummy_env_data
        )
        
        y_pred, segment_frame, feature_names = handler.parse_and_predict()
        
        # Verify expected columns exist
        expected_columns = {
            'features', 'segment_raw', 'segment_norm_interp', 
            'segment_bounds', 'taste', 'trial', 'raw_features',
            'pred', 'pred_names', 'pred_proba'
        }
        
        assert expected_columns.issubset(set(segment_frame.columns)), \
            f"Missing expected columns. Expected {expected_columns}, got {set(segment_frame.columns)}"
        
        # Verify data types
        assert segment_frame['pred'].dtype in [np.int32, np.int64], \
            "Predictions should be integers"
        assert segment_frame['pred_names'].dtype == object, \
            "Prediction names should be strings"

    def test_feature_names_count(self, paths, dummy_env_data):
        """Verify correct number of features (8: 5 original + 3 PCA)"""
        artifact_dir, model_save_dir = paths
        
        handler = ClassifierHandler(
            model_dir=model_save_dir,
            output_dir=artifact_dir,
            env_path=dummy_env_data
        )
        
        y_pred, segment_frame, feature_names = handler.parse_and_predict()
        
        # Verify 8 features total
        assert len(feature_names) == 8, \
            f"Expected 8 features, got {len(feature_names)}"
        
        # Verify PCA features are included
        pca_features = [f for f in feature_names if 'pca' in f.lower()]
        assert len(pca_features) == 3, \
            f"Expected 3 PCA features, got {len(pca_features)}"

    def test_prediction_names_match_codes(self, paths, dummy_env_data):
        """Verify prediction names correctly map to prediction codes"""
        artifact_dir, model_save_dir = paths
        
        handler = ClassifierHandler(
            model_dir=model_save_dir,
            output_dir=artifact_dir,
            env_path=dummy_env_data
        )
        
        y_pred, segment_frame, feature_names = handler.parse_and_predict()
        
        # Load event dictionary
        event_dict = handler.load_event_types()
        inv_event_dict = {v: k for k, v in event_dict.items()}
        
        # Verify each prediction name matches its code
        for idx, row in segment_frame.iterrows():
            expected_name = inv_event_dict[row['pred']]
            assert row['pred_names'] == expected_name, \
                f"Prediction name mismatch at index {idx}: expected {expected_name}, got {row['pred_names']}"

    def test_prediction_probabilities_sum_to_one(self, paths, dummy_env_data):
        """Verify prediction probabilities sum to 1.0 for each sample"""
        artifact_dir, model_save_dir = paths
        
        handler = ClassifierHandler(
            model_dir=model_save_dir,
            output_dir=artifact_dir,
            env_path=dummy_env_data
        )
        
        y_pred, segment_frame, feature_names = handler.parse_and_predict()
        
        # Check each prediction probability sums to 1.0
        for idx, proba in enumerate(segment_frame['pred_proba']):
            proba_sum = np.sum(proba)
            assert np.isclose(proba_sum, 1.0, rtol=1e-5), \
                f"Probabilities at index {idx} sum to {proba_sum}, expected 1.0"

    def test_pipeline_consistency_across_runs(self, paths, dummy_env_data):
        """Verify pipeline produces identical results across multiple runs"""
        artifact_dir, model_save_dir = paths
        
        # Run pipeline twice
        handler1 = ClassifierHandler(
            model_dir=model_save_dir,
            output_dir=artifact_dir,
            env_path=dummy_env_data
        )
        y_pred1, segment_frame1, feature_names1 = handler1.parse_and_predict()
        
        handler2 = ClassifierHandler(
            model_dir=model_save_dir,
            output_dir=artifact_dir,
            env_path=dummy_env_data
        )
        y_pred2, segment_frame2, feature_names2 = handler2.parse_and_predict()
        
        # Verify predictions are identical
        np.testing.assert_array_equal(y_pred1, y_pred2,
            err_msg="Predictions differ across pipeline runs")
        
        # Verify feature names are identical
        np.testing.assert_array_equal(feature_names1, feature_names2,
            err_msg="Feature names differ across pipeline runs")
        
        # Verify segment frame predictions are identical
        np.testing.assert_array_equal(
            segment_frame1['pred'].values,
            segment_frame2['pred'].values,
            err_msg="Segment frame predictions differ across pipeline runs"
        )


class TestErrorHandling:
    """Test suite for error handling with missing or corrupted artifacts"""

    @pytest.fixture(scope="class")
    def paths(self):
        """Get artifact and model paths"""
        artifact_dir, model_save_dir = get_paths()
        return artifact_dir, model_save_dir

    @pytest.fixture(scope="class")
    def dummy_env_data(self, tmp_path_factory):
        """Create dummy EMG envelope data for testing"""
        tmp_dir = tmp_path_factory.mktemp("test_data")
        env_path = os.path.join(tmp_dir, "test_env.npy")
        
        np.random.seed(42)
        dummy_env = np.random.rand(2, 3, 7000) * 100
        np.save(env_path, dummy_env)
        
        return env_path

    def test_missing_model_raises_error(self, paths, dummy_env_data, tmp_path):
        """Verify appropriate error when model file is missing"""
        artifact_dir, model_save_dir = paths
        
        # Create temporary directory without model
        temp_model_dir = tmp_path / "empty_model_dir"
        temp_model_dir.mkdir()
        
        handler = ClassifierHandler(
            model_dir=str(temp_model_dir),
            output_dir=artifact_dir,
            env_path=dummy_env_data
        )
        
        # Verify error is raised when trying to load missing model
        with pytest.raises(Exception):  # XGBoost raises various exceptions
            handler.load_model()

    def test_missing_event_dict_raises_error(self, paths, dummy_env_data, tmp_path):
        """Verify appropriate error when event dictionary is missing"""
        artifact_dir, model_save_dir = paths
        
        # Create temporary directory without event dict
        temp_artifact_dir = tmp_path / "empty_artifact_dir"
        temp_artifact_dir.mkdir()
        
        handler = ClassifierHandler(
            model_dir=model_save_dir,
            output_dir=str(temp_artifact_dir),
            env_path=dummy_env_data
        )
        
        # Verify error is raised when trying to load missing event dict
        with pytest.raises(FileNotFoundError):
            handler.load_event_types()

    def test_invalid_env_path_raises_error(self, paths):
        """Verify appropriate error when EMG envelope path is invalid"""
        artifact_dir, model_save_dir = paths
        
        handler = ClassifierHandler(
            model_dir=model_save_dir,
            output_dir=artifact_dir,
            env_path="/nonexistent/path/to/env.npy"
        )
        
        # Verify error is raised when trying to load missing env file
        with pytest.raises(FileNotFoundError):
            handler.load_env_file()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
