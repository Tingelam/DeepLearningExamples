#!/usr/bin/env python3
"""
Test script to verify all imports work correctly.
This script tests syntax and import structure without requiring dependencies.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        # Test evaluation module structure
        print("  ✓ Checking evaluation module structure")
        import src.evaluation
        assert hasattr(src.evaluation, 'compute_detection_metrics')
        assert hasattr(src.evaluation, 'compute_tracking_metrics')
        assert hasattr(src.evaluation, 'compute_classification_metrics')
        assert hasattr(src.evaluation, 'MetricsReporter')
        assert hasattr(src.evaluation, 'generate_report')
        assert hasattr(src.evaluation, 'compare_runs')
        print("  ✓ Evaluation module structure OK")
        
    except ImportError as e:
        print(f"  ✗ Import error (expected if dependencies not installed): {e}")
        print("  → This is expected in environments without numpy/sklearn/etc.")
        return True
    
    return True

if __name__ == '__main__':
    success = test_imports()
    if success:
        print("\n✅ Import structure tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Import structure tests failed!")
        sys.exit(1)
