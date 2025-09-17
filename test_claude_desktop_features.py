#!/usr/bin/env python3
"""
Test runner for Claude Desktop intelligence features.
Run this to validate all the new functionality we added.
"""

import sys
import os
import subprocess
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all our new modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        from mcp_server.claude_integration import PrometheusChatBot
        print("✅ claude_integration.py imports successfully")
    except Exception as e:
        print(f"❌ claude_integration.py import failed: {e}")
        return False
    
    try:
        from mcp_server.tools.prometheus_tools import (
            find_best_metric_with_metadata_v2,
            search_metrics,
            execute_promql
        )
        print("✅ prometheus_tools.py imports successfully")
    except Exception as e:
        print(f"❌ prometheus_tools.py import failed: {e}")
        return False
    
    try:
        from mcp_server.observability_mcp import ObservabilityMCPServer
        print("✅ observability_mcp.py imports successfully")
    except Exception as e:
        print(f"❌ observability_mcp.py import failed: {e}")
        return False
    
    return True

def test_model_config():
    """Test model configuration."""
    print("\n📋 Testing model configuration...")
    
    try:
        with open('deploy/helm/model-config.json', 'r') as f:
            config = json.load(f)
        
        print(f"✅ model-config.json loaded with {len(config)} models")
        
        # Check for Anthropic models
        anthropic_models = [k for k in config.keys() if 'anthropic' in k]
        print(f"✅ Found {len(anthropic_models)} Anthropic models")
        
        # Check required fields
        for model_name, model_config in config.items():
            required_fields = ['external', 'requiresApiKey', 'provider']
            for field in required_fields:
                if field not in model_config:
                    print(f"❌ Missing field '{field}' in model '{model_name}'")
                    return False
        
        print("✅ All models have required fields")
        return True
        
    except Exception as e:
        print(f"❌ Model config test failed: {e}")
        return False

def test_dependencies():
    """Test that key dependencies are available."""
    print("\n📦 Testing dependencies...")
    
    dependencies = [
        'anthropic',
        'fastmcp', 
        'structlog',
        'trio',
        'dateparser'
    ]
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep} available")
        except ImportError:
            print(f"❌ {dep} missing")
            return False
    
    return True

def test_mcp_tools():
    """Test MCP tools functionality."""
    print("\n🔧 Testing MCP tools...")
    
    try:
        from mcp_server.tools.prometheus_tools import _extract_keywords_for_filtering
        
        # Test keyword extraction
        keywords = _extract_keywords_for_filtering("How many pods are running?")
        assert "pod" in keywords
        print("✅ Keyword extraction working")
        
        # Test scoring
        from mcp_server.tools.prometheus_tools import _score_metric_with_metadata_for_question
        
        score = _score_metric_with_metadata_for_question(
            "kube_pod_status_phase",
            "The phase of a pod", 
            "gauge",
            "How many pods are running?"
        )
        assert score > 0
        print("✅ Metric scoring working")
        
        return True
        
    except Exception as e:
        print(f"❌ MCP tools test failed: {e}")
        return False

def run_pytest():
    """Run the full pytest suite."""
    print("\n🧪 Running pytest suite...")
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            'tests/mcp_server/test_claude_integration.py',
            'tests/mcp_server/test_prometheus_tools_enhanced.py',
            'tests/ui/test_claude_desktop_ui.py',
            '-v'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ All pytest tests passed")
            return True
        else:
            print(f"❌ Some tests failed:\n{result.stdout}\n{result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ pytest execution failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Claude Desktop Intelligence Features")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Model Config Tests", test_model_config),
        ("Dependency Tests", test_dependencies),
        ("MCP Tools Tests", test_mcp_tools),
        ("Pytest Suite", run_pytest)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Claude Desktop intelligence features are working!")
        return True
    else:
        print("⚠️ Some features need attention.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
