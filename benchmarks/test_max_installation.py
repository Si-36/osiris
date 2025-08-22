# test_max_installation.py
"""Test if MAX is properly installed"""

def test_max_installation():
    """Test MAX Engine installation"""
    try:
        from max import engine
        print("✅ MAX Engine import successful")
        
        # Try creating a session
        session = engine.InferenceSession()
        print("✅ InferenceSession created successfully")
        
        # Check other imports
        from max.dtype import DType
        print("✅ DType import successful")
        
        from max.graph import Graph
        print("✅ Graph import successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ MAX Engine not properly installed: {e}")
        print("\nInstall with:")
        print("pip install modular --extra-index-url https://modular.gateway.scarf.sh/simple/")
        return False

if __name__ == "__main__":
    test_max_installation()
