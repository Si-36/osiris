#!/usr/bin/env python3
"""
🧠 AURA Intelligence - Main Entry Point

Clean entry point connecting AURA core engine with Ultimate API system.
"""

import sys
from pathlib import Path

def main():
    """Main function to start AURA Intelligence."""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                    🧠 AURA Intelligence                      ║
    ║                   Production Ready v3.0.0                   ║
    ║                                                              ║
    ║  🧠 Core Engine: core/src/aura_intelligence/                ║
    ║  🌐 API System: ultimate_api_system/                        ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    print("🚀 Starting AURA Intelligence Platform...")
    
    # Add paths
    core_path = Path(__file__).parent / "core" / "src"
    api_path = Path(__file__).parent / "ultimate_api_system"
    
    sys.path.insert(0, str(core_path))
    sys.path.insert(0, str(api_path))
    
    try:
        # Import and run the ultimate API system
        from ultimate_api_system.max_aura_api import main as api_main
        print("✅ Loaded Ultimate API System")
        print("📍 Starting server...")
        api_main()
        
    except ImportError as e:
        print(f"❌ Failed to import Ultimate API System: {e}")
        print("💡 Trying AURA Intelligence API...")
        
        try:
            # Try the aura_intelligence_api
            from aura_intelligence_api.ultimate_connected_system import UltimateConnectedSystem
            from aura_intelligence_api.ultimate_core_api import app
            import uvicorn
            
            print("✅ Loaded AURA Intelligence API")
            print("📍 Starting connected system server...")
            uvicorn.run(app, host="0.0.0.0", port=8080)
            
        except ImportError as e2:
            print(f"❌ Failed to import AURA Intelligence API: {e2}")
            print("💡 Trying simple fallback API...")
            
            try:
                pass
            # Fallback to simple API
            import uvicorn
            from fastapi import FastAPI
            
            app = FastAPI(title="AURA Intelligence", version="3.0.0")
            
            @app.get("/")
            def root():
                return {"message": "AURA Intelligence API", "status": "running"}
            
            @app.get("/health")
            def health():
                return {"status": "healthy"}
            
            print("🌐 Starting simple API server...")
            print("📍 Server will be available at: http://localhost:8080")
            print("📚 API docs at: http://localhost:8080/docs")
            uvicorn.run(app, host="0.0.0.0", port=8080)
            
        except ImportError:
            print("❌ FastAPI not available. Please install dependencies:")
            print("   pip install fastapi uvicorn")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n👋 Shutting down AURA Intelligence...")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()