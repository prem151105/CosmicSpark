#!/usr/bin/env python3
"""
CosmicSpark - Main Runner Script
Provides easy commands to run different components of the system
"""

import argparse
import subprocess
import sys
import os
import time
import signal
from pathlib import Path
import webbrowser

def run_command(command: str, cwd: str = None, wait: bool = True):
    """Run a command with proper error handling"""
    print(f"🚀 Running: {command}")
    try:
        if wait:
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                check=True
            )
            return result.returncode == 0
        else:
            # For non-blocking commands
            return subprocess.Popen(
                command,
                shell=True,
                cwd=cwd,
                start_new_session=True
            )
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
        return False

def setup_environment():
    """Setup the development environment"""
    print("🔧 Setting up CosmicSpark environment...")
    return run_command("python scripts/setup_environment.py")

def start_backend():
    """Start the FastAPI backend"""
    print("🔧 Starting backend API...")
    return run_command("uvicorn src.api.main:app --reload", wait=False)

def start_frontend():
    """Start the Streamlit frontend"""
    print("🎨 Starting frontend...")
    return run_command("streamlit run frontend/streamlit_app.py", wait=False)

def run_crawler():
    """Run the web crawler"""
    print("🕷️  Running web crawler...")
    return run_command("python scripts/run_crawler.py --url https://www.mosdac.gov.in --max-pages 50 --update-vs --update-kg")

def build_knowledge_graph():
    """Build the knowledge graph"""
    print("🕸️  Building knowledge graph...")
    if not Path("data/crawled_data.json").exists():
        print("❌ No crawled data found. Run crawler first.")
        return False
    return run_command("python scripts/build_knowledge_graph.py --input data/crawled_data.json --incremental")

def run_tests():
    """Run the test suite"""
    print("🧪 Running tests...")
    return run_command("pytest tests/ -v")

def check_health():
    """Check system health"""
    print("🏥 Checking system health...")
    
    # Check if services are running
    services = [
        ("Backend API", "http://localhost:8000/health"),
        ("Frontend", "http://localhost:8501")
    ]
    
    try:
        import requests
        for name, url in services:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    print(f"✅ {name}: Running")
                else:
                    print(f"⚠️  {name}: Responding but status {response.status_code}")
            except requests.exceptions.RequestException:
                print(f"❌ {name}: Not accessible")
    except ImportError:
        print("❌ requests library not installed. Run: pip install requests")
        return False
    
    return True

def show_logs():
    """Show application logs"""
    print("📋 Showing recent logs...")
    log_file = Path("logs/app.log")
    if log_file.exists():
        return run_command(f"tail -f {log_file}")
    else:
        print("❌ Log file not found")
        return False

def clean_data():
    """Clean all data directories"""
    print("🧹 Cleaning data directories...")
    
    import shutil
    
    dirs_to_clean = [
        "data/chromadb",
        "data/processed",
        "logs"
    ]
    
    for dir_path in dirs_to_clean:
        if Path(dir_path).exists():
            shutil.rmtree(dir_path)
            print(f"🗑️  Cleaned {dir_path}")
    
    # Recreate directories
    for dir_path in dirs_to_clean:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"📁 Recreated {dir_path}")
    
    return True

def run_development():
    """Run the application in development mode"""
    print("🚀 Starting CosmicSpark in development mode...")
    
    processes = []
    
    try:
        # Start backend
        print("\n🔧 Starting Backend API...")
        backend_proc = start_backend()
        processes.append(backend_proc)
        
        # Wait for backend to start
        print("⏳ Waiting for backend to start...")
        time.sleep(5)
        
        # Open API docs in browser
        print("🌐 Opening API documentation in browser...")
        webbrowser.open("http://localhost:8000/docs")
        
        # Start frontend
        print("\n🎨 Starting Frontend...")
        frontend_proc = start_frontend()
        processes.append(frontend_proc)
        
        print("\n🚀 Development server is running!")
        print("Press Ctrl+C to stop all services")
        
        # Keep the script running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping all services...")
        for proc in processes:
            if proc:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except (subprocess.TimeoutExpired, ProcessLookupError):
                    try:
                        proc.kill()
                    except:
                        pass
        print("✅ All services stopped")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(
        description="CosmicSpark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py setup          # Setup environment
  python run.py dev            # Start development server (backend + frontend)
  python run.py backend        # Start backend only
  python run.py frontend       # Start frontend only
  python run.py crawl          # Run web crawler
  python run.py build-kg       # Build knowledge graph
  python run.py test           # Run tests
  python run.py health         # Check system health
  python run.py logs           # Show logs
  python run.py clean          # Clean data
        """
    )
    
    parser.add_argument(
        "command",
        choices=[
            "setup", "dev", "backend", "frontend",
            "crawl", "build-kg", "test", "health", "logs",
            "clean"
        ],
        help="Command to execute"
    )
    
    args = parser.parse_args()
    
    # Command mapping
    commands = {
        "setup": setup_environment,
        "dev": run_development,
        "backend": start_backend,
        "frontend": start_frontend,
        "crawl": run_crawler,
        "build-kg": build_knowledge_graph,
        "test": run_tests,
        "health": check_health,
        "logs": show_logs,
        "clean": clean_data
    }
    
    # Execute command
    success = commands[args.command]()
    
    if success is not None:  # Some commands like 'dev' might return None
        if success:
            print(f"\n✅ Command '{args.command}' completed successfully!")
            sys.exit(0)
        else:
            print(f"\n❌ Command '{args.command}' failed!")
            sys.exit(1)

if __name__ == "__main__":
    main()
