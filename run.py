#!/usr/bin/env python3
"""
MOSDAC AI Help Bot - Main Runner Script
Provides easy commands to run different components of the system
"""

import argparse
import subprocess
import sys
import os
import time
from pathlib import Path

def run_command(command: str, cwd: str = None):
    """Run a command with proper error handling"""
    print(f"🚀 Running: {command}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            check=True
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
        return False

def setup_environment():
    """Setup the development environment"""
    print("🔧 Setting up MOSDAC AI Help Bot environment...")
    return run_command("python scripts/setup_environment.py")

def start_services():
    """Start all Docker services"""
    print("🐳 Starting Docker services...")
    return run_command("docker-compose up -d")

def stop_services():
    """Stop all Docker services"""
    print("🛑 Stopping Docker services...")
    return run_command("docker-compose down")

def start_backend():
    """Start the FastAPI backend"""
    print("🔧 Starting backend API...")
    return run_command("python -m src.api.main")

def start_frontend():
    """Start the Streamlit frontend"""
    print("🎨 Starting frontend...")
    return run_command("streamlit run frontend/streamlit_app.py")

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
        ("Frontend", "http://localhost:8501"),
        ("Neo4j", "http://localhost:7474"),
        ("Ollama", "http://localhost:11434/api/tags")
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

def full_setup():
    """Complete setup and initialization"""
    print("🚀 Running full MOSDAC AI Help Bot setup...")
    
    steps = [
        ("Environment Setup", setup_environment),
        ("Start Services", start_services),
        ("Wait for Services", lambda: time.sleep(30) or True),
        ("Run Crawler", run_crawler),
        ("Build Knowledge Graph", build_knowledge_graph),
        ("Health Check", check_health)
    ]
    
    for step_name, step_func in steps:
        print(f"\n📋 Step: {step_name}")
        if not step_func():
            print(f"❌ Failed at step: {step_name}")
            return False
        print(f"✅ Completed: {step_name}")
    
    print("\n🎉 Full setup completed successfully!")
    print("\n📍 Access points:")
    print("   Frontend: http://localhost:8501")
    print("   API Docs: http://localhost:8000/docs")
    print("   Neo4j: http://localhost:7474")
    print("   Grafana: http://localhost:3000")
    
    return True

def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(
        description="MOSDAC AI Help Bot Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py setup          # Setup environment
  python run.py start          # Start all services
  python run.py backend        # Start backend only
  python run.py frontend       # Start frontend only
  python run.py crawl          # Run web crawler
  python run.py build-kg       # Build knowledge graph
  python run.py test           # Run tests
  python run.py health         # Check system health
  python run.py logs           # Show logs
  python run.py clean          # Clean data
  python run.py full-setup     # Complete setup
        """
    )
    
    parser.add_argument(
        "command",
        choices=[
            "setup", "start", "stop", "backend", "frontend",
            "crawl", "build-kg", "test", "health", "logs",
            "clean", "full-setup"
        ],
        help="Command to execute"
    )
    
    args = parser.parse_args()
    
    # Command mapping
    commands = {
        "setup": setup_environment,
        "start": start_services,
        "stop": stop_services,
        "backend": start_backend,
        "frontend": start_frontend,
        "crawl": run_crawler,
        "build-kg": build_knowledge_graph,
        "test": run_tests,
        "health": check_health,
        "logs": show_logs,
        "clean": clean_data,
        "full-setup": full_setup
    }
    
    # Execute command
    success = commands[args.command]()
    
    if success:
        print(f"\n✅ Command '{args.command}' completed successfully!")
        sys.exit(0)
    else:
        print(f"\n❌ Command '{args.command}' failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()