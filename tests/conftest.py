import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# tests should never accidentally hit a real cluster
os.environ.setdefault("KSERVE_GATEWAY", "http://test-gateway")
os.environ.setdefault("ROUTER_TIMEOUT_S", "2")
