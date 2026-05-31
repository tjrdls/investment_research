"""pytest 설정 — repo 루트를 sys.path 에 넣어 `import src.*` 가 항상 동작하게 한다."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
