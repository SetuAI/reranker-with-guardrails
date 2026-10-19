"""
Shared setup for the tests.

Every test here runs offline. The guardrail modules create an OpenAI client when
they are imported, which only needs SOME key to exist -- no test ever lets a real
API call through. Anything that would call a model is replaced in the test itself.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")

sys.path.insert(0, str(Path(__file__).parent.parent))
