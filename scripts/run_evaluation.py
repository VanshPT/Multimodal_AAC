import os
import sys
from pathlib import Path

import django

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "softgo.settings")
django.setup()

from home.aac.evaluation.run_eval import run_evaluation  # noqa: E402


if __name__ == "__main__":
    run_evaluation(user_id="demo_user")
    print("Evaluation completed.")
