"""Make a StatsPro Pro key for a customer who paid by bKash.

    LICENSE_SECRET="your-long-secret" python tools/make_key.py --days 7
    python tools/make_key.py --days 30 --secret "your-long-secret"
    python tools/make_key.py --lifetime --secret "..."

The secret must equal the LICENSE_SECRET you saved in Streamlit secrets.
Never commit it or share it.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules import licensing  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=30, help="validity in days (default 30)")
    ap.add_argument("--lifetime", action="store_true", help="key that never expires")
    ap.add_argument("--secret", default=os.environ.get("LICENSE_SECRET", ""))
    a = ap.parse_args()
    try:
        key = licensing.make_key(a.secret, None if a.lifetime else a.days)
    except ValueError as exc:
        sys.exit(f"Error: {exc}")
    print(key)


if __name__ == "__main__":
    main()
