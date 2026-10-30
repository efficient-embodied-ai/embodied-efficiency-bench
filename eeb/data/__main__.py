"""Allow ``python -m eeb.data download|pack-traces|inspect|perturb ...`` without the umbrella CLI."""

import sys

from eeb.data.cli import main

if __name__ == "__main__":
  sys.exit(main())
