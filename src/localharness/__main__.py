"""The entry the autostart unit runs: `python -I -m localharness serve --config-dir <dir>`.

`-I` (isolated mode) ignores `PYTHON*` variables and the user site, and `-m localharness` finds this
module, which runs the same CLI the `localharness` console script does."""
from localharness.cli.app import main

if __name__ == "__main__":
    main()
