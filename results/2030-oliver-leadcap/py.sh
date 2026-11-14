#!/bin/bash
exec uv run -q --no-project --python 3.10 --with numpy==2.2.6 --with opencv-python-headless==4.10.0.84 --with pin==3.3.1 --with pyyaml --with scipy --with pandas --with pyarrow --with msgpack python "$@"
