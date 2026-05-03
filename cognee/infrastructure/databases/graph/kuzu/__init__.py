"""Legacy Kuzu import compatibility for the Ladybug graph backend."""

import importlib
import sys

import ladybug


sys.modules.setdefault("kuzu", ladybug)
sys.modules.setdefault("kuzu.database", importlib.import_module("ladybug.database"))
