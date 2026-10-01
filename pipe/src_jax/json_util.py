from __future__ import annotations

import json
import pathlib

import numpy as np


def json_convert(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if hasattr(obj, "tolist"):
        return {"$nparray": obj.tolist()}
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def json_deconvert(obj):
    if len(obj) == 1:
        key, value = next(iter(obj.items()))
        if key == "$nparray":
            return np.array(value)
    return obj


def save_json(path, obj):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, default=json_convert), encoding="utf-8")


def load_json(path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"), object_hook=json_deconvert)
