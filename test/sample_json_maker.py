import json
import numpy as np


def write_json(data, fname):
    def _conv(o):
        if isinstance(o, (np.int64, np.int32)):
            return int(o)
        raise TypeError

    with open(fname, "w", encoding="utf-8") as f:
        json_str = json.dumps(data, ensure_ascii=False, default=_conv)
        f.write(json_str)

    return f


def load_json(fname):
    with open(fname, encoding="utf-8-sig") as f:
        json_obj = json.load(f)

    return json_obj, f
