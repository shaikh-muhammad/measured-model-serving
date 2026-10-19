"""Development-only real renderer checks; no model files or benchmark execution."""
import importlib.util
import itertools
import json
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("portfolio_report",ROOT/"bench/report.py")
report=importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def test_development_graphics_determinism_and_real_text_bounds(tmp_path):
    pytest.importorskip("matplotlib",reason="Install requirements-dev.lock for real graphics validation")
    evidence=report.load_evidence()
    data=report.portfolio_data(evidence)
    first,geometry=report.render_graphics(data)
    second,repeated_geometry=report.render_graphics(data)
    assert first==second
    assert geometry==repeated_geometry
    for name,raw in first.items():
        assert raw==(ROOT/name).read_bytes(),name
        (tmp_path/Path(name).name).write_bytes(raw)
        boxes=geometry[name]
        assert [box["text"] for box in boxes]==[label["text"] for label in data["graphics"][name]["texts"]]
        assert len({box["id"] for box in boxes})==len(boxes)
        width,height=data["graphics"][name]["size"]
        for box in boxes:
            left,top,right,bottom=box["bbox"]
            x0,y0,x1,y1=box["region"]
            assert x0<=left<=right<=x1 and y0<=top<=bottom<=y1,(name,box)
            assert 0<=left<=right<=width and 0<=top<=bottom<=height,(name,box)
            assert len(box["font_sha256"])==64
        for a,b in itertools.combinations(boxes,2):
            a0,a1,a2,a3=a["bbox"];b0,b1,b2,b3=b["bbox"]
            assert min(a2,b2)<=max(a0,b0) or min(a3,b3)<=max(a1,b1),(name,a,b)
        print(name,":",len(boxes),"real text boxes; no collisions, overflow or edge clipping; deterministic PNG")
    (tmp_path/"measured-text-geometry.json").write_text(json.dumps(geometry,indent=2))
