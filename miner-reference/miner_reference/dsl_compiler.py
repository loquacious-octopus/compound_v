"""Compile low-poly primitive-decomposition DSL into validator-compliant Three.js."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


class DslError(ValueError):
    """Raised when a DSL document cannot be compiled safely."""


_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")
_MATERIAL_TYPES = {"basic", "standard", "physical", "line", "points"}
_PRIMITIVE_KINDS = {"box", "sphere", "cylinder", "cone", "capsule", "torus"}
_PART_KINDS = {"primitive", "lathe", "tube", "line_edges", "instanced"}
_TEXTURE_KINDS = {"stripe", "checker", "speckle"}


def compile_dsl_document(document: dict[str, Any]) -> str:
    """Return a complete ES module for a validated DSL document."""

    spec = _validate_document(document)
    materials = spec["materials"]
    parts = spec["parts"]

    lines: list[str] = [
        "export default function generate(THREE) {",
        "  const root = new THREE.Group();",
        "",
        *(_compile_materials(materials)),
        "",
        *(_compile_parts(parts)),
        "",
        "  fitToUnitCube(THREE, root);",
        "  return root;",
        "}",
        "",
        _helper_source(),
    ]
    return "\n".join(lines)


def compile_dsl_file(input_path: Path, output_path: Path) -> None:
    document = json.loads(input_path.read_text(encoding="utf-8"))
    source = compile_dsl_document(document)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(source, encoding="utf-8")


def _validate_document(document: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise DslError("DSL root must be an object")
    version = document.get("version")
    if version != 1:
        raise DslError("DSL version must be 1")
    materials = document.get("materials")
    parts = document.get("parts")
    if not isinstance(materials, list) or not materials:
        raise DslError("materials must be a non-empty list")
    if not isinstance(parts, list) or not parts:
        raise DslError("parts must be a non-empty list")
    if len(materials) > 32:
        raise DslError("materials exceeds v1 limit of 32")
    if len(parts) > 180:
        raise DslError("parts exceeds validator-safe v1 limit of 180")

    material_ids: set[str] = set()
    for material in materials:
        if not isinstance(material, dict):
            raise DslError("each material must be an object")
        material_id = _required_ident(material, "id")
        if material_id in material_ids:
            raise DslError(f"duplicate material id: {material_id}")
        material_ids.add(material_id)
        material_type = material.get("type", "standard")
        if material_type not in _MATERIAL_TYPES:
            raise DslError(f"unsupported material type: {material_type}")
        _optional_color(material, "color")
        texture = material.get("texture")
        if texture is not None:
            _validate_texture(texture)

    for part in parts:
        if not isinstance(part, dict):
            raise DslError("each part must be an object")
        kind = part.get("kind")
        if kind not in _PART_KINDS:
            raise DslError(f"unsupported part kind: {kind}")
        material = part.get("material")
        if not isinstance(material, str) or material not in material_ids:
            raise DslError(f"part references unknown material: {material}")
        _validate_transform(part)
        if kind == "primitive":
            _validate_primitive(part)
        elif kind == "lathe":
            _validate_lathe(part)
        elif kind == "tube":
            _validate_tube(part)
        elif kind == "line_edges":
            _validate_line_edges(part)
        elif kind == "instanced":
            _validate_instanced(part)

    return document


def _compile_materials(materials: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for material in materials:
        name = f"mat_{material['id']}"
        material_type = material.get("type", "standard")
        params = [_param("color", _hex_to_number(material.get("color", "#cccccc")))]
        if material_type in {"standard", "physical"}:
            params.append(_param("roughness", _number(material.get("roughness", 0.55))))
            params.append(_param("metalness", _number(material.get("metalness", 0.0))))
        if material_type == "physical":
            params.append(_param("clearcoat", _number(material.get("clearcoat", 0.0))))
            params.append(_param("clearcoatRoughness", _number(material.get("clearcoat_roughness", 0.35))))
        texture = material.get("texture")
        if texture:
            texture_name = f"tex_{material['id']}"
            lines.extend(_compile_texture(texture_name, texture))
            params.append(_param("map", texture_name, raw=True))
        if material_type == "line":
            params.append(_param("vertexColors", "true", raw=True))
            lines.append(f"  const {name} = new THREE.LineBasicMaterial({{ {', '.join(params)} }});")
        elif material_type == "points":
            params.append(_param("size", _number(material.get("size", 0.02))))
            params.append(_param("vertexColors", "true", raw=True))
            lines.append(f"  const {name} = new THREE.PointsMaterial({{ {', '.join(params)} }});")
        elif material_type == "basic":
            lines.append(f"  const {name} = new THREE.MeshBasicMaterial({{ {', '.join(params)} }});")
        elif material_type == "physical":
            lines.append(f"  const {name} = new THREE.MeshPhysicalMaterial({{ {', '.join(params)} }});")
        else:
            lines.append(f"  const {name} = new THREE.MeshStandardMaterial({{ {', '.join(params)} }});")
    return lines


def _compile_texture(name: str, texture: dict[str, Any]) -> list[str]:
    kind = texture.get("kind")
    size = _int(texture.get("size", 64), 8, 512)
    color_a = _hex_to_rgb(texture.get("color_a", "#ffffff"))
    color_b = _hex_to_rgb(texture.get("color_b", "#777777"))
    scale = _int(texture.get("scale", 8), 1, 128)
    lines = [
        f"  const {name} = makePatternTexture(THREE, {json.dumps(kind)}, {size}, "
        f"[{color_a[0]}, {color_a[1]}, {color_a[2]}], [{color_b[0]}, {color_b[1]}, {color_b[2]}], {scale});"
    ]
    return lines


def _compile_parts(parts: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for index, part in enumerate(parts):
        kind = part["kind"]
        if kind == "primitive":
            lines.extend(_compile_primitive(index, part))
        elif kind == "lathe":
            lines.extend(_compile_lathe(index, part))
        elif kind == "tube":
            lines.extend(_compile_tube(index, part))
        elif kind == "line_edges":
            lines.extend(_compile_line_edges(index, part))
        elif kind == "instanced":
            lines.extend(_compile_instanced(index, part))
        lines.append("")
    return lines


def _compile_primitive(index: int, part: dict[str, Any]) -> list[str]:
    geo = _geometry_source(part["shape"])
    material = f"mat_{part['material']}"
    lines = [
        f"  const geo_{index} = {geo};",
        f"  const obj_{index} = new THREE.Mesh(geo_{index}, {material});",
        *_compile_transform(index, part),
        f"  root.add(obj_{index});",
    ]
    return lines


def _compile_lathe(index: int, part: dict[str, Any]) -> list[str]:
    points = ", ".join(f"new THREE.Vector2({_number(x)}, {_number(y)})" for x, y in part["profile"])
    segments = _int(part.get("segments", 48), 6, 128)
    phi_start = _number(part.get("phi_start", 0))
    phi_length = _number(part.get("phi_length", 6.283185307179586))
    material = f"mat_{part['material']}"
    return [
        f"  const profile_{index} = [{points}];",
        f"  const geo_{index} = new THREE.LatheGeometry(profile_{index}, {segments}, {phi_start}, {phi_length});",
        f"  const obj_{index} = new THREE.Mesh(geo_{index}, {material});",
        *_compile_transform(index, part),
        f"  root.add(obj_{index});",
    ]


def _compile_tube(index: int, part: dict[str, Any]) -> list[str]:
    points = ", ".join(f"new THREE.Vector3({_number(x)}, {_number(y)}, {_number(z)})" for x, y, z in part["points"])
    tubular_segments = _int(part.get("tubular_segments", 48), 2, 256)
    radius = _number(part.get("radius", 0.02))
    radial_segments = _int(part.get("radial_segments", 8), 3, 32)
    closed = "true" if part.get("closed", False) else "false"
    material = f"mat_{part['material']}"
    return [
        f"  const curve_{index} = new THREE.CatmullRomCurve3([{points}], {closed});",
        f"  const geo_{index} = new THREE.TubeGeometry(curve_{index}, {tubular_segments}, {radius}, {radial_segments}, {closed});",
        f"  const obj_{index} = new THREE.Mesh(geo_{index}, {material});",
        *_compile_transform(index, part),
        f"  root.add(obj_{index});",
    ]


def _compile_line_edges(index: int, part: dict[str, Any]) -> list[str]:
    geo = _geometry_source(part["shape"])
    material = f"mat_{part['material']}"
    return [
        f"  const baseGeo_{index} = {geo};",
        f"  const geo_{index} = new THREE.EdgesGeometry(baseGeo_{index});",
        f"  const obj_{index} = new THREE.LineSegments(geo_{index}, {material});",
        *_compile_transform(index, part),
        f"  root.add(obj_{index});",
    ]


def _compile_instanced(index: int, part: dict[str, Any]) -> list[str]:
    geo = _geometry_source(part["shape"])
    material = f"mat_{part['material']}"
    count = _int(part["count"], 1, 50000)
    pattern = part.get("pattern", {})
    pattern_kind = pattern.get("kind", "line")
    lines = [
        f"  const geo_{index} = {geo};",
        f"  const obj_{index} = new THREE.InstancedMesh(geo_{index}, {material}, {count});",
        "  const dummy = new THREE.Object3D();",
        f"  for (let i = 0; i < {count}; i++) {{",
    ]
    if pattern_kind == "ring":
        radius = _number(pattern.get("radius", 0.25))
        y = _number(pattern.get("y", 0))
        lines.extend(
            [
                f"    const a = (i / {count}) * 6.283185307179586;",
                f"    dummy.position.set(Math.cos(a) * {radius}, {y}, Math.sin(a) * {radius});",
                "    dummy.rotation.set(0, -a, 0);",
            ]
        )
    elif pattern_kind == "grid":
        cols = min(_int(pattern.get("columns", max(1, int(count**0.5))), 1, 512), count)
        spacing = _number(pattern.get("spacing", 0.08))
        lines.extend(
            [
                f"    const x = (i % {cols}) - ({cols} - 1) / 2;",
                f"    const z = Math.floor(i / {cols}) - (Math.ceil({count} / {cols}) - 1) / 2;",
                f"    dummy.position.set(x * {spacing}, 0, z * {spacing});",
                "    dummy.rotation.set(0, 0, 0);",
            ]
        )
    else:
        spacing = _number(pattern.get("spacing", 0.05))
        lines.extend(
            [
                f"    const x = (i - ({count} - 1) / 2) * {spacing};",
                "    dummy.position.set(x, 0, 0);",
                "    dummy.rotation.set(0, 0, 0);",
            ]
        )
    scale = _vec3(part.get("instance_scale", part.get("scale", [1, 1, 1])))
    lines.extend(
        [
            f"    dummy.scale.set({_number(scale[0])}, {_number(scale[1])}, {_number(scale[2])});",
            "    dummy.updateMatrix();",
            f"    obj_{index}.setMatrixAt(i, dummy.matrix);",
            "  }",
            *_compile_transform(index, {**part, "scale": [1, 1, 1]}),
            f"  root.add(obj_{index});",
        ]
    )
    return lines


def _compile_transform(index: int, part: dict[str, Any]) -> list[str]:
    position = _vec3(part.get("position", [0, 0, 0]))
    rotation = _vec3(part.get("rotation", [0, 0, 0]))
    scale = _vec3(part.get("scale", [1, 1, 1]))
    return [
        f"  obj_{index}.position.set({_number(position[0])}, {_number(position[1])}, {_number(position[2])});",
        f"  obj_{index}.rotation.set({_number(rotation[0])}, {_number(rotation[1])}, {_number(rotation[2])});",
        f"  obj_{index}.scale.set({_number(scale[0])}, {_number(scale[1])}, {_number(scale[2])});",
    ]


def _geometry_source(shape: dict[str, Any]) -> str:
    kind = shape["kind"]
    if kind == "box":
        size = _vec3(shape.get("size", [1, 1, 1]))
        return f"new THREE.BoxGeometry({_number(size[0])}, {_number(size[1])}, {_number(size[2])})"
    if kind == "sphere":
        return (
            f"new THREE.SphereGeometry({_number(shape.get('radius', 0.5))}, "
            f"{_int(shape.get('width_segments', 32), 3, 128)}, {_int(shape.get('height_segments', 16), 2, 64)})"
        )
    if kind == "cylinder":
        return (
            f"new THREE.CylinderGeometry({_number(shape.get('radius_top', 0.5))}, "
            f"{_number(shape.get('radius_bottom', 0.5))}, {_number(shape.get('height', 1))}, "
            f"{_int(shape.get('radial_segments', 32), 3, 128)})"
        )
    if kind == "cone":
        return (
            f"new THREE.ConeGeometry({_number(shape.get('radius', 0.5))}, {_number(shape.get('height', 1))}, "
            f"{_int(shape.get('radial_segments', 32), 3, 128)})"
        )
    if kind == "capsule":
        return (
            f"new THREE.CapsuleGeometry({_number(shape.get('radius', 0.2))}, {_number(shape.get('length', 0.5))}, "
            f"{_int(shape.get('cap_segments', 8), 1, 32)}, {_int(shape.get('radial_segments', 16), 3, 64)})"
        )
    if kind == "torus":
        return (
            f"new THREE.TorusGeometry({_number(shape.get('radius', 0.3))}, {_number(shape.get('tube', 0.05))}, "
            f"{_int(shape.get('radial_segments', 12), 3, 64)}, {_int(shape.get('tubular_segments', 48), 3, 192)})"
        )
    raise DslError(f"unsupported primitive shape: {kind}")


def _helper_source() -> str:
    return """function makePatternTexture(THREE, kind, size, colorA, colorB, scale) {
  const data = new Uint8Array(size * size * 4);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = (y * size + x) * 4;
      let useA = false;
      if (kind === "checker") {
        useA = (Math.floor(x / scale) + Math.floor(y / scale)) % 2 === 0;
      } else if (kind === "speckle") {
        useA = ((x * 17 + y * 31 + x * y * 7) % Math.max(3, scale)) === 0;
      } else {
        useA = Math.floor(x / scale) % 2 === 0;
      }
      const c = useA ? colorA : colorB;
      data[i] = c[0];
      data[i + 1] = c[1];
      data[i + 2] = c[2];
      data[i + 3] = 255;
    }
  }
  const tex = new THREE.DataTexture(data, size, size, THREE.RGBAFormat);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.needsUpdate = true;
  return tex;
}

function fitToUnitCube(THREE, root) {
  const box = new THREE.Box3().setFromObject(root);
  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  box.getSize(size);
  box.getCenter(center);
  const maxDim = Math.max(size.x, size.y, size.z);
  if (maxDim > 0) {
    const scale = 0.94 / maxDim;
    root.scale.setScalar(scale);
    root.position.set(-center.x * scale, -center.y * scale, -center.z * scale);
  }
}
"""


def _validate_texture(texture: Any) -> None:
    if not isinstance(texture, dict):
        raise DslError("texture must be an object")
    if texture.get("kind") not in _TEXTURE_KINDS:
        raise DslError(f"unsupported texture kind: {texture.get('kind')}")
    _int(texture.get("size", 64), 8, 512)
    _int(texture.get("scale", 8), 1, 128)
    _optional_color(texture, "color_a")
    _optional_color(texture, "color_b")


def _validate_transform(part: dict[str, Any]) -> None:
    for key in ("position", "rotation", "scale"):
        if key in part:
            _vec3(part[key])


def _validate_primitive(part: dict[str, Any]) -> None:
    shape = part.get("shape")
    if not isinstance(shape, dict) or shape.get("kind") not in _PRIMITIVE_KINDS:
        raise DslError(f"primitive has unsupported shape: {shape}")


def _validate_lathe(part: dict[str, Any]) -> None:
    profile = part.get("profile")
    if not isinstance(profile, list) or len(profile) < 2 or len(profile) > 64:
        raise DslError("lathe profile must have 2-64 points")
    for point in profile:
        _vec2(point)
    _int(part.get("segments", 48), 6, 128)


def _validate_tube(part: dict[str, Any]) -> None:
    points = part.get("points")
    if not isinstance(points, list) or len(points) < 2 or len(points) > 64:
        raise DslError("tube points must have 2-64 points")
    for point in points:
        _vec3(point)
    _int(part.get("tubular_segments", 48), 2, 256)
    _int(part.get("radial_segments", 8), 3, 32)


def _validate_line_edges(part: dict[str, Any]) -> None:
    _validate_primitive(part)


def _validate_instanced(part: dict[str, Any]) -> None:
    _validate_primitive(part)
    _int(part.get("count"), 1, 50000)
    pattern = part.get("pattern", {})
    if not isinstance(pattern, dict):
        raise DslError("instanced pattern must be an object")
    if pattern.get("kind", "line") not in {"line", "ring", "grid"}:
        raise DslError(f"unsupported instancing pattern: {pattern.get('kind')}")


def _required_ident(obj: dict[str, Any], key: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not _IDENT_RE.fullmatch(value):
        raise DslError(f"{key} must be a safe identifier")
    return value


def _optional_color(obj: dict[str, Any], key: str) -> None:
    if key in obj and (not isinstance(obj[key], str) or not _HEX_RE.fullmatch(obj[key])):
        raise DslError(f"{key} must be #RRGGBB")


def _hex_to_number(value: str) -> str:
    if not _HEX_RE.fullmatch(value):
        raise DslError(f"invalid color: {value}")
    return "0x" + value[1:]


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    if not _HEX_RE.fullmatch(value):
        raise DslError(f"invalid color: {value}")
    return int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16)


def _param(key: str, value: str, *, raw: bool = False) -> str:
    return f"{key}: {value if raw else value}"


def _vec2(value: Any) -> tuple[float, float]:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise DslError("expected [x, y]")
    return float(value[0]), float(value[1])


def _vec3(value: Any) -> tuple[float, float, float]:
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise DslError("expected [x, y, z]")
    return float(value[0]), float(value[1]), float(value[2])


def _number(value: Any) -> str:
    number = float(value)
    if not -10000 <= number <= 10000:
        raise DslError(f"number out of safe range: {value}")
    text = f"{number:.6f}".rstrip("0").rstrip(".")
    if text == "-0":
        return "0"
    return text or "0"


def _int(value: Any, minimum: int, maximum: int) -> int:
    if not isinstance(value, int):
        raise DslError(f"expected integer, got {value!r}")
    if value < minimum or value > maximum:
        raise DslError(f"integer {value} outside [{minimum}, {maximum}]")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compile a procedural object DSL JSON file to Three.js.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    compile_dsl_file(args.input, args.out)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
