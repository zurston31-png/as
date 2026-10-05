"""Shared Pydantic base.

Why this exists: we want `extra="forbid"` so a typo in a field name raises
instead of being silently ignored on a 45-field trade record. But that makes
`model_dump()` output un-revalidatable, because a dump includes computed
fields (`TradeRecord.r_multiple`, `CostBreakdown.total`,
`ComponentScore.points`), which are not constructor inputs. The database and
report layers need both directions, so the base provides:

  * `model_dump()`    -- full dump including computed fields; for reports and
                         DB writes, where derived columns are wanted.
  * `to_init_dict()`  -- constructor inputs only, recursively; re-validatable.
  * `from_mapping()`  -- tolerant construction that recursively drops computed
                         and foreign keys; for reading rows back out of SQLite
                         or JSON.
  * `replace()`       -- validated copy-with-changes (`model_copy` skips
                         validation, which would let an invalid record exist).

The recursion matters: a flat implementation passes `TradeRecord` and then
fails on the nested `CostBreakdown`.
"""

from __future__ import annotations

import types
from functools import lru_cache
from typing import Any, Mapping, Self, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict

_CONTAINER_ORIGINS = (list, tuple, set, frozenset)


def _model_in(annotation: Any) -> type["FrozenModel"] | None:
    """Return the FrozenModel subclass directly named by `annotation`, if any."""
    if isinstance(annotation, type) and issubclass(annotation, FrozenModel):
        return annotation
    return None


@lru_cache(maxsize=None)
def _exclude_spec(cls: type["FrozenModel"]) -> frozenset | tuple:
    """Recursive pydantic `exclude` spec covering every computed field.

    Returned as hashable data (for caching) and converted to the dict/set
    form pydantic expects by `_as_exclude`.
    """
    items: list[tuple[str, Any]] = [(k, True) for k in cls.model_computed_fields]
    for name, field in cls.model_fields.items():
        sub = _spec_for_annotation(field.annotation)
        if sub:
            items.append((name, sub))
    return tuple(sorted(items, key=lambda kv: kv[0]))


def _spec_for_annotation(annotation: Any) -> Any:
    """Exclude spec for one annotation, or None when nothing to exclude."""
    model = _model_in(annotation)
    if model is not None:
        spec = _exclude_spec(model)
        return spec or None

    origin = get_origin(annotation)
    if origin is None:
        return None
    args = get_args(annotation)

    if origin in (Union, types.UnionType):
        for arg in args:
            if arg is type(None):
                continue
            sub = _spec_for_annotation(arg)
            if sub:
                return sub
        return None

    if origin in _CONTAINER_ORIGINS:
        for arg in args:
            if arg is Ellipsis:
                continue
            sub = _spec_for_annotation(arg)
            if sub:
                return (("__all__", sub),)
        return None

    if origin is dict and len(args) == 2:
        sub = _spec_for_annotation(args[1])
        return (("__all__", sub),) if sub else None

    return None


def _as_exclude(spec: Any) -> Any:
    """Convert the hashable spec into pydantic's exclude structure."""
    if spec is True or spec is None:
        return spec
    return {key: (True if val is True else _as_exclude(val)) for key, val in spec}


def _normalize_exclude(spec: Any) -> dict[str, Any] | bool:
    """Normalize a pydantic exclude argument (set | dict | None) to dict form."""
    if spec is True:
        return True
    if spec is None:
        return {}
    if isinstance(spec, Mapping):
        return {str(k): _normalize_exclude(v) for k, v in spec.items()}
    if isinstance(spec, (set, frozenset, list, tuple)):
        return {str(k): True for k in spec}
    raise TypeError(f"unsupported exclude spec: {spec!r}")


def _merge_exclude(left: Any, right: Any) -> dict[str, Any] | bool:
    """Union of two exclude specs.

    A caller passing `exclude={"name"}` must not lose the automatic
    computed-field exclusion -- that would make `model_dump` emit a computed
    field that re-enters `to_init_dict` (an infinite recursion when a
    computed field is itself defined in terms of `to_init_dict`).
    """
    a, b = _normalize_exclude(left), _normalize_exclude(right)
    if a is True or b is True:
        return True
    merged: dict[str, Any] = dict(a)
    for key, value in b.items():
        merged[key] = _merge_exclude(merged[key], value) if key in merged else value
    return merged


def _clean_value(annotation: Any, value: Any) -> Any:
    """Recursively strip computed/foreign keys from a deserialized value."""
    model = _model_in(annotation)
    if model is not None:
        if isinstance(value, Mapping):
            return model.from_mapping(value)
        return value

    origin = get_origin(annotation)
    if origin is None:
        return value
    args = get_args(annotation)

    if origin in (Union, types.UnionType):
        if value is None:
            return None
        for arg in args:
            if arg is type(None):
                continue
            cleaned = _clean_value(arg, value)
            if cleaned is not value:
                return cleaned
        return value

    if origin in _CONTAINER_ORIGINS and isinstance(value, (list, tuple, set, frozenset)):
        elem = next((a for a in args if a is not Ellipsis), None)
        if elem is None:
            return value
        return [_clean_value(elem, v) for v in value]

    if origin is dict and len(args) == 2 and isinstance(value, Mapping):
        return {k: _clean_value(args[1], v) for k, v in value.items()}

    return value


class FrozenModel(BaseModel):
    """Immutable, strict-on-construction, round-trippable."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    @classmethod
    def computed_keys(cls) -> frozenset[str]:
        return frozenset(cls.model_computed_fields)

    @classmethod
    def input_keys(cls) -> frozenset[str]:
        return frozenset(cls.model_fields)

    @classmethod
    def _init_exclude(cls, caller_exclude: Any = None) -> dict[str, Any] | bool | None:
        """Computed-field exclusions, unioned with whatever the caller asked for."""
        auto = _as_exclude(_exclude_spec(cls))
        if not auto and not caller_exclude:
            return None
        return _merge_exclude(auto or {}, caller_exclude)

    def to_init_dict(self, **kwargs: Any) -> dict[str, Any]:
        """Dump restricted to constructor inputs, recursively.

        A caller-supplied `exclude` is unioned with the computed-field
        exclusions rather than replacing them.
        """
        exclude = self._init_exclude(kwargs.pop("exclude", None))
        if exclude:
            kwargs["exclude"] = exclude
        return self.model_dump(**kwargs)

    def to_init_json(self, **kwargs: Any) -> str:
        exclude = self._init_exclude(kwargs.pop("exclude", None))
        if exclude:
            kwargs["exclude"] = exclude
        return self.model_dump_json(**kwargs)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> Self:
        """Construct from a mapping that may carry computed or foreign keys.

        Use for deserializing database rows and JSON reports. Unknown keys
        are dropped rather than raising, because a stored row legitimately
        contains derived columns. Direct construction still forbids extras,
        so typos in code are still caught.
        """
        fields = cls.model_fields
        kwargs = {
            key: _clean_value(fields[key].annotation, value)
            for key, value in data.items()
            if key in fields
        }
        return cls(**kwargs)

    def replace(self, **changes: Any) -> Self:
        """Validated copy-with-changes."""
        return type(self)(**{**self.to_init_dict(), **changes})
