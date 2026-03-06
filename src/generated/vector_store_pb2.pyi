from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class UpsertRequest(_message.Message):
    __slots__ = ("id", "text")
    ID_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    id: int
    text: str
    def __init__(self, id: _Optional[int] = ..., text: _Optional[str] = ...) -> None: ...

class UpsertResponse(_message.Message):
    __slots__ = ("upsert_status",)
    UPSERT_STATUS_FIELD_NUMBER: _ClassVar[int]
    upsert_status: str
    def __init__(self, upsert_status: _Optional[str] = ...) -> None: ...

class SearchRequest(_message.Message):
    __slots__ = ("query_text", "top_k")
    QUERY_TEXT_FIELD_NUMBER: _ClassVar[int]
    TOP_K_FIELD_NUMBER: _ClassVar[int]
    query_text: str
    top_k: int
    def __init__(self, query_text: _Optional[str] = ..., top_k: _Optional[int] = ...) -> None: ...

class SearchResponse(_message.Message):
    __slots__ = ("results",)
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    results: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, results: _Optional[_Iterable[str]] = ...) -> None: ...
