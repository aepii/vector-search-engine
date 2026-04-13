"""
Tests for CoordinatorServicer write fan-out.

The coordinator talks to live gRPC shards. Tests replace the real stubs
with MagicMock objects so no network connections are needed.

    patch("coordinator.X")    — intercepts X as seen inside coordinator.py
    MagicMock()               — stand-in object that records every call made to it
    mock.assert_called_once() — fails if the mock wasn't called exactly once

Run from project root:
    .venv/Scripts/python -m pytest src/tests/test_coordinator.py -v
"""
import numpy as np
import grpc
from unittest.mock import MagicMock, patch
import vector_store_pb2
from coordinator import CoordinatorServicer


class _FakeRpcError(grpc.RpcError):
    """Minimal gRPC error with the details() method the coordinator logs."""
    def details(self):
        return "simulated shard failure"


def _make_coordinator(n_shards=2):
    """
    Return a CoordinatorServicer wired to n_shards mock stubs.

    EmbeddingModel and gRPC channels are patched out during construction;
    the mock stubs persist in _stub_map for the duration of each test.
    """
    hosts = [f"fake:{5000 + i}" for i in range(n_shards)]
    mock_stubs = [MagicMock() for _ in hosts]

    def _encode(x):
        # Coordinator calls encode(string) for single items and encode(list) for batches.
        n = len(x) if isinstance(x, list) else 1
        return np.zeros((n, 384)) if n > 1 else np.zeros(384)

    with patch("coordinator.EmbeddingModel") as MockEmbed, \
         patch("coordinator.grpc.insecure_channel"), \
         patch("coordinator.vector_store_pb2_grpc.VectorStoreStub", side_effect=mock_stubs):
        MockEmbed.return_value.encode.side_effect = _encode
        coord = CoordinatorServicer(hosts)

    return coord, mock_stubs


def test_upsert_writes_to_all_shards():
    """Upsert must call every registered shard, not just one."""
    coord, stubs = _make_coordinator(n_shards=3)
    for stub in stubs:
        stub.Upsert.return_value = vector_store_pb2.UpsertResponse(status="ok")

    coord.Upsert(
        vector_store_pb2.UpsertRequest(
            trace_id="t1",
            item=vector_store_pb2.UpsertItem(id=1, text="hello", embedding=[]),
        ),
        context=MagicMock(),
    )

    for stub in stubs:
        stub.Upsert.assert_called_once()


def test_upsert_batch_sends_full_batch_to_all_shards():
    """UpsertBatch must send the full item list to every shard (not a partition)."""
    coord, stubs = _make_coordinator(n_shards=3)
    for stub in stubs:
        stub.UpsertBatch.return_value = vector_store_pb2.UpsertBatchResponse(statuses=["ok", "ok"])

    coord.UpsertBatch(
        vector_store_pb2.UpsertBatchRequest(
            trace_id="t2",
            items=[
                vector_store_pb2.UpsertItem(id=1, text="a", embedding=[]),
                vector_store_pb2.UpsertItem(id=2, text="b", embedding=[]),
            ],
        ),
        context=MagicMock(),
    )

    for stub in stubs:
        stub.UpsertBatch.assert_called_once()
        sent_items = stub.UpsertBatch.call_args[0][0].items
        assert len(sent_items) == 2


def test_upsert_tolerates_shard_failure():
    """A shard RPC error must not raise; the coordinator returns a response regardless."""
    coord, stubs = _make_coordinator(n_shards=2)
    stubs[0].Upsert.side_effect = _FakeRpcError()
    stubs[1].Upsert.return_value = vector_store_pb2.UpsertResponse(status="ok")

    response = coord.Upsert(
        vector_store_pb2.UpsertRequest(
            trace_id="t3",
            item=vector_store_pb2.UpsertItem(id=1, text="hello", embedding=[]),
        ),
        context=MagicMock(),
    )
    assert response is not None


def test_search_fans_out_to_all_shards():
    """Regression guard: Search must still query every shard."""
    coord, stubs = _make_coordinator(n_shards=3)
    for stub in stubs:
        stub.Search.return_value = vector_store_pb2.SearchResponse(results=[])

    coord.Search(
        vector_store_pb2.SearchRequest(
            trace_id="t4",
            query_text="test query",
            query_vector=[0.0] * 384,
            top_k=3,
        ),
        context=MagicMock(),
    )

    for stub in stubs:
        stub.Search.assert_called_once()
