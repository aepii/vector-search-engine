# Distributed Vector Search Engine

A *distributed system* for storing and querying [vector embeddings](https://www.pinecone.io/learn/vector-database/). This project implements a partitioned vector index utilizing a coordinator-worker architecture.

## Architecture

```text
Client
  └─► Coordinator (port 50050)  — hash-routes writes, fans out reads
          ├─► Shard 0 (port 50051)
          ├─► Shard 1 (port 50052)
          └─► Shard 2 (port 50053)
```

Writes (`Upsert`, `UpsertBatch`) are routed via a consistent hash ring (SHA-256, 150 virtual nodes per host) to the next N clockwise physical nodes, where N is configurable via `REPLICATION_FACTOR` (default `0` = all nodes). At small scale this gives full replication — every shard holds every vector. Reads (`Search`) broadcast to all shards in parallel, merge results by score, and deduplicate before returning.

Shards self-register by sending periodic heartbeats to the coordinator. The coordinator tracks liveness via a background sweep and automatically removes shards that stop responding. Startup order does not matter — shards can come up before or after the coordinator and will register on their first successful heartbeat.

The coordinator also exposes a `CoordinatorControl` service on port 50050 for manual registration and deregistration if needed:

```bash
# Example: manually register or deregister a shard (via grpcurl or a client script)
RegisterNode   { host: "localhost:50054" }
DeregisterNode { host: "localhost:50054" }
```

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/aepii/vector-search-engine.git
cd vector-search-engine
```

### 2. Setup a virtual environment

```bash
# Create the environment
python -m venv .venv

# Activate on Windows
.venv\Scripts\activate

# Activate on macOS/Linux
source .venv/bin/activate
```

All subsequent commands assume the virtual environment is active.

### 3.  Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Compile the Protobuf Definitions

```bash
python -m grpc_tools.protoc \
    -I protos \
    --python_out=src \
    --grpc_python_out=src \
    protos/vector_store.proto
```

### 5. Configure Environment

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

## Running Locally

Start each component in separate terminals.

### 1. Navigate to the source directory

```bash
cd src
```

### 2. Start the Coordinator

```bash
python -m coordinator
```

### 3. Start Server Nodes (Shards)

Run one server per terminal. Each shard writes its vector DB to `data/shard_{PORT}.db`
by default — the directory is created automatically on first startup.

```bash
$env:SERVER_PORT=50051; python -m server
```

```bash
$env:SERVER_PORT=50052; python -m server
```

```bash
$env:SERVER_PORT=50053; python -m server
```

Override the DB location with `DB_PATH` if needed (e.g. `$env:DB_PATH="/data/shard.db"`).

### 4. Run Benchmark (after all services are ready)

```bash
python -m benchmarks.benchmark
```
