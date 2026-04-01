# Distributed Vector Search Engine

A *distributed system* for storing and querying [vector embeddings](https://www.pinecone.io/learn/vector-database/). This project implements a partitioned vector index utilizing a coordinator-worker architecture.

## Setup and Installation
### 1. Clone the repository:

```bash
git clone https://github.com/aepii/raft-vector-search-engine.git
cd raft-vector-search-engine
```

### 2. Setup a virtual environment:

```bash
# Create the environment
python -m venv .venv

# Activate on Windows
.venv\Scripts\activate

# Activate on macOS/Linux
source .venv/bin/activate
```

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
Open separate terminals for each component:
```bash
# Terminal 1 — Server
python src/server.py
 
# Terminal 2 — Client
python src/client.py
```
