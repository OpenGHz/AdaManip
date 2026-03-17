import argparse
import json
import numpy as np
from pathlib import Path
from try_to_remember.sentence import StagesToSentenceEmbedding


def _load_language_expanded(input_path: Path) -> dict:
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "expanded_minimal_chains" not in data:
        raise ValueError(
            "Missing key 'expanded_minimal_chains' in language_expanded.json"
        )
    if "operation_set" not in data:
        raise ValueError("Missing key 'operation_set' in language_expanded.json")

    chains = data["expanded_minimal_chains"]
    operations = data["operation_set"]

    if not isinstance(chains, list):
        raise TypeError("'expanded_minimal_chains' must be a list")

    if isinstance(operations, list):
        operation_list = operations
    elif isinstance(operations, dict):
        operation_list = list(operations.keys())
    else:
        raise TypeError("'operation_set' must be a list or dict")

    for i, chain in enumerate(chains):
        if not isinstance(chain, list):
            raise TypeError(
                f"expanded_minimal_chains[{i}] must be a list of operations"
            )
        if not all(isinstance(step, str) for step in chain):
            raise TypeError(
                f"expanded_minimal_chains[{i}] contains non-string operation"
            )

    if not all(isinstance(op, str) for op in operation_list):
        raise TypeError("operation_set contains non-string operation")

    return {
        "chains": chains,
        "operations": operation_list,
    }


def _encode_texts(
    embedder: StagesToSentenceEmbedding,
    items: list,
    normalize: bool,
) -> np.ndarray:
    if not items:
        return np.zeros((0, 0), dtype=np.float32)
    vectors = []
    for item in items:
        vec = embedder.encode(item)
        vectors.append(np.asarray(vec, dtype=np.float32))

    embeddings = np.stack(vectors, axis=0).astype(np.float32)

    if normalize and embeddings.size > 0:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-12, a_max=None)
        embeddings = embeddings / norms

    return embeddings


def _save_embedding_dict(
    output_path: Path,
    model_name: str,
    normalized: bool,
    chain_embeddings: np.ndarray,
    operations: list[str],
    operation_embeddings: np.ndarray,
) -> None:
    if chain_embeddings.ndim == 2 and chain_embeddings.shape[0] > 0:
        embedding_dim = int(chain_embeddings.shape[1])
    elif operation_embeddings.ndim == 2 and operation_embeddings.shape[0] > 0:
        embedding_dim = int(operation_embeddings.shape[1])
    else:
        embedding_dim = 0

    payload = {
        "encoder": {
            "name": model_name,
            "output_dim": embedding_dim,
            "normalized": bool(normalized),
        },
        "expanded_minimal_chains": chain_embeddings.tolist(),
        "operation_set": {
            op: emb.tolist() for op, emb in zip(operations, operation_embeddings)
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess language labels into a single embedding dict with encoder metadata."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to language_expanded.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for language_embedding_dict.json (default: sibling of input)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="moka-ai/m3e-small",
        choices=["moka-ai/m3e-small", "sentence-transformers/all-MiniLM-L6-v2"],
        help="Sentence embedding model name",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Model device, e.g. cuda or cpu",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="L2 normalize embeddings",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow downloading model weights when cache miss",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = args.input.resolve()
    if input_path.is_dir():
        input_path = input_path / "language_expanded.json"
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if args.output is None:
        output_path = input_path.parent / "language_embedding_dict.json"
    else:
        output_path = args.output.resolve()

    model_name = args.model
    normalized = model_name in {"sentence-transformers/all-MiniLM-L6-v2"}
    normalize = args.normalize if not normalized else False

    data = _load_language_expanded(input_path)
    chains = data["chains"]
    operations = data["operations"]

    embedder = StagesToSentenceEmbedding(
        model_name=model_name,
        device=args.device,
        local_files_only=not args.allow_download,
    )

    chain_embeddings = _encode_texts(
        embedder=embedder,
        items=chains,
        normalize=normalize,
    )
    operation_embeddings = _encode_texts(
        embedder=embedder,
        items=operations,
        normalize=normalize,
    )

    _save_embedding_dict(
        output_path=output_path,
        model_name=model_name,
        normalized=normalized or normalize,
        chain_embeddings=chain_embeddings,
        operations=operations,
        operation_embeddings=operation_embeddings,
    )

    print(f"Saved embedding dict: {output_path}")
    print(f"chains: {len(chains)}, operations: {len(operations)}")


if __name__ == "__main__":
    main()
