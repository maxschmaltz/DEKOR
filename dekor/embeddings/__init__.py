from dekor.embeddings.torch_embeddings import TorchEmbeddings
from dekor.embeddings.flair_embeddings import FlairEmbeddings


__all_embeddings__ = {
	TorchEmbeddings.name: TorchEmbeddings,
	FlairEmbeddings.name: FlairEmbeddings
}

__all__ = list(__all_embeddings__.values())