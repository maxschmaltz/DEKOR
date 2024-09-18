from dekor.embeddings.torch_embeddings import TorchEmbeddings
from dekor.embeddings.flair_embeddings import FlairEmbeddings


__all__ = {
	TorchEmbeddings.name: TorchEmbeddings,
	FlairEmbeddings.name: FlairEmbeddings
}