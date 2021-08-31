from flash.core.registry import FlashRegistry  # noqa: F401
from flash.image.embedding.loss import register_vissl_losses  # noqa: F401
from flash.image.embedding.model import ImageEmbedder  # noqa: F401

IMAGE_EMBEDDER_LOSS_FUNTIONS = FlashRegistry("loss_functions")
register_vissl_losses(IMAGE_EMBEDDER_LOSS_FUNTIONS)
