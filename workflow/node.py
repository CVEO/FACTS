from abc import ABC, abstractmethod
from typing import List

# In demo.py, we defined ImageProcessor. To avoid circular imports and keep the
# components decoupled, we won't import it here. Instead, we rely on the fact
# that the objects passed to the nodes will have the necessary attributes
# (like .img_data, .name, etc.). This is known as duck typing.

# This import is removed to rely on duck typing, matching the intended design.
# from image_processor import ImageProcessor


class BaseNode(ABC):
    """
    An abstract base class for all workflow nodes.

    It defines a common interface for all processing nodes in the workflow.
    Each node is initialized with a list of image objects and must implement
    a `process` method.
    """
    def __init__(self, image_processors: List):
        """
        Initializes the node with a list of image processors.

        Args:
            image_processors: A list of image processor objects (duck-typed)
                              that the node will operate on.
        """
        if not isinstance(image_processors, list):
            raise TypeError("Input must be a list of image processors.")
        self.image_processors = image_processors

    @abstractmethod
    def process(self, *args, **kwargs):
        """
        The main processing method for the node.
        This method must be implemented by all subclasses.
        """
        pass