from typing import Any, Callable, Type, Union, List, Dict, Set, Tuple
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


R = torch.Tensor # Type for encoded/embedding tensors
T = Any # Type for data to encode

class NNEngine:

    dimensions: int

    _encode_str_models: dict[str, Callable]
    _if_models: dict[str, Callable]

    def __init__(self, dimensions: int):
        self.dimensions = dimensions
        self._encode_str_models = {}
        self._if_models = {}

    def if_op(
        self,
        /,
        condition: Any,
        value: Any,
        *,
        true_fn: Callable,
        false_fn: Callable,
        decision_threshold: float = 0.5,
        activation_fn: Callable = nn.ReLU,
    ):
        enc_value = self.encode(value)
        enc_condition = self.encode(condition)

        if_model_name = f"mlp-{self.dimensions}-{activation_fn.__name__}"
        if if_model_name not in self._if_models:
            # Define MLP inline
            self._if_models[if_model_name] = nn.Sequential(
                nn.Linear(2 * self.dimensions, self.dimensions),
                activation_fn(),
                nn.Linear(self.dimensions, self.dimensions // 2),
                activation_fn(),
                nn.Linear(self.dimensions // 2, 1),
                nn.Sigmoid()
            )

        combined_input = torch.cat((enc_condition, enc_value))
        enc_decision = self._if_models[if_model_name](combined_input).item()

        if enc_decision > decision_threshold:
            return true_fn(self)
        else:
            return false_fn(self)

    def encode(self, data: T, dimensions: int = None, **kwargs) -> R:
        if dimensions is None:
            dimensions = self.dimensions
            
        if data is None:
            return self._encode_none(dimensions=dimensions, **kwargs)
        
        # Handle atomic types
        if isinstance(data, (int, float)):
            return self._encode_number(data, dimensions=dimensions, **kwargs)
        
        if isinstance(data, str):
            return self._encode_string(data, dimensions=dimensions, **kwargs)
        
        if isinstance(data, bool):
            return self._encode_boolean(data, dimensions=dimensions, **kwargs)
        
        # Handle container types
        if isinstance(data, list):
            return self._encode_list(data, dimensions=dimensions, **kwargs)
        
        if isinstance(data, tuple):
            return self._encode_tuple(data, dimensions=dimensions, **kwargs)
        
        if isinstance(data, dict):
            return self._encode_dict(data, dimensions=dimensions, **kwargs)
        
        if isinstance(data, set):
            return self._encode_set(data, dimensions=dimensions, **kwargs)
        
        # Handle type objects
        if isinstance(data, type):
            return self._encode_type(data, dimensions=dimensions, **kwargs)
        
        # Handle custom objects - convert to dictionary of attributes
        if hasattr(data, "__dict__"):
            return self._encode_object(data, dimensions=dimensions, **kwargs)
        
        # Fallback for other types: encode their string representation
        return self._encode_string(str(data), dimensions=dimensions, **kwargs)

    def decode(self, data):
        pass

    def select(self, data, query):
        pass

    def update(self, data, updates):
        pass




    def _encode_number(
        self,
        /,
        value: Union[int, float],
        *,
        dimensions: int,
        frequency: float = 10000.0,
        **kwargs,
    ) -> R:
        if not isinstance(value, (int, float)):
            raise TypeError("Input 'value' must be a number.")
        if not isinstance(dimensions, int) or dimensions <= 0:
            raise ValueError("'dimensions' must be a positive integer.")

        position = torch.tensor([[value]], dtype=torch.float32)

        # Create a tensor for dimension indices (0, 1, ..., dimensions-1)
        # dim_indices = torch.arange(dimensions, dtype=torch.float32).unsqueeze(0) # Unused

        # Calculate the divisor term: frequency^(2i/dimensions)
        # div_term will have ceil(dimensions / 2) elements.
        div_term = torch.exp(
            torch.arange(0, dimensions, 2, dtype=torch.float32)
            * -(torch.log(torch.tensor(frequency)) / dimensions)
        )
        div_term = div_term.unsqueeze(0)  # Shape: (1, ceil(dimensions/2))

        # Create the positional encoding tensor
        pe = torch.zeros(1, dimensions, dtype=torch.float32)

        # Apply sin to even indices in the array; 2i
        # This will use all elements of div_term. If dimensions is odd,
        # pe[0, 0::2] will be longer than pe[0, 1::2] by one.
        pe[0, 0::2] = torch.sin(position * div_term)

        # Apply cos to odd indices in the array; 2i+1
        # Ensure we only use the part of div_term that corresponds to available cosine slots.
        num_cos_terms = dimensions // 2
        if num_cos_terms > 0:
            pe[0, 1::2] = torch.cos(position * div_term[:, :num_cos_terms])

        return pe.squeeze()  # Return as a tensor


    def _encode_string(
        self,
        /,
        text: str,
        *,
        dimensions: int,
        model_name: str = "all-MiniLM-L6-v2",
        **kwargs,
    ) -> R:
        if model_name not in self._encode_str_models:
            self._encode_str_models[model_name] = SentenceTransformer(model_name)

        model = self._encode_str_models[model_name]
        embedding_np = model.encode(text)
        embedding_tensor = torch.from_numpy(embedding_np).float()

        # align the embedding to the desired dimensions
        # FIXME: use the Matryoshka technique in the future to handle this more elegantly
        current_dim = embedding_tensor.shape[0]
        if current_dim == dimensions:
            final_embedding = embedding_tensor
        elif current_dim < dimensions:
            padding_size = dimensions - current_dim
            padding = torch.zeros(
                padding_size, dtype=embedding_tensor.dtype, device=embedding_tensor.device
            )
            final_embedding = torch.cat((embedding_tensor, padding))
        else:  # current_dim > dimensions
            final_embedding = embedding_tensor[:dimensions]

        return final_embedding


    def _encode_boolean(
        self,
        /,
        value: bool,
        *,
        dimensions: int,
        **kwargs,
    ) -> R:
        if value:
            return torch.ones(dimensions, dtype=torch.float32)
        else:
            return torch.zeros(dimensions, dtype=torch.float32)


    def _encode_none(
        self,
        /,
        *,
        dimensions: int,
        **kwargs,
    ) -> R:
        return torch.zeros(dimensions, dtype=torch.float32)


    def _encode_list(
        self,
        /,
        items: List[T],
        *,
        dimensions: int,
        reduce_fn: Callable = lambda x, y: x + y,
        **kwargs,
    ) -> R:
        sum_embeddings = torch.zeros(dimensions, dtype=torch.float32)
        for item in items:
            item_embedding = self.encode(item, dimensions=dimensions, **kwargs)
            sum_embeddings = reduce_fn(sum_embeddings, item_embedding)
        return sum_embeddings  # Consider averaging: sum_embeddings / len(items)


    def _encode_tuple(
        self, /, items: Tuple[T, ...], *, dimensions: int, **kwargs
    ) -> R:
        # Tuples are ordered, but for now, treat them like lists
        return self._encode_list(items, dimensions=dimensions, **kwargs)


    def _encode_set(self, /, items: Set[T], *, dimensions: int, **kwargs) -> R:
        # Sets are unordered, but for simplicity, treat them like lists
        return self._encode_list(items, dimensions=dimensions, **kwargs)


    def _encode_dict(
        self, /, data: Dict[Any, Any], *, dimensions: int, **kwargs
    ) -> R:

        aggregated_embedding = torch.zeros(dimensions, dtype=torch.float32)
        for k, v in data.items():
            key_embedding = self.encode(k, dimensions=dimensions, **kwargs)
            value_embedding = self.encode(v, dimensions=dimensions, **kwargs)
            # Key embedding transforms the value embedding using element-wise multiplication.
            # This resulting "key-skewed value encoding" is then aggregated.
            # More sophisticated interaction methods (e.g., attention) could be used here.
            key_transformed_value_embedding = key_embedding * value_embedding
            aggregated_embedding += key_transformed_value_embedding
        return aggregated_embedding  # Consider averaging


    def _encode_type(
        self, /, type_obj: Type, *, dimensions: int, **kwargs
    ) -> R:
        # Create a dictionary of attributes from the type object
        type_attrs = {
            k: getattr(type_obj, k) for k in dir(type_obj) if not k.startswith("__")
        }
        return self._encode_dict(type_attrs, dimensions=dimensions, **kwargs)


    def _encode_object(self, /, obj: Any, *, dimensions: int, **kwargs) -> R:
        # Create a dictionary of attributes from the object
        obj_attrs = {k: getattr(obj, k) for k in dir(obj) if not k.startswith("__")}
        return self._encode_dict(obj_attrs, dimensions=dimensions, **kwargs)
