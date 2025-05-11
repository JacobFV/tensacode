from typing import Any, Callable, Type, Union, List, Dict, Set, Tuple
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from glom import assign
import numpy as np
from typing import get_type_hints, get_args, get_origin
import math
from transformers import GPT2LMHeadModel, GPT2Tokenizer

R = torch.Tensor  # Type for encoded/embedding tensors
T = Any  # Type for data to encode

AtomicType = Union[int, float, str, bool]
ContainerType = Union[list, tuple, dict, set]

class NNEngine:

    dimensions: int
    frequency: float

    _encode_str_model: SentenceTransformer
    _if_model: nn.Sequential
    _update_query_model: nn.Sequential
    _update_value_model: nn.Sequential
    _decode_boolean_model: nn.Sequential
    _decode_boolean_threshold: float
    _match_threshold: float
    _decode_str_tokenizer: GPT2Tokenizer
    _decode_str_model: GPT2LMHeadModel
    _decode_str_projection: nn.Linear = None

    def __init__(
        self,
        dimensions: int,
        *,
        text_model_name: str = "all-MiniLM-L6-v2",
        if_activation_fn: Callable = nn.ReLU,
        frequency: float = 65536.0,
        decode_boolean_threshold: float = 0.5,
        text_decoder_name: str = "gpt2",
        match_threshold: float = 0.5,
    ):
        self.dimensions = dimensions
        self.frequency = frequency
        self._decode_boolean_threshold = decode_boolean_threshold
        self._match_threshold = match_threshold

        # Initialize text embedding model
        self._encode_str_model = SentenceTransformer(text_model_name)
        self.text_model_name = text_model_name

        # Initialize text decoding models
        self._decode_str_tokenizer = GPT2Tokenizer.from_pretrained(text_decoder_name)
        self._decode_str_model = GPT2LMHeadModel.from_pretrained(text_decoder_name)
        self.text_decoder_name = text_decoder_name
        
        # Initialize projection layer for string decoding
        self._decode_str_projection = torch.nn.Linear(
            self.dimensions,
            self._decode_str_model.config.n_embd
        )

        # Initialize if model
        self._if_model = nn.Sequential(
            nn.Linear(2 * self.dimensions, self.dimensions),
            if_activation_fn(),
            nn.Linear(self.dimensions, self.dimensions // 2),
            if_activation_fn(),
            nn.Linear(self.dimensions // 2, 1),
            nn.Sigmoid(),
        )
        self.if_activation_fn = if_activation_fn

        # Initialize update query model
        self._update_query_model = nn.Sequential(
            nn.Linear(self.dimensions, self.dimensions),
            if_activation_fn(),
            nn.Linear(self.dimensions, self.dimensions // 2),
            if_activation_fn(),
            nn.Linear(self.dimensions // 2, self.dimensions),
        )

        # Initialize update value model
        self._update_value_model = nn.Sequential(
            nn.Linear(2 * self.dimensions, self.dimensions),
            if_activation_fn(),
            nn.Linear(self.dimensions, self.dimensions // 2),
            if_activation_fn(),
            nn.Linear(self.dimensions // 2, self.dimensions),
        )
        
        # Initialize boolean decode model
        self._decode_boolean_model = nn.Sequential(
            nn.Linear(self.dimensions, self.dimensions // 2),
            if_activation_fn(),
            nn.Linear(self.dimensions // 2, 1),
            nn.Sigmoid(),
        )


        self._decode_list_decide_continue_decoding_vec = self.encode("should i continue decoding?")

    def _if(
        self,
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

        combined_input = torch.cat((enc_condition, enc_value))
        enc_decision = self._if_model(combined_input).item()

        if enc_decision > decision_threshold:
            return true_fn(self)
        else:
            return false_fn(self)

    def encode(self, data: T, **kwargs) -> R:
        if data is None:
            return self._encode_none(**kwargs)

        # Handle atomic types
        if isinstance(data, (int, float)):
            return self._encode_number(data, **kwargs)

        if isinstance(data, str):
            return self._encode_string(data, **kwargs)

        if isinstance(data, bool):
            return self._encode_boolean(data, **kwargs)

        # Handle container types
        if isinstance(data, list):
            return self._encode_list(data, **kwargs)

        if isinstance(data, tuple):
            return self._encode_tuple(data, **kwargs)

        if isinstance(data, dict):
            return self._encode_dict(data, **kwargs)

        if isinstance(data, set):
            return self._encode_set(data, **kwargs)

        # Handle type objects
        if isinstance(data, type):
            return self._encode_type(data, **kwargs)

        # Handle custom objects - convert to dictionary of attributes
        if hasattr(data, "__dict__"):
            return self._encode_object(data, **kwargs)

        # Fallback for other types: encode their string representation
        return self._encode_string(str(data), **kwargs)

    def decode(self, data: R, type_: Type[T]) -> T:
        # Handle None type
        if type_ is type(None):
            return None
            
        # Handle atomic types
        if type_ in (int, float):
            return self._decode_number(data, type_)
            
        if type_ is str:
            return self._decode_string(data)
            
        if type_ is bool:
            return self._decode_boolean(data)
            
        # Extract type information for container types
        origin = get_origin(type_)
        args = get_args(type_)
        
        # Handle container types
        if origin is list or type_ is list:
            elem_type = args[0] if args else None
            return self._decode_list(data, elem_type)
            
        if origin is tuple or type_ is tuple:
            # For tuple, pass the specific types for each position
            return self._decode_tuple(data, args if args else None)
            
        if origin is dict or type_ is dict:
            key_type = args[0] if len(args) > 0 else str
            value_type = args[1] if len(args) > 1 else None
            return self._decode_dict(data, key_type, value_type)
            
        if origin is set or type_ is set:
            elem_type = args[0] if args else None
            return self._decode_set(data, elem_type)
            
        # Handle custom objects
        if hasattr(type_, "__dict__"):
            return self._decode_object(data, type_)
            
        # Fallback for other types
        return type_(self._decode_string(data))

    def select(self, data: Any, query: T, *, max_depth: int = 10, **kwargs) -> Tuple[List[str], T]:

        if max_depth == 0:
            return ([], data)

        # Encode the query to get its representation
        query_encoding = self.encode(query, **kwargs)

        # Handle atomic types
        if data is None:
            return ([], data)
        if isinstance(data, AtomicType):
            return ([], data)

        data_dict = {}
        k_enc_list = []

        # Handle container types
        if isinstance(data, list):
            data_dict = {i: item for i, item in enumerate(data)}
            k_enc_list = [self.encode(item, **kwargs) for item in data]
        elif isinstance(data, tuple):
            data_dict = {i: item for i, item in enumerate(data)}
            k_enc_list = [self.encode(item, **kwargs) for item in data]
        elif isinstance(data, set):
            data_dict = {i: item for i, item in enumerate(data)}
            k_enc_list = [self.encode(item, **kwargs) for item in data]
        elif isinstance(data, dict):
            data_dict = data
            k_enc_list = [self.encode(k, **kwargs) * self.encode(v, **kwargs) for k, v in data.items()]
        # Handle type objects
        elif isinstance(data, type):
            data_dict = {
                k: getattr(data, k)
                for k in dir(data)
                if not k.startswith("__") and not callable(getattr(data, k))
            }
            k_enc_list = [self.encode(k, **kwargs) * self.encode(v, **kwargs) for k, v in data_dict.items()]
        # Handle custom objects - convert to dictionary of attributes
        elif hasattr(data, "__dict__"):
            data_dict = {
                k: getattr(data, k)
                for k in dir(data)
                if not k.startswith("__") and not callable(getattr(data, k))
            }
            k_enc_list = [self.encode(k, **kwargs) * self.encode(v, **kwargs) for k, v in data_dict.items()]
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        # Select the best match
        similarities = [torch.nn.functional.cosine_similarity(query_encoding, k_enc) for k_enc in k_enc_list]

        best_similarity_index, best_similarity = max(enumerate(similarities), key=lambda x: x[1])

        if best_similarity < self._match_threshold:
            # just return the object as is
            # at this point, going deeper is not likely to yield better results
            return ([], data)

        best_k = list(data_dict.keys())[best_similarity_index]
        path, selected_val = self.select(data_dict[best_k], query, max_depth=max_depth-1, **kwargs)
        
        # Prepend the current key to the path
        return ([best_k] + path, selected_val)

    def update(self, object: ContainerType, update: R):
        update_encoding = self.encode(update)
        update_query_enc = self._update_query_model(update_encoding)
        path, old_val = self.select(object, update_query_enc)
        old_val_enc = self.encode(old_val)
        new_val_enc = self._update_value_model(old_val_enc, update_encoding)
        new_val = self.decode(new_val_enc, type(old_val))

        if not path:
            return new_val  # If no path, just return the update directly
        
        path_str = '.'.join(path)
        assign(object, path_str, new_val)
        
        return object

    def _encode_number(
        self,
        value: Union[int, float],
        **kwargs,
    ) -> R:
        if not isinstance(value, (int, float)):
            raise TypeError("Input 'value' must be a number.")
        if self.dimensions < 2:
            raise ValueError("Dimensions must be at least 2 for number encoding")
        
        position = torch.tensor([[value]], dtype=torch.float32)
        
        # Calculate frequencies that scale exponentially: 2^(i/2) for each pair of indices
        # This gives frequency 1 for indices 0,1; 2 for indices 2,3; 4 for indices 4,5; etc.
        exponents = torch.floor(torch.arange(0, self.dimensions, dtype=torch.float32) / 2)
        frequencies = 2 ** exponents  # Use powers of 2 instead of e for efficiency
        
        # Create the output tensor
        pe = torch.zeros(self.dimensions, dtype=torch.float32)
        
        # Apply sin to even indices (0,2,4,...)
        pe[0::2] = torch.sin(position * frequencies[0::2])
        
        # Apply cos to odd indices (1,3,5,...)
        pe[1::2] = torch.cos(position * frequencies[1::2])
        
        return pe  # Return as a 1D tensor

    def _encode_string(
        self,
        text: str,
        *,
        normalize: bool = True,
        **kwargs,
    ) -> R:
        embedding_np = self._encode_str_model.encode(text)
        embedding_tensor = torch.from_numpy(embedding_np).float()

        # align the embedding to the desired dimensions
        # FIXME: use the Matryoshka technique in the future to handle this more elegantly
        current_dim = embedding_tensor.shape[0]
        if current_dim == self.dimensions:
            final_embedding = embedding_tensor
        elif current_dim < self.dimensions:
            padding_size = self.dimensions - current_dim
            padding = torch.zeros(
                padding_size,
                dtype=embedding_tensor.dtype,
                device=embedding_tensor.device,
            )
            final_embedding = torch.cat((embedding_tensor, padding))
        else:  # current_dim > dimensions
            final_embedding = embedding_tensor[: self.dimensions]

        return final_embedding

    def _encode_boolean(
        self,
        value: bool,
        **kwargs,
    ) -> R:
        if value:
            return torch.ones(self.dimensions, dtype=torch.float32)
        else:
            return torch.zeros(self.dimensions, dtype=torch.float32)

    def _encode_none(
        self,
        *,
        fill_value: float = 0.0,
        **kwargs,
    ) -> R:
        return torch.zeros(self.dimensions, dtype=torch.float32) + fill_value

    def _encode_list(
        self,
        items: List[T],
        *,
        reduce_fn: Callable = lambda x, y: x + y,
        **kwargs,
    ) -> R:
        sum_embeddings = torch.zeros(self.dimensions, dtype=torch.float32)
        for item in items:
            item_embedding = self.encode(item, **kwargs)
            sum_embeddings = reduce_fn(sum_embeddings, item_embedding)
        return sum_embeddings  # Consider averaging: sum_embeddings / len(items)

    def _encode_tuple(
        self, items: Tuple[T, ...], *, ordered: bool = True, **kwargs
    ) -> R:
        # Tuples are ordered, but for now, treat them like lists
        return self._encode_list(items, **kwargs)

    def _encode_set(self, items: Set[T], *, unordered: bool = True, **kwargs) -> R:
        # Sets are unordered, but for simplicity, treat them like lists
        return self._encode_list(items, **kwargs)

    def _encode_dict(
        self, data: Dict[Any, Any], *, normalize: bool = False, **kwargs
    ) -> R:

        aggregated_embedding = torch.zeros(self.dimensions, dtype=torch.float32)
        for k, v in data.items():
            key_embedding = self.encode(k, **kwargs)
            value_embedding = self.encode(v, **kwargs)
            # Key embedding transforms the value embedding using element-wise multiplication.
            # This resulting "key-skewed value encoding" is then aggregated.
            # More sophisticated interaction methods (e.g., attention) could be used here.
            key_transformed_value_embedding = key_embedding * value_embedding
            aggregated_embedding += key_transformed_value_embedding

        if normalize and len(data) > 0:
            aggregated_embedding = aggregated_embedding / len(data)

        return aggregated_embedding

    def _encode_type(
        self, type_obj: Type, *, include_methods: bool = False, **kwargs
    ) -> R:
        # Create a dictionary of attributes from the type object
        type_attrs = {
            k: getattr(type_obj, k)
            for k in dir(type_obj)
            if not k.startswith("__")
            and (include_methods or not callable(getattr(type_obj, k)))
        }
        return self._encode_dict(type_attrs, **kwargs)

    def _encode_object(
        self, obj: Any, *, include_methods: bool = False, **kwargs
    ) -> R:
        # Create a dictionary of attributes from the object
        obj_attrs = {
            k: getattr(obj, k)
            for k in dir(obj)
            if not k.startswith("__")
            and (include_methods or not callable(getattr(obj, k)))
        }
        return self._encode_dict(obj_attrs, **kwargs)

    def _decode_number(self, data: R, type_=None) -> Union[int, float]:
        if self.dimensions < 2:
            raise ValueError("Dimensions must be at least 2 for number decoding")
        
        # Extract sine and cosine components
        sine_components = data[0::2]
        cosine_components = data[1::2]
        
        # Calculate frequencies (same as in encoding)
        exponents = torch.floor(torch.arange(0, self.dimensions, dtype=torch.float32) / 2)
        frequencies = 2 ** exponents
        
        # Use the first pair (sine and cosine at the lowest frequency)
        # Sorry we are only using the first pair, but it's good enough for now
        sine_val = sine_components[0].item()
        cosine_val = cosine_components[0].item()
        
        # Get the frequency for this pair
        freq = frequencies[0].item()
        
        # Use atan2 to get the phase
        # atan2(sin(x), cos(x)) = x (mod 2π)
        phase = math.atan2(sine_val, cosine_val)
        
        # Calculate the value
        value = phase / freq
        
        # Convert to int if requested
        if type_ == int:
            return int(round(value))
        return value

    def _decode_string(self, data: R) -> str:
        # Use transformers to convert vector to text
        
        # Reshape input vector to match model dimensions
        reshaped_input = data.unsqueeze(0)  # Add batch dimension
        
        # Project input to match model's embedding dimension
        reshaped_input = self._decode_str_projection(reshaped_input)
        
        # Generate text from the vector
        outputs = self._decode_str_model.generate(
            inputs_embeds=reshaped_input,
            max_length=50,
            do_sample=True,
            top_p=0.95,
            top_k=50
        )
        
        return self._decode_str_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
    def _decode_boolean(self, data: R) -> bool:
        lolgit = self._decode_boolean_model(data)
        return lolgit > self._decode_boolean_threshold

    def _decode_list(self, data: R, elem_type: Type = None) -> List:
        result_list = []
        prev_item_enc = data
        while self._if(self._decode_list_decide_continue_decoding_vec, prev_item_enc, true_fn=lambda: True, false_fn=lambda: False):
            new_item_enc = self._decode_new_item_model(data, prev_item_enc)
            new_item = self.decode(new_item_enc)
            result_list.append(new_item)
            prev_item_enc = new_item_enc
        return result_list

    def _decode_tuple(self, data: R, tuple_types: Tuple[Type, ...] = None) -> Tuple:
        if tuple_types:
            return tuple(self.decode(data, elem_type) for elem_type in tuple_types)
        else:
            return tuple(self._decode_list(data))

    def _decode_dict(self, data: R, key_type: Type = str, value_type: Type = None) -> Dict:
        return {key: self.decode(data[key], value_type) for key in data}

    def _decode_set(self, data: R, elem_type: Type = None) -> Set:
        return set(self._decode_list(data, elem_type))
        
    def _decode_object(self, data: R, type_: Type) -> Any:
        try:
            type_hints = get_type_hints(type_)
        except (TypeError, NameError):
            type_hints = {}

        initial_attr_dict: Dict[str, Any]
        try:
            # Attempt to decode the input tensor 'data' into a Python dictionary.
            # Values in this dictionary are generically decoded at this stage.
            # This relies on _decode_dict's ability to convert R to Dict.
            decoded_data = self._decode_dict(data)
            if isinstance(decoded_data, dict):
                initial_attr_dict = decoded_data
            else:
                # If _decode_dict doesn't return a dict (e.g., on error or unexpected data format),
                # default to an empty dictionary for attribute population.
                initial_attr_dict = {}
        except Exception:
            # If an error occurs during _decode_dict(data),
            # default to an empty dictionary.
            initial_attr_dict = {}

        obj = None
        # Try different initialization strategies for the object.
        # _initialize_from_dict will use the generically decoded initial_attr_dict.
        # If __init__ or from_dict are type-sensitive, they might fail, which is acceptable,
        # as subsequent attribute setting will handle type coercion.
        initialization_methods = [
            lambda: type_.__new__(type_),
            lambda: type_(),
            lambda: self._initialize_from_dict(type_, initial_attr_dict)
        ]
        
        for init_method in initialization_methods:
            try:
                candidate_obj = init_method()
                if candidate_obj is not None: # Ensure __new__ or __init__ didn't return None
                    obj = candidate_obj
                    break
            except (TypeError, ValueError, AttributeError):
                continue # Try next initialization method
        
        if obj is None:
            # If all initialization methods fail, return the initial dictionary representation
            # (with generically decoded values).
            return initial_attr_dict
        
        # Object successfully initialized.
        # Now, apply attributes from initial_attr_dict, using type hints for proper coercion.
        # This may overwrite attributes set during _initialize_from_dict, but ensures correct typing.
        for attr_name, raw_attr_value in initial_attr_dict.items():
            # Skip special methods and attributes
            if attr_name.startswith('__') and attr_name.endswith('__'):
                continue
                
            if attr_name in type_hints:
                expected_type = type_hints[attr_name]
                self._apply_typed_attribute(obj, attr_name, raw_attr_value, expected_type)
            else:
                # No type hint available. Set attribute with the raw decoded value.
                try:
                    setattr(obj, attr_name, raw_attr_value)
                except (AttributeError, TypeError):
                    # Skip if attribute can't be set (e.g., read-only)
                    pass
        
        return obj
        
    def _initialize_from_dict(self, type_: Type, data: Dict) -> Any:
        """
        Try to initialize an object using a dictionary of attributes.
        The 'data' dict contains generically decoded values.
        """
        # Check for common initialization patterns
        if hasattr(type_, 'from_dict'):
            return type_.from_dict(data)
        
        if hasattr(type_, '__init__'):
            # Try to match init parameters with dict keys
            try:
                import inspect
                init_params = inspect.signature(type_.__init__).parameters
                # Filter dict to only include keys that match init parameters
                init_args = {k: v for k, v in data.items() if k in init_params}
                return type_(**init_args)
            except (TypeError, ValueError):
                pass # Initialization with __init__ failed
        
        # Default to raising an error if no suitable initialization found,
        # to allow _decode_object to try other methods.
        raise TypeError("Could not initialize from dictionary via from_dict or __init__")
        
    def _apply_typed_attribute(self, obj: Any, attr_name: str, attr_value: Any, expected_type: Type) -> None:
        """
        Set an attribute on 'obj' with 'attr_value', attempting to conform it to 'expected_type'.
        """
        origin = get_origin(expected_type)
        args = get_args(expected_type)
        
        # Handle Union and Optional types
        if origin is Union:
            # Handle Optional: if None is a valid type and value is None
            if type(None) in args and attr_value is None:
                try:
                    setattr(obj, attr_name, None)
                except (AttributeError, TypeError):
                    pass # Skip if attr is read-only or None not assignable
                return
            
            # Try each non-None type in the Union
            for arg_type in args:
                if arg_type is type(None):
                    continue
                try:
                    # Attempt to encode the current value and decode it to the specific arg_type
                    encoded_val = self.encode(attr_value)
                    typed_value = self.decode(encoded_val, arg_type)
                    setattr(obj, attr_name, typed_value)
                    return # Successfully set with one of the Union types
                except Exception:
                    # Failure with this arg_type (encode, decode, or setattr). Try next.
                    continue
            # If loop completes, no Union type led to successful assignment. Do nothing further.
            return

        # For non-Union types, attempt assignment in a specific order of preference:
        try:
            # 1. Most robust: re-encode current value and decode to the specific expected_type
            encoded_val = self.encode(attr_value)
            final_value = self.decode(encoded_val, expected_type)
            setattr(obj, attr_name, final_value)
            return
        except Exception:
            pass # Fall through to the next attempt

        try:
            # 2. Fallback: direct type conversion (e.g., int("123"))
            final_value = expected_type(attr_value)
            setattr(obj, attr_name, final_value)
            return
        except Exception:
            pass # Fall through to the next attempt

        try:
            # 3. Last resort: set the attribute_value as-is
            setattr(obj, attr_name, attr_value)
        except (AttributeError, TypeError):
            # If setting as-is fails (e.g., read-only, fundamental type mismatch)
            pass # Skip setting the attribute
