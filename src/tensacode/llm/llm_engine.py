from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json
from typing import Any, Callable, Type, Union, List, Dict, Set, Tuple

# Import BaseEngine to inherit from it
from tensacode.base.base_engine import BaseEngine, R as BaseR, T as BaseT, ContainerType as BaseContainerType

# Define R and T specifically for this LLM Engine
# R will be a string (JSON string or LLM prompt/response)
# T can be Any Python type that we attempt to serialize/deserialize
R = str
T = BaseT # Keep T as Any
ContainerType = BaseContainerType


class LLMEngine(BaseEngine):
    def __init__(self, model_name: str = "gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Set pad_token_id in model config if not already set
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def _generate_response(self, prompt: str, max_new_tokens: int = 100) -> str:
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024-max_new_tokens) # Ensure space for generation
        
        # Ensure attention_mask is also passed
        attention_mask = (inputs != self.tokenizer.pad_token_id).long()
        
        # Move inputs and attention_mask to the model's device
        device = self.model.device
        inputs = inputs.to(device)
        attention_mask = attention_mask.to(device)

        outputs = self.model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.pad_token_id,
            attention_mask=attention_mask,
            eos_token_id=self.tokenizer.eos_token_id
        )
        # Slice the output to get only the generated tokens
        response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        return response.strip()

    def encode(self, data: T, **kwargs) -> R:
        try:
            return json.dumps(data)
        except TypeError:
            # Fallback for non-serializable types
            return json.dumps(str(data))

    def decode(self, data: R, type_: Type[T]) -> T:
        try:
            py_obj = json.loads(data)
            # Basic attempt to cast if type_ is provided and py_obj is not already of that type
            # This is a simplistic approach for complex types.
            if not isinstance(py_obj, type_) and type_ is not Any:
                try:
                    return type_(py_obj)
                except (TypeError, ValueError):
                    # If casting fails, return the parsed object as is
                    return py_obj
            return py_obj
        except json.JSONDecodeError:
            if type_ is str: # If expecting a string and it's not JSON, return as is
                return data # type: ignore
            # Or, if it was a stringified object that failed to parse, but type_ is string
            # this branch might be hit.
            # Consider if we should raise error or return data as is if not type_ is str
            raise ValueError(f"Failed to decode JSON data and type is not str: {data}")


    def _if(
        self,
        condition: Any,
        value: Any,
        *,
        true_fn: Callable[[BaseEngine], Any], # Adjusted Callable signature
        false_fn: Callable[[BaseEngine], Any], # Adjusted Callable signature
        decision_threshold: float = 0.5,  # Kept for signature compatibility
        match_threshold: float = 0.5, # Kept for signature compatibility (user added this to base)
        activation_fn: Callable = None,   # Kept for signature compatibility
    ):
        condition_str = self.encode(condition)
        value_str = self.encode(value)

        prompt = (
            f"Evaluate the following condition in the context of the given value.\n"
            f"Condition: {condition_str}\n"
            f"Value: {value_str}\n"
            f"Based on this, should a 'true' or 'false' path be taken? "
            f"Respond with only the word 'TRUE' or 'FALSE'."
        )
        
        response = self._generate_response(prompt, max_new_tokens=5) # Short response expected

        if "TRUE" in response.upper():
            return true_fn(self)
        elif "FALSE" in response.upper():
            return false_fn(self)
        else:
            # Fallback if LLM response is unclear
            # print(f"Warning: LLM '_if' decision was unclear: '{response}'. Defaulting to false path.")
            return false_fn(self)

    def select(self, data: Any, query: T, *, max_depth: int = 10, **kwargs) -> Tuple[List[str], T]:
        data_str = self.encode(data)
        query_str = self.encode(query)

        prompt = (
            f"You are an AI assistant. Your task is to select a part of a JSON data structure based on a query.\n"
            f"Data (JSON string): {data_str}\n"
            f"Query: {query_str}\n"
            f"Conceptually, search up to a depth of {max_depth}.\n"
            f"Your response must be a JSON object with two keys: 'path' and 'value'.\n"
            f"- 'path': A list of strings or integers representing keys/indices to the selected element. Example: ['key1', 0, 'key2']. If no specific part matches, use an empty list [].\n"
            f"- 'value': The selected data itself. If path is empty, this should be the original data.\n"
            f"Example response for selecting data['a'][0] which contains 'hello': {{'path': ['a', 0], 'value': 'hello'}}\n"
            f"Example if no good match: {json.dumps({'path': [], 'value': data})} \n"
            f"Response (JSON object):"
        )

        llm_response_str = self._generate_response(prompt, max_new_tokens=250)

        try:
            parsed_response = json.loads(llm_response_str)
            path = parsed_response.get("path", [])
            selected_value = parsed_response.get("value") # This value is already a Python object

            if not isinstance(path, list) or selected_value is None :
                # print(f"Warning: LLM 'select' response path not a list or value is missing. Full response: {llm_response_str}")
                return [], data # Fallback

            return path, selected_value
        except json.JSONDecodeError:
            # print(f"Warning: LLM 'select' response was not valid JSON: '{llm_response_str}'.")
            return [], data # Fallback
        except Exception as e:
            # print(f"Warning: Error parsing LLM 'select' response: '{llm_response_str}'. Error: {e}")
            return [], data


    def update(self, obj_to_update: ContainerType, update_info_str: R) -> ContainerType:
        obj_str = self.encode(obj_to_update)
        # update_info_str is R, which is a string (intended to be JSON of the update)

        prompt = (
            f"You are an AI assistant that updates a JSON data structure.\n"
            f"Current Object (JSON string): {obj_str}\n"
            f"Update Information (JSON string describing changes or new values): {update_info_str}\n"
            f"Your task is to intelligently update the Current Object using the Update Information. "
            f"This might involve replacing parts, adding new fields, or modifying existing values based on the semantic meaning of the Update Information.\n"
            f"Your response must be the entire new, updated JSON object as a single JSON string.\n"
            f"Example: If Current Object is {json.dumps({'a':1, 'b':2})} and Update Information is {json.dumps({'b':3, 'c':4})}, "
            f"the response should be {json.dumps({'a':1, 'b':3, 'c':4})}.\n"
            f"Response (the new JSON object as a string):"
        )
        
        llm_response_str = self._generate_response(prompt, max_new_tokens=500)

        try:
            # The LLM should return a string which is the new JSON object
            updated_obj_py = json.loads(llm_response_str)
            
            # Basic check if the result is a container type, though LLM might return primitives if it misinterprets
            if not isinstance(updated_obj_py, (dict, list, tuple, set)):
                # This could happen if the update results in a scalar, or LLM misunderstands.
                # Depending on strictness, one might raise an error or try to wrap it.
                # For now, we'll trust the LLM's output if it's valid JSON.
                # print(f"Warning: LLM 'update' response is valid JSON but not a primary container type: {type(updated_obj_py)}")
                pass

            return updated_obj_py # type: ignore
        except json.JSONDecodeError:
            # print(f"Warning: LLM 'update' response was not valid JSON: '{llm_response_str}'. Returning original object.")
            return obj_to_update
        except Exception as e:
            # print(f"Warning: Error processing LLM 'update' response: '{llm_response_str}'. Error: {e}. Returning original object.")
            return obj_to_update
