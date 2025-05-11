from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

_R = TypeVar('_R')  # Return type for cached value producer

# Global stores for hooks
_GLOBAL_HOOK_STATES: Dict[Any, Any] = {}
_GLOBAL_EFFECTS_DATA: Dict[Any, Dict[str, Any]] = {}
_GLOBAL_CACHED_VALUES: Dict[Any, Dict[int, Any]] = {}
_GLOBAL_CACHED_INDICES: Dict[Any, List[int]] = {}


def use_state(key: Any, initial_value: Any) -> Tuple[Any, Callable[[Any], None]]:
    """A hook to add state, keyed globally."""
    if key not in _GLOBAL_HOOK_STATES:
        _GLOBAL_HOOK_STATES[key] = initial_value() if callable(initial_value) else initial_value
    
    def set_state(new_value_or_updater: Any):
        _GLOBAL_HOOK_STATES[key] = (
            new_value_or_updater(_GLOBAL_HOOK_STATES[key]) 
            if callable(new_value_or_updater) 
            else new_value_or_updater
        )
        
    return _GLOBAL_HOOK_STATES[key], set_state


def use_effect(key: Any, effect_fn: Callable[[], Optional[Callable[[], None]]], deps: Optional[List[Any]] = None):
    """A hook to perform side effects, keyed globally."""
    data = _GLOBAL_EFFECTS_DATA.get(key)
    
    is_first_run = data is None
    if is_first_run:
        data = {'current_deps': deps, 'prev_deps': None, 'cleanup_fn': None}
        _GLOBAL_EFFECTS_DATA[key] = data
    else:
        data['prev_deps'], data['current_deps'] = data['current_deps'], deps
    
    deps_changed = (
        deps is None or 
        data['prev_deps'] is None or
        len(deps) != len(data['prev_deps']) or
        any(cd != pd for cd, pd in zip(deps, data['prev_deps']))
    )
    
    if is_first_run or deps_changed:
        if callable(data['cleanup_fn']):
            try: 
                data['cleanup_fn']()
            except Exception as e:
                print(f"Error in effect cleanup for key '{key}': {e}")
        
        try:
            data['cleanup_fn'] = effect_fn()
        except Exception as e:
            print(f"Error in effect execution for key '{key}': {e}")


def use_cached(idx: int, cache_key: Any, value_producer_fn: Callable[[], _R]) -> Optional[_R]:
    """Caches calls and retrieves the nearest prior index value if the index is a miss."""
    # Initialize cache structures if needed
    if cache_key not in _GLOBAL_CACHED_VALUES:
        _GLOBAL_CACHED_VALUES[cache_key] = {}
        _GLOBAL_CACHED_INDICES[cache_key] = []
    
    values = _GLOBAL_CACHED_VALUES[cache_key]
    indices = _GLOBAL_CACHED_INDICES[cache_key]
    
    # Direct cache hit
    if idx in values:
        return values[idx]
    
    # Find nearest prior idx
    prior_value = None
    nearest_prior_idx = None
    for i in indices:
        if i < idx and (nearest_prior_idx is None or i > nearest_prior_idx):
            nearest_prior_idx = i
    
    if nearest_prior_idx is not None:
        prior_value = values[nearest_prior_idx]
    
    # Calculate and store the actual value
    actual_value = value_producer_fn()
    values[idx] = actual_value
    if idx not in indices:
        indices.append(idx)
        indices.sort()  # Keep indices sorted for future lookups
    
    return prior_value


__all__ = ["use_state", "use_effect", "use_cached"]
