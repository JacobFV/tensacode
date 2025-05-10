# TensaCode

- __init__.py
- base/
    - base_engine.py
        class BaseEngine:
            def _if ... wrap around the if impls
            def encode
            def decode
            def select
            def update
    - _ops/{if,encode,decode,select,update}.py
- nn/
    - nn_engine.py
    - _ops/{if,encode,decode,select,update}.py
- llm/
    - llm_engine.py
    - _ops/{if,encode,decode,select,update}.py
_utils/
