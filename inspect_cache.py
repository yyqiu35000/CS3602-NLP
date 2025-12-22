from transformers.cache_utils import DynamicCache
import inspect

print("DynamicCache source:")
try:
    print(inspect.getsource(DynamicCache))
except Exception as e:
    print(e)

print("\nDynamicCache.update source:")
try:
    print(inspect.getsource(DynamicCache.update))
except Exception as e:
    print(e)
