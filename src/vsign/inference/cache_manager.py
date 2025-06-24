"""
LRU Cache Manager for Sign Language Recognition Inference.
Provides sequence-based result caching.
"""
import hashlib
import io
import logging
import numpy as np
import time
from collections import OrderedDict
from PIL import Image

# Configure logging
logger = logging.getLogger(__name__)

# Cache configuration
DEFAULT_MAX_CACHE_SIZE = 50

def get_image_hash(image_array):
    """
    Compute MD5 hash for image (PIL Image or numpy array).
    
    Args:
        image_array: PIL Image or numpy array
        
    Returns:
        str: MD5 hash of the image
    """
    if isinstance(image_array, Image.Image):
        img_byte_arr = io.BytesIO()
        image_array.save(img_byte_arr, format=image_array.format or 'PNG')
        img_byte_arr = img_byte_arr.getvalue()
    elif isinstance(image_array, np.ndarray):
        img_byte_arr = image_array.tobytes()
    else:
        raise ValueError(f"Unsupported image type: {type(image_array)}")
    
    return hashlib.md5(img_byte_arr).hexdigest()

def get_image_sequence_hash_from_images(images):
    """
    Compute hash for image sequence using individual image hashes.
    Enables better cache reuse when images repeat across sequences.
    
    Args:
        images (list): List of PIL Images or numpy arrays
        
    Returns:
        str: Combined hash of the sequence
    """
    individual_hashes = []
    for img in images:
        img_hash = get_image_hash(img)
        individual_hashes.append(img_hash)
    
    combined_string = "_".join(individual_hashes)
    return hashlib.md5(combined_string.encode()).hexdigest()

class LRUInferenceCache(OrderedDict):
    """LRU cache for inference results with automatic eviction."""
    
    def __init__(self, maxsize=DEFAULT_MAX_CACHE_SIZE):
        self.maxsize = maxsize
        super().__init__()
        logger.info(f"Initialized LRU inference cache with max size: {maxsize}")
        
    def __getitem__(self, key):
        """Get result and mark as recently used."""
        value = super().__getitem__(key)
        self.move_to_end(key)
        logger.info(f"Cache hit for key: {key[:8]}...")
        return value
    
    def __setitem__(self, key, value):
        """Store result with LRU eviction."""
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        
        # Evict oldest if limit exceeded
        if len(self) > self.maxsize:
            oldest = next(iter(self))
            logger.info(f"Cache limit reached. Removing oldest item: {oldest[:8]}...")
            del self[oldest]
            
        logger.info(f"Cached result for key: {key[:8]}... (cache size: {len(self)}/{self.maxsize})")

def compute_image_sequence_hash(img_list):
    """
    Compute hash for image sequence using raw image data.
    
    Args:
        img_list (list): List of numpy arrays
        
    Returns:
        str: MD5 hash of the sequence
    """
    hasher = hashlib.md5()
    
    for img_array in img_list:
        img_bytes = img_array.tobytes()
        hasher.update(img_bytes)
    
    # Include sequence metadata
    sequence_info = f"{len(img_list)}_{img_list[0].shape if img_list else 'empty'}"
    hasher.update(sequence_info.encode())
    
    return hasher.hexdigest()

def compute_file_sequence_hash(file_paths):
    """
    Compute hash for file sequence using paths and modification times.
    
    Args:
        file_paths (list): List of file paths
        
    Returns:
        str: MD5 hash of the file sequence
    """
    import os
    
    hasher = hashlib.md5()
    
    for file_path in file_paths:
        if os.path.exists(file_path):
            mtime = os.path.getmtime(file_path)
            file_info = f"{file_path}_{mtime}"
        else:
            file_info = f"{file_path}_missing"
        hasher.update(file_info.encode())
    
    return hasher.hexdigest()

def is_rephrasing_skipped_message(text):
    """
    Check if the rephrased sentence is a 'skipped' message (no actual rephrasing).
    
    Args:
        text (str): The rephrased sentence text
        
    Returns:
        bool: True if this is a skipped message, False if actual rephrasing
    """
    if not text:
        return True
        
    skip_indicators = [
        "Rephrasing skipped:",
        "No API key provided",
        "Empty recognition",
        "API Key not provided",
        "Rephrasing error:",
        "LLM rephrase sentence failed:"
    ]
    
    return any(indicator in text for indicator in skip_indicators)

class CachedInferenceResult:
    """Container for cached inference results with metadata."""
    
    def __init__(self, result_string, rephrased_sentence, timing_info, timestamp=None):
        self.result_string = result_string
        self.rephrased_sentence = rephrased_sentence
        self.timing_info = timing_info
        self.timestamp = timestamp or time.time()
        self.cache_hit_count = 0
    
    def increment_hit_count(self):
        """Increment cache hit counter."""
        self.cache_hit_count += 1
    
    def update_rephrased_sentence(self, new_rephrased_sentence, additional_llm_time):
        """
        Update the rephrased sentence and LLM timing in this cached result.
        
        Args:
            new_rephrased_sentence (str): The new rephrased sentence
            additional_llm_time (float): Additional LLM processing time
        """
        self.rephrased_sentence = new_rephrased_sentence
        self.timing_info['llm_time'] += additional_llm_time
        self.timing_info['total_time'] += additional_llm_time
        logger.info(f"Updated cached result with new rephrased sentence")
    
    def needs_rephrasing(self, api_key_provided):
        """
        Check if this cached result needs LLM rephrasing.
        
        Args:
            api_key_provided (bool): Whether an API key is now provided
            
        Returns:
            bool: True if rephrasing should be performed
        """
        return (api_key_provided and 
                self.result_string and  # Has model prediction
                is_rephrasing_skipped_message(self.rephrased_sentence))  # But no actual rephrasing
        
    def to_dict(self):
        """Convert to dictionary for debugging."""
        return {
            'result_string': self.result_string,
            'rephrased_sentence': self.rephrased_sentence,
            'timing_info': self.timing_info,
            'timestamp': self.timestamp,
            'cache_hit_count': self.cache_hit_count
        }

class InferenceCacheManager:
    """Main cache manager for inference results."""
    
    def __init__(self, max_cache_size=DEFAULT_MAX_CACHE_SIZE):
        self.cache = LRUInferenceCache(max_cache_size)
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'llm_updates': 0  # Track LLM-only updates
        }
        logger.info(f"Initialized inference cache manager with max size: {max_cache_size}")
    
    def get_cache_key(self, img_list=None, file_paths=None, use_optimized_hashing=True):
        """
        Generate cache key for image sequence.
        
        Args:
            img_list: List of image arrays
            file_paths: List of file paths
            use_optimized_hashing: Use individual image hashing for better cache reuse
            
        Returns:
            str: Cache key for the sequence
        """
        if img_list is not None:
            if use_optimized_hashing:
                return get_image_sequence_hash_from_images(img_list)
            else:
                return compute_image_sequence_hash(img_list)
        elif file_paths is not None:
            return compute_file_sequence_hash(file_paths)
        else:
            raise ValueError("Either img_list or file_paths must be provided")
    
    def get_cached_result(self, cache_key):
        """Retrieve cached inference result by key."""
        self.stats['total_requests'] += 1
        
        if cache_key in self.cache:
            self.stats['cache_hits'] += 1
            cached_result = self.cache[cache_key]
            cached_result.increment_hit_count()
            logger.info(f"Cache hit! Retrieved result for key: {cache_key[:8]}...")
            return cached_result
        else:
            self.stats['cache_misses'] += 1
            logger.info(f"Cache miss for key: {cache_key[:8]}...")
            return None
    
    def store_result(self, cache_key, result_string, rephrased_sentence, timing_info):
        """Store inference result in cache."""
        cached_result = CachedInferenceResult(
            result_string=result_string,
            rephrased_sentence=rephrased_sentence,
            timing_info=timing_info
        )
        
        self.cache[cache_key] = cached_result
        logger.info(f"Stored new result in cache for key: {cache_key[:8]}...")
    
    def update_cached_rephrasing(self, cache_key, new_rephrased_sentence, llm_time_taken):
        """
        Update only the rephrased sentence in an existing cached result.
        
        Args:
            cache_key (str): The cache key for the result to update
            new_rephrased_sentence (str): The new rephrased sentence
            llm_time_taken (float): Time taken for LLM processing
            
        Returns:
            bool: True if update was successful, False if key not found
        """
        if cache_key in self.cache:
            cached_result = self.cache[cache_key]
            cached_result.update_rephrased_sentence(new_rephrased_sentence, llm_time_taken)
            self.stats['llm_updates'] += 1
            logger.info(f"Updated rephrasing for cached result: {cache_key[:8]}...")
            return True
        else:
            logger.warning(f"Attempted to update non-existent cache key: {cache_key[:8]}...")
            return False
    
    def check_needs_rephrasing(self, cache_key, api_key_provided):
        """
        Check if a cached result needs LLM rephrasing and increment hit statistics.
        
        Args:
            cache_key (str): The cache key to check
            api_key_provided (bool): Whether an API key is provided
            
        Returns:
            tuple: (needs_rephrasing, cached_result)
                - needs_rephrasing: True if rephrasing should be performed
                - cached_result: The cached result object (or None)
        """
        # Use get_cached_result to properly increment statistics
        cached_result = self.get_cached_result(cache_key)
        
        if cached_result:
            needs_rephrasing = cached_result.needs_rephrasing(api_key_provided)
            return needs_rephrasing, cached_result
        return False, None
    
    def get_cache_stats(self):
        """Get comprehensive cache statistics."""
        total_requests = self.stats['total_requests']
        hit_rate = (self.stats['cache_hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'total_requests': total_requests,
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'hit_rate_percent': round(hit_rate, 2),
            'cache_size': len(self.cache),
            'max_cache_size': self.cache.maxsize,
            'llm_updates': self.stats['llm_updates']
        }
    
    def clear_cache(self):
        """Clear all caches."""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def get_cache_info_string(self):
        """Get formatted cache info for UI display."""
        stats = self.get_cache_stats()
        llm_info = f" | LLM Updates: {stats['llm_updates']}" if stats['llm_updates'] > 0 else ""
        return (f"Cache: {stats['cache_size']}/{stats['max_cache_size']} | "
                f"Hit Rate: {stats['hit_rate_percent']}%{llm_info}")

# Global cache manager instance
_global_cache_manager = None

def get_global_cache_manager():
    """Get global cache manager instance (singleton)."""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = InferenceCacheManager()
    return _global_cache_manager

def initialize_cache_manager(max_cache_size=DEFAULT_MAX_CACHE_SIZE):
    """Initialize global cache manager with custom settings."""
    global _global_cache_manager
    _global_cache_manager = InferenceCacheManager(max_cache_size)
    logger.info(f"Initialized global cache manager with size: {max_cache_size}") 