"""
LRU Cache Manager for Sign Language Recognition Inference.
Provides both individual image caching and sequence-based result caching.
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
DEFAULT_MAX_IMAGE_CACHE_SIZE = 300
DEFAULT_SIMILARITY_THRESHOLD = 0.95

class LRUImageCache(OrderedDict):
    """LRU cache for individual images with automatic eviction."""
    
    def __init__(self, maxsize=DEFAULT_MAX_IMAGE_CACHE_SIZE):
        self.maxsize = maxsize
        super().__init__()
        logger.info(f"Initialized LRU image cache with max size: {maxsize}")
        
    def __getitem__(self, key):
        """Get image and mark as recently used."""
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value
    
    def __setitem__(self, key, value):
        """Store image with LRU eviction."""
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        
        # Evict oldest if limit exceeded
        if len(self) > self.maxsize:
            oldest = next(iter(self))
            logger.info(f"Image cache limit reached. Removing oldest item.")
            del self[oldest]

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
    """Main cache manager with dual-tier caching: images + inference results."""
    
    def __init__(self, max_cache_size=DEFAULT_MAX_CACHE_SIZE, max_image_cache_size=DEFAULT_MAX_IMAGE_CACHE_SIZE):
        self.cache = LRUInferenceCache(max_cache_size)
        self.image_cache = LRUImageCache(max_image_cache_size)
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'image_cache_hits': 0,
            'image_cache_misses': 0
        }
        logger.info(f"Initialized inference cache manager with max size: {max_cache_size}")
        logger.info(f"Initialized image cache with max size: {max_image_cache_size}")
    
    def get_cached_image(self, image_hash):
        """Retrieve cached image by hash."""
        if image_hash in self.image_cache:
            self.stats['image_cache_hits'] += 1
            logger.info(f"Image cache hit for hash: {image_hash[:8]}...")
            return self.image_cache[image_hash]
        else:
            self.stats['image_cache_misses'] += 1
            return None
    
    def store_image(self, image_hash, image):
        """Store image in cache."""
        self.image_cache[image_hash] = image
        logger.info(f"Stored image in cache for hash: {image_hash[:8]}...")
    
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
    
    def check_and_cache_images(self, img_list):
        """
        Check and cache individual images for efficiency.
        
        Args:
            img_list: List of images to check and cache
            
        Returns:
            dict: Cache hit/miss statistics for images
        """
        image_stats = {'hits': 0, 'misses': 0, 'new_cached': 0}
        
        for img in img_list:
            img_hash = get_image_hash(img)
            
            if img_hash in self.image_cache:
                image_stats['hits'] += 1
            else:
                image_stats['misses'] += 1
                self.store_image(img_hash, img)
                image_stats['new_cached'] += 1
        
        return image_stats
    
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
    
    def get_cache_stats(self):
        """Get comprehensive cache statistics."""
        total_requests = self.stats['total_requests']
        hit_rate = (self.stats['cache_hits'] / total_requests * 100) if total_requests > 0 else 0
        
        total_image_requests = self.stats['image_cache_hits'] + self.stats['image_cache_misses']
        image_hit_rate = (self.stats['image_cache_hits'] / total_image_requests * 100) if total_image_requests > 0 else 0
        
        return {
            'total_requests': total_requests,
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'hit_rate_percent': round(hit_rate, 2),
            'cache_size': len(self.cache),
            'max_cache_size': self.cache.maxsize,
            'image_cache_hits': self.stats['image_cache_hits'],
            'image_cache_misses': self.stats['image_cache_misses'],
            'image_hit_rate_percent': round(image_hit_rate, 2),
            'image_cache_size': len(self.image_cache),
            'max_image_cache_size': self.image_cache.maxsize
        }
    
    def clear_cache(self):
        """Clear all caches."""
        self.cache.clear()
        self.image_cache.clear()
        logger.info("All caches cleared")
    
    def clear_inference_cache(self):
        """Clear inference result cache only."""
        self.cache.clear()
        logger.info("Inference cache cleared")
    
    def clear_image_cache(self):
        """Clear image cache only."""
        self.image_cache.clear()
        logger.info("Image cache cleared")
    
    def get_cache_info_string(self):
        """Get formatted cache info for UI display."""
        stats = self.get_cache_stats()
        return (f"Results Cache: {stats['cache_size']}/{stats['max_cache_size']} | "
                f"Images Cache: {stats['image_cache_size']}/{stats['max_image_cache_size']} | "
                f"Hit Rate: {stats['hit_rate_percent']}%")

# Global cache manager instance
_global_cache_manager = None

def get_global_cache_manager():
    """Get global cache manager instance (singleton)."""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = InferenceCacheManager()
    return _global_cache_manager

def initialize_cache_manager(max_cache_size=DEFAULT_MAX_CACHE_SIZE, max_image_cache_size=DEFAULT_MAX_IMAGE_CACHE_SIZE):
    """Initialize global cache manager with custom settings."""
    global _global_cache_manager
    _global_cache_manager = InferenceCacheManager(max_cache_size, max_image_cache_size)
    logger.info(f"Initialized global cache manager with size: {max_cache_size}")
    logger.info(f"Initialized global image cache with size: {max_image_cache_size}") 