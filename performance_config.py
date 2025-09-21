# Performance Configuration for Different Scales
# Adjust these settings based on your data size and requirements

import os

# Environment-based configurations
DEPLOYMENT_MODE = os.getenv('DEPLOYMENT_MODE', 'development')  # development, production, enterprise

# Configuration profiles
CONFIGS = {
    'development': {
        'batch_size': 100,
        'max_workers': 4,
        'chunk_size': 500,
        'embedding_cache_size': 500,
        'sample_sizes': {
            'small_files': None,      # Process all records
            'medium_files': 1000,     # Sample 1K records
            'large_files': 2000,      # Sample 2K records
        },
        'milvus_params': {
            'pool_size': 10,
            'nlist': 1024,
            'nprobe': 16
        }
    },
    
    'production': {
        'batch_size': 500,
        'max_workers': 8,
        'chunk_size': 1000,
        'embedding_cache_size': 1000,
        'sample_sizes': {
            'small_files': None,      # Process all records
            'medium_files': 5000,     # Sample 5K records
            'large_files': 10000,     # Sample 10K records
        },
        'milvus_params': {
            'pool_size': 20,
            'nlist': 2048,
            'nprobe': 32
        }
    },
    
    'enterprise': {
        'batch_size': 1000,
        'max_workers': 16,
        'chunk_size': 2000,
        'embedding_cache_size': 2000,
        'sample_sizes': {
            'small_files': None,      # Process all records
            'medium_files': None,     # Process all records
            'large_files': 50000,     # Sample 50K records for very large files
        },
        'milvus_params': {
            'pool_size': 50,
            'nlist': 4096,
            'nprobe': 64
        }
    }
}

# Get current configuration
def get_config():
    return CONFIGS.get(DEPLOYMENT_MODE, CONFIGS['development'])

# File size categorization thresholds (number of records)
SIZE_THRESHOLDS = {
    'small': 1000,      # < 1K records
    'medium': 10000,    # 1K - 10K records  
    'large': 100000,    # 10K - 100K records
    'xlarge': float('inf')  # > 100K records
}

def categorize_file_size(record_count):
    """Categorize file size based on record count"""
    for size, threshold in SIZE_THRESHOLDS.items():
        if record_count < threshold:
            return size
    return 'xlarge'

def get_sample_size(record_count):
    """Get appropriate sample size based on file size and deployment mode"""
    config = get_config()
    file_category = categorize_file_size(record_count)
    
    if file_category == 'small':
        return config['sample_sizes']['small_files']
    elif file_category == 'medium':
        return config['sample_sizes']['medium_files']
    else:
        return config['sample_sizes']['large_files']

# Performance monitoring
def log_performance_config():
    config = get_config()
    print(f"""
🚀 Performance Configuration - {DEPLOYMENT_MODE.upper()} Mode
================================================================
Batch Size: {config['batch_size']}
Max Workers: {config['max_workers']} 
Chunk Size: {config['chunk_size']}
Cache Size: {config['embedding_cache_size']}
Milvus Pool: {config['milvus_params']['pool_size']}
================================================================
    """)

# Scaling recommendations
SCALING_GUIDE = """
📊 SCALING RECOMMENDATIONS:

🔹 DEVELOPMENT (< 50K records):
   • Fast prototyping with sampling
   • 4 workers, 100 batch size
   • ~2-5 minutes processing

🔹 PRODUCTION (50K - 500K records):
   • Balanced speed/accuracy 
   • 8 workers, 500 batch size
   • ~5-15 minutes processing

🔹 ENTERPRISE (500K+ records):
   • Maximum performance
   • 16 workers, 1000 batch size
   • ~10-30 minutes processing
   • Consider distributed processing

💡 TO SCALE UP:
   1. Set DEPLOYMENT_MODE environment variable
   2. Increase Milvus resources (CPU/Memory)
   3. Use SSD storage for better I/O
   4. Consider horizontal scaling with multiple Milvus nodes
"""

if __name__ == "__main__":
    log_performance_config()
    print(SCALING_GUIDE)