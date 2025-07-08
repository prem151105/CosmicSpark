"""
MOSDAC AI Help Bot - Core Package Initialization
Sets up environment variables to prevent TensorFlow Lite delegate errors
"""

import os

# Set environment variables to prevent TensorFlow Lite delegate errors
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU usage

# Disable TensorFlow GPU memory growth to prevent conflicts
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Set threading options for better stability
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'