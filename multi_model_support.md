# Multi-Model Support for Background Removal

## Overview

The background removal application now supports multiple model types with different input requirements and capabilities. The system automatically detects the model type and applies the appropriate preprocessing.

## Supported Model Types

### 1. U2-Net (u2net)
- **Input Resolution**: 320x320 pixels
- **Normalization**: ImageNet standard (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Use Case**: Good balance between speed and quality
- **Model Files**: Files containing "u2net" in the name
- **Max Image Optimization**: 2000px

### 2. U2-NetP (u2netp)
- **Input Resolution**: 320x320 pixels  
- **Normalization**: ImageNet standard (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Use Case**: Lighter version for faster processing
- **Model Files**: Files containing "u2netp", "small", or "lite" in the name
- **Max Image Optimization**: 2000px

### 3. IS-Net (isnet)
- **Input Resolution**: 1024x1024 pixels
- **Normalization**: Centered (mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0])
- **Use Case**: Highest quality results, best for detailed backgrounds
- **Model Files**: Files containing "isnet" or "general-use" in the name
- **Max Image Optimization**: 3000px

## Model Detection

The system automatically detects model types based on filename patterns:

```python
def detect_model_type(model_path: str) -> str:
    model_name = os.path.basename(model_path).lower()
    
    if 'isnet' in model_name or 'general-use' in model_name:
        return 'isnet'
    elif 'u2net' in model_name:
        if 'u2netp' in model_name or 'small' in model_name or 'lite' in model_name:
            return 'u2netp'
        else:
            return 'u2net'
    else:
        return 'u2net'  # default fallback
```

## Key Features

### Automatic Preprocessing
- Each model type uses its own preprocessing pipeline
- Correct input resolution is applied automatically
- Appropriate normalization values are used

### Intelligent Image Optimization
- IS-Net models: Images larger than 3000px are optimized
- U2-Net models: Images larger than 2000px are optimized
- Aspect ratio is always preserved

### Enhanced UI Information
- Model type is displayed in the model selector
- Input resolution is shown for each model
- Processing information includes model type and input size

## Usage Examples

### Model Filename Examples
- `u2net.pth` → Detected as 'u2net'
- `u2netp.pth` → Detected as 'u2netp'  
- `isnet-general-use.pth` → Detected as 'isnet'
- `u2net_small.onnx` → Detected as 'u2netp'

### Processing Information
The system now logs detailed information:
- Model type used
- Input resolution applied
- Processing time
- Image optimization details

## Performance Considerations

### IS-Net Models
- **Pros**: Highest quality results, better fine detail preservation
- **Cons**: Slower processing due to 1024x1024 input resolution
- **Best for**: High-quality images, complex backgrounds, detailed subjects

### U2-Net Models  
- **Pros**: Good balance of speed and quality
- **Cons**: Lower resolution than IS-Net
- **Best for**: General-purpose background removal, real-time applications

### U2-NetP Models
- **Pros**: Fastest processing
- **Cons**: Lower quality than full U2-Net
- **Best for**: Mobile applications, resource-constrained environments

## Troubleshooting

### Model Not Detected Correctly
- Ensure model filename contains recognizable keywords
- Check the model detection logic if using custom names
- Default fallback is U2-Net if detection fails

### Poor Results
- Try a different model type (IS-Net for higher quality)
- Ensure image quality is sufficient for the model
- Check if image optimization is affecting results

### Performance Issues
- Use U2-NetP for faster processing
- Enable image optimization for large images
- Consider model type based on your quality requirements

## Integration Notes

### For Developers
- Model type is stored in the model_info dictionary
- Preprocessing functions are model-type aware
- Error handling includes model type information
- Logging captures model-specific metadata

### API Changes
- `process_image()` now takes `model_info` instead of `session`
- `load_all_models()` returns enhanced model information
- Model selection UI shows type and resolution information

## Future Enhancements

- Support for additional model architectures
- Custom model type configuration
- Performance benchmarking per model type
- Batch processing with mixed model types