import { Box, Button, CircularProgress, Typography } from '@mui/material';
import { styled } from '@mui/material/styles';
import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { apiService } from '../../services/api';
import { MAX_FILE_SIZE, SUPPORTED_FORMATS } from '../../types/api';

interface ImageUploadAreaProps {
    onImageSelect: (file: File | null) => void;
    selectedImage: File | null;
    isProcessing: boolean;
    onProcessingStart: () => void;
    onProcessingComplete: (result: string) => void;
    selectedModel?: string;
}

const DropzoneContainer = styled(Box)<{ isDragActive: boolean; hasImage: boolean }>(({ isDragActive, hasImage }) => ({
    border: `2px dashed ${isDragActive ? '#3b82f6' : '#d1d5db'}`,
    borderRadius: '12px',
    padding: hasImage ? '0' : '48px 24px',
    textAlign: 'center',
    cursor: 'pointer',
    backgroundColor: isDragActive ? '#f8fafc' : '#f9fafb',
    transition: 'all 0.2s ease-in-out',
    minHeight: '320px',
    width: '100%',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 2,
    boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.1)',
    '&:hover': {
        borderColor: '#9ca3af',
        backgroundColor: isDragActive ? '#f1f5f9' : '#f1f5f9',
    },
}));

const ImageIcon = styled(Box)<{ isDragActive: boolean }>(({ isDragActive }) => ({
    width: 64,
    height: 64,
    borderRadius: '50%',
    backgroundColor: isDragActive ? '#dbeafe' : '#e5e7eb',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '24px',
    color: isDragActive ? '#2563eb' : '#6b7280',
    marginBottom: '16px',
}));

const ProcessButton = styled(Button)({
    marginTop: '20px',
    background: 'linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)',
    color: 'white',
    fontWeight: 600,
    padding: '14px 32px',
    borderRadius: '8px',
    textTransform: 'none',
    fontSize: '16px',
    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
    '&:hover': {
        background: 'linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%)',
        transform: 'translateY(-1px)',
        boxShadow: '0 6px 8px -1px rgba(0, 0, 0, 0.15)',
    },
    '&:disabled': {
        background: '#9ca3af',
        color: 'white',
        transform: 'none',
        boxShadow: 'none',
    },
});

export const ImageUploadArea: React.FC<ImageUploadAreaProps> = ({
    onImageSelect,
    selectedImage,
    isProcessing,
    onProcessingStart,
    onProcessingComplete,
    selectedModel = 'u2net',
}) => {
    const [error, setError] = useState<string | null>(null);

    const onDrop = useCallback(
        (acceptedFiles: File[]) => {
            const file = acceptedFiles[0];
            if (file) {
                // Validate file type
                if (!SUPPORTED_FORMATS.includes(file.type as typeof SUPPORTED_FORMATS[number])) {
                    setError('Formato no soportado. Use JPG, PNG, WEBP o BMP.');
                    return;
                }

                // Validate file size
                if (file.size > MAX_FILE_SIZE) {
                    setError('Archivo demasiado grande. Máximo 10MB.');
                    return;
                }

                setError(null);
                onImageSelect(file);
            }
        },
        [onImageSelect]
    );

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: {
            'image/*': ['.jpeg', '.jpg', '.png', '.webp', '.bmp'],
        },
        multiple: false,
    });

    const handleProcess = async () => {
        if (!selectedImage) return;

        try {
            onProcessingStart();
            const result = await apiService.processImage(selectedImage, selectedModel);
            onProcessingComplete(result.result_image);
        } catch (error) {
            console.error('Processing failed:', error);
            setError('Error al procesar la imagen. Inténtelo de nuevo.');
        }
    };

    return (
        <Box sx={{ width: '100%' }}>
            <DropzoneContainer {...getRootProps()} isDragActive={isDragActive} hasImage={!!selectedImage} data-cy="dropzone">
                <input {...getInputProps()} />

                {selectedImage ? (
                    // Show uploaded image
                    <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2, width: '100%', height: '100%', position: 'relative' }}>
                        {/* Close button */}
                        <Box
                            onClick={(e) => {
                                e.stopPropagation();
                                onImageSelect(null);
                                setError(null);
                            }}
                            sx={{
                                position: 'absolute',
                                top: 8,
                                right: 8,
                                width: 32,
                                height: 32,
                                borderRadius: '50%',
                                backgroundColor: 'white',
                                boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                cursor: 'pointer',
                                zIndex: 1,
                                border: '1px solid #e5e7eb',
                                '&:hover': {
                                    backgroundColor: '#f9fafb',
                                    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.15)',
                                },
                            }}
                        >
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#6b7280" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M18 6L6 18"></path>
                                <path d="M6 6l12 12"></path>
                            </svg>
                        </Box>

                        <img
                            src={URL.createObjectURL(selectedImage)}
                            alt="Uploaded"
                            style={{
                                maxWidth: '100%',
                                maxHeight: '280px',
                                width: 'auto',
                                height: 'auto',
                                objectFit: 'contain',
                                borderRadius: '8px',
                                flex: 1,
                            }}
                        />
                        <Box sx={{ textAlign: 'center', flexShrink: 0 }}>
                            <Typography variant="subtitle2" sx={{ fontWeight: 600, color: '#374151' }}>
                                {selectedImage.name}
                            </Typography>
                            <Typography variant="caption" sx={{ color: '#6b7280' }}>
                                {(selectedImage.size / 1024 / 1024).toFixed(2)} MB
                            </Typography>
                        </Box>
                    </Box>
                ) : (
                    // Show placeholder
                    <>
                        <ImageIcon isDragActive={isDragActive}>
                            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <rect width="18" height="18" x="3" y="3" rx="2" ry="2"></rect>
                                <circle cx="9" cy="9" r="2"></circle>
                                <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"></path>
                            </svg>
                        </ImageIcon>

                        <Typography variant="h6" sx={{ fontWeight: 600, color: '#374151', mb: 1 }}>
                            Subir una imagen
                        </Typography>

                        <Typography variant="body2" sx={{ color: '#6b7280', mb: 1, maxWidth: '400px' }}>
                            Arrastra y suelta tu imagen aquí, o haz clic para seleccionar
                        </Typography>

                        <Typography variant="caption" sx={{ color: '#9ca3af' }}>
                            Formatos: JPG, PNG, WEBP (Máx: 10MB)
                        </Typography>
                    </>
                )}

                {error && (
                    <Box
                        sx={{
                            mt: 2,
                            p: 2,
                            backgroundColor: '#fef2f2',
                            border: '1px solid #fecaca',
                            borderRadius: '8px',
                            color: '#991b1b',
                            maxWidth: '400px',
                            width: '100%',
                        }}
                    >
                        <Typography variant="body2">{error}</Typography>
                    </Box>
                )}
            </DropzoneContainer>

            {selectedImage && (
                <ProcessButton
                    onClick={handleProcess}
                    disabled={isProcessing}
                    fullWidth
                    sx={{ mt: 3 }}
                    data-cy="process-button"
                >
                    {isProcessing ? (
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <CircularProgress size={20} sx={{ color: 'white' }} />
                            Procesando...
                        </Box>
                    ) : (
                        'Procesar Imagen'
                    )}
                </ProcessButton>
            )}
        </Box>
    );
}; 