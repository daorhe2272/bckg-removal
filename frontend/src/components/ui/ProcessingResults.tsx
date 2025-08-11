import { Box, Button, Typography } from '@mui/material';
import { styled } from '@mui/material/styles';
import React from 'react';
import { downloadImage } from '../../services/api';

interface ProcessingResultsProps {
    originalImage: File | null;
    processedImage: string;
}

const ResultContainer = styled(Box)({
    marginTop: '32px',
    padding: '24px',
    backgroundColor: 'white',
    borderRadius: '12px',
    border: '1px solid #e5e7eb',
    boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.1)',
    width: '100%',
});

const ImageContainer = styled(Box)({
    position: 'relative',
    borderRadius: '8px',
    overflow: 'hidden',
    border: '2px solid #e5e7eb',
    backgroundColor: 'white',
    width: '100%',
});

const TransparencyPattern = styled(Box)({
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    background: `
    linear-gradient(45deg, #ccc 25%, transparent 25%),
    linear-gradient(-45deg, #ccc 25%, transparent 25%),
    linear-gradient(45deg, transparent 75%, #ccc 75%),
    linear-gradient(-45deg, transparent 75%, #ccc 75%)
  `,
    backgroundSize: '20px 20px',
    backgroundPosition: '0 0, 0 10px, 10px -10px, -10px 0px',
    opacity: 0.1,
});

const DownloadButton = styled(Button)({
    background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
    color: 'white',
    fontWeight: 600,
    padding: '14px 28px',
    borderRadius: '8px',
    textTransform: 'none',
    fontSize: '16px',
    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
    '&:hover': {
        background: 'linear-gradient(135deg, #059669 0%, #047857 100%)',
        transform: 'translateY(-1px)',
        boxShadow: '0 6px 8px -1px rgba(0, 0, 0, 0.15)',
    },
});

export const ProcessingResults: React.FC<ProcessingResultsProps> = ({
    originalImage,
    processedImage,
}) => {
    const handleDownload = () => {
        if (originalImage) {
            const filename = originalImage.name.replace(/\.[^/.]+$/, '_sin_fondo.png');
            downloadImage(processedImage, filename);
        }
    };

    return (
        <ResultContainer>
            <Typography variant="h6" sx={{ fontWeight: 600, color: '#374151', mb: 3 }}>
                Resultados del Procesamiento
            </Typography>

            <Box sx={{
                display: 'flex',
                gap: { xs: 2, md: 3 },
                flexDirection: { xs: 'column', md: 'row' },
                width: '100%'
            }}>
                {/* Original Image */}
                <Box sx={{ flex: 1, width: '100%' }}>
                    <Typography variant="subtitle2" sx={{ fontWeight: 600, color: '#6b7280', mb: 1 }}>
                        Imagen Original
                    </Typography>
                    <ImageContainer>
                        {originalImage && (
                            <img
                                src={URL.createObjectURL(originalImage)}
                                alt="Original"
                                style={{
                                    width: '100%',
                                    height: '220px',
                                    objectFit: 'cover',
                                }}
                            />
                        )}
                    </ImageContainer>
                </Box>

                {/* Processed Image */}
                <Box sx={{ flex: 1, width: '100%' }}>
                    <Typography variant="subtitle2" sx={{ fontWeight: 600, color: '#6b7280', mb: 1 }}>
                        Fondo Removido
                    </Typography>
                    <ImageContainer>
                        <TransparencyPattern />
                        <img
                            src={processedImage}
                            alt="Processed"
                            style={{
                                width: '100%',
                                height: '220px',
                                objectFit: 'cover',
                                position: 'relative',
                                zIndex: 1,
                            }}
                        />
                    </ImageContainer>
                </Box>
            </Box>

            <Box sx={{ mt: 4, textAlign: 'center' }}>
                <DownloadButton onClick={handleDownload}>
                    📥 Descargar Imagen Procesada
                </DownloadButton>
            </Box>
        </ResultContainer>
    );
}; 