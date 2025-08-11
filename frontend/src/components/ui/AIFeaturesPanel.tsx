import { Box, List, ListItem, ListItemIcon, ListItemText, Typography } from '@mui/material';
import { styled } from '@mui/material/styles';
import React from 'react';

const FeaturePanel = styled(Box)({
    backgroundColor: 'white',
    borderRadius: '12px',
    padding: '24px',
    border: '1px solid #e5e7eb',
    boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.1)',
    height: 'fit-content',
});

const ProcessingIcon = styled(Box)({
    width: 64,
    height: 64,
    borderRadius: '50%',
    backgroundColor: '#f3f4f6',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '24px',
    color: '#6b7280',
    margin: '0 auto 24px auto',
});

const FeatureIcon = styled(Box)({
    width: 6,
    height: 6,
    borderRadius: '50%',
    backgroundColor: '#3b82f6',
});

const features = [
    'Redes neuronales avanzadas para eliminación precisa del fondo',
    'Múltiples modelos de IA optimizados para diferentes casos de uso',
    'Salida PNG de alta calidad con transparencia',
    'Funciona con retratos, productos y escenas complejas'
];

export const AIFeaturesPanel: React.FC = () => {
    return (
        <Box>
            {/* Processing Results Section */}
            <FeaturePanel sx={{ mb: 3 }}>
                <Box
                    sx={{
                        border: '2px dashed #d1d5db',
                        borderRadius: '12px',
                        padding: '32px 24px',
                        textAlign: 'center',
                        backgroundColor: '#f9fafb',
                        minHeight: '200px',
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        justifyContent: 'center',
                        gap: 2,
                        transition: 'all 0.2s ease-in-out',
                        '&:hover': {
                            backgroundColor: '#f1f5f9',
                            borderColor: '#9ca3af',
                        },
                    }}
                >
                    <ProcessingIcon>
                        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"></path>
                            <path d="M14 2v4a2 2 0 0 0 2 2h4"></path>
                            <circle cx="10" cy="12" r="2"></circle>
                            <path d="m20 17-1.296-1.296a2.41 2.41 0 0 0-3.408 0L9 22"></path>
                        </svg>
                    </ProcessingIcon>
                    <Typography
                        variant="h6"
                        sx={{
                            fontWeight: 600,
                            color: '#6b7280',
                            mb: 1,
                        }}
                    >
                        Aquí podrás ver tu imagen editada
                    </Typography>
                </Box>
            </FeaturePanel>

            {/* AI Features Section */}
            <FeaturePanel>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1.5 }}>
                    <Box sx={{ fontSize: '20px' }}>✨</Box>
                    <Typography variant="subtitle1" sx={{ fontWeight: 600, color: '#374151' }}>
                        Características Potenciadas por IA
                    </Typography>
                </Box>

                <List sx={{ p: 0 }}>
                    {features.map((feature, index) => (
                        <ListItem key={index} sx={{ px: 0, py: 0.5, alignItems: 'center' }}>
                            <ListItemIcon sx={{ minWidth: 20 }}>
                                <FeatureIcon />
                            </ListItemIcon>
                            <ListItemText
                                primary={feature}
                                primaryTypographyProps={{
                                    variant: 'body2',
                                    sx: { color: '#4b5563', lineHeight: 1.5 },
                                }}
                            />
                        </ListItem>
                    ))}
                </List>
            </FeaturePanel>
        </Box>
    );
}; 