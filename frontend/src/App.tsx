import { Box, CssBaseline, GlobalStyles, Typography } from '@mui/material';
import { ThemeProvider } from '@mui/material/styles';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useState } from 'react';
import { AIFeaturesPanel } from './components/ui/AIFeaturesPanel';
import { ImageUploadArea } from './components/ui/ImageUploadArea';
import { ModelSelector } from './components/ui/ModelSelector';
import { ProcessingResults } from './components/ui/ProcessingResults';
import { apiService } from './services/api';
import { theme } from './theme/theme';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000,
      retry: 1,
    },
  },
});

// Global styles to ensure full width
const globalStyles = (
  <GlobalStyles
    styles={{
      'html, body': {
        margin: 0,
        padding: 0,
        width: '100%',
        height: '100%',
        overflowX: 'hidden',
      },
      '#root': {
        margin: 0,
        padding: 0,
        width: '100%',
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'column',
      },
      '*': {
        boxSizing: 'border-box',
      },
    }}
  />
);

function App() {
  const [uploadedImage, setUploadedImage] = useState<File | null>(null);
  const [processedImage, setProcessedImage] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string>('u2net');
  const [models, setModels] = useState<any[]>([]);
  const [functionsHealthy, setFunctionsHealthy] = useState<boolean | null>(null);

  // Simple fetchers (no react-query for brevity)
  if (models.length === 0) {
    apiService.getModels().then((res) => setModels(res.models)).catch(() => { });
  }
  if (functionsHealthy === null) {
    apiService.getProcessingStatus().then((res) => setFunctionsHealthy(res.functions_available)).catch(() => setFunctionsHealthy(false));
  }

  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        {globalStyles}
        <Box
          sx={{
            width: '100%',
            minHeight: '100vh',
            backgroundColor: '#f8fafc',
            fontFamily: theme.typography.fontFamily,
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          {/* Header - Shorter */}
          <Box
            sx={{
              width: '100%',
              backgroundColor: 'white',
              borderBottom: '1px solid #e2e8f0',
              boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.1)',
              flexShrink: 0,
            }}
          >
            <Box
              sx={{
                py: 1.5,
                px: '5%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                width: '100%',
              }}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <Box
                  sx={{
                    width: 40,
                    height: 40,
                    borderRadius: '8px',
                    background: 'linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '20px',
                  }}
                >
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"></path>
                    <path d="M5 3v4"></path>
                    <path d="M19 17v4"></path>
                    <path d="M3 5h4"></path>
                    <path d="M17 19h4"></path>
                  </svg>
                </Box>
                <Box>
                  <Typography variant="h6" sx={{ fontWeight: 600, color: '#1e293b' }}>
                    Fondastic
                  </Typography>
                  <Typography variant="caption" sx={{ color: '#64748b' }}>
                    Eliminación de Fondo con IA
                  </Typography>
                </Box>
              </Box>

              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                {/* Listo Section - Separate */}
                <Box
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 1,
                  }}
                >
                  <Box
                    sx={{
                      width: 8,
                      height: 8,
                      borderRadius: '50%',
                      backgroundColor: '#10b981',
                    }}
                  />
                  <Typography variant="body2" sx={{ color: '#0369a1', fontWeight: 500 }}>
                    Listo
                  </Typography>
                </Box>

                {/* Potenciado por IA Section - Separate */}
                <Box
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 1,
                    backgroundColor: '#eff6ff',
                    px: 2,
                    py: 1,
                    borderRadius: '20px',
                  }}
                >
                  <Box sx={{ color: '#3b82f6' }}>⚡</Box>
                  <Typography variant="body2" sx={{ color: '#3b82f6', fontWeight: 600 }}>
                    Potenciado por IA
                  </Typography>
                </Box>
              </Box>
            </Box>
          </Box>

          {/* Main Content - Percentage-based Width */}
          <Box sx={{ width: '100%', py: 4, flex: 1 }}>
            <Box
              sx={{
                display: 'flex',
                gap: 4,
                flexDirection: { xs: 'column', lg: 'row' },
                width: '85%',
                maxWidth: '1400px',
                mx: 'auto',
              }}
            >
              {/* Left Column - Upload Area */}
              <Box sx={{ flex: 1, minWidth: 0 }}>
                <Box
                  sx={{
                    backgroundColor: 'white',
                    borderRadius: '12px',
                    padding: '24px',
                    boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.1)',
                  }}
                >
                  <Typography variant="h4" sx={{ mb: 3, fontWeight: 600, color: '#1e293b' }}>
                    Subir Imagen
                  </Typography>

                  <ModelSelector
                    models={models}
                    selectedModel={selectedModel}
                    onModelChange={setSelectedModel}
                  />

                  <ImageUploadArea
                    onImageSelect={setUploadedImage}
                    selectedImage={uploadedImage}
                    isProcessing={isProcessing}
                    onProcessingStart={() => setIsProcessing(true)}
                    onProcessingComplete={(result: string) => {
                      setProcessedImage(result);
                      setIsProcessing(false);
                    }}
                    selectedModel={selectedModel}
                  />

                  {processedImage && (
                    <ProcessingResults
                      originalImage={uploadedImage}
                      processedImage={processedImage}
                    />
                  )}
                </Box>
              </Box>

              {/* Right Column - AI Features Panel */}
              <Box sx={{
                width: { xs: '100%', lg: '380px' },
                flexShrink: 0,
                maxWidth: { xs: '100%', lg: '380px' }
              }}>
                <Box sx={{ mb: 2, p: 2, borderRadius: '8px', border: '1px solid #e5e7eb', backgroundColor: 'white' }} data-cy="functions-status">
                  <Typography variant="subtitle2" sx={{ fontWeight: 600, color: '#374151', mb: 1 }}>
                    Estado del Servicio de Procesamiento
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Box sx={{ width: 8, height: 8, borderRadius: '50%', backgroundColor: functionsHealthy ? '#10b981' : '#ef4444' }} />
                    <Typography variant="body2" sx={{ color: functionsHealthy ? '#065f46' : '#991b1b' }}>
                      {functionsHealthy ? 'Disponible' : 'No disponible'}
                    </Typography>
                  </Box>
                </Box>

                <AIFeaturesPanel />
              </Box>
            </Box>
          </Box>
        </Box>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

export default App;
