import { Box, CircularProgress, Modal, Typography } from '@mui/material';
import { styled } from '@mui/material/styles';
import React from 'react';
import { tokens } from '../../utils/designTokens';

interface LoadingOverlayProps {
    open: boolean;
    message?: string;
    progress?: number;
}

const ModalBackdrop = styled(Box)({
    position: 'fixed',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    background: tokens.elementStyling.loadingOverlay.backdrop.background,
    backdropFilter: tokens.elementStyling.loadingOverlay.backdrop.backdropFilter,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 9999,
});

const LoadingModal = styled(Box)({
    background: tokens.elementStyling.loadingOverlay.modal.background,
    borderRadius: tokens.elementStyling.loadingOverlay.modal.borderRadius,
    boxShadow: tokens.elementStyling.loadingOverlay.modal.shadow,
    padding: tokens.elementStyling.loadingOverlay.modal.padding,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: tokens.spacing.lg,
    minWidth: '300px',
    maxWidth: '400px',
    textAlign: 'center',
});

const StyledCircularProgress = styled(CircularProgress)({
    color: tokens.colorPalette.primary.blue500,
    '& .MuiCircularProgress-circle': {
        strokeLinecap: 'round',
    },
});

const ProgressBar = styled(Box)({
    width: '100%',
    height: '8px',
    background: tokens.elementStyling.loadingOverlay.progressBar.background,
    borderRadius: tokens.elementStyling.loadingOverlay.progressBar.borderRadius,
    overflow: 'hidden',
    position: 'relative',
});

const ProgressFill = styled(Box)<{ progress: number }>(({ progress }) => ({
    height: '100%',
    background: tokens.elementStyling.loadingOverlay.progressBar.fill,
    borderRadius: tokens.elementStyling.loadingOverlay.progressBar.borderRadius,
    width: `${progress}%`,
    transition: 'width 0.3s ease',
}));

export const LoadingOverlay: React.FC<LoadingOverlayProps> = ({
    open,
    message = 'Processing image...',
    progress,
}) => {
    return (
        <Modal
            open={open}
            disableEscapeKeyDown
            onClose={() => { }} // Prevent closing
        >
            <ModalBackdrop>
                <LoadingModal>
                    <StyledCircularProgress size={60} thickness={4} />

                    <Typography variant="h4" color="textPrimary">
                        {message}
                    </Typography>

                    {progress !== undefined && (
                        <Box width="100%">
                            <ProgressBar>
                                <ProgressFill progress={progress} />
                            </ProgressBar>
                            <Typography variant="caption" color="textSecondary" sx={{ mt: 1 }}>
                                {Math.round(progress)}% complete
                            </Typography>
                        </Box>
                    )}

                    <Typography variant="body2" color="textSecondary">
                        This may take a few seconds...
                    </Typography>
                </LoadingModal>
            </ModalBackdrop>
        </Modal>
    );
}; 