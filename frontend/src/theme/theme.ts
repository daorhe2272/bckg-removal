import { createTheme } from '@mui/material/styles';
import { tokens } from '../utils/designTokens';

// Create Material-UI theme based on our design tokens
export const theme = createTheme({
    palette: {
        primary: {
            main: tokens.colorPalette.primary.blue500,
            dark: tokens.colorPalette.primary.blue600,
            light: tokens.colorPalette.primary.blue100,
            contrastText: tokens.colorPalette.neutral.white,
        },
        secondary: {
            main: tokens.colorPalette.secondary.indigo500,
            dark: tokens.colorPalette.secondary.indigo600,
            light: tokens.colorPalette.secondary.indigo50,
            contrastText: tokens.colorPalette.neutral.white,
        },
        success: {
            main: tokens.colorPalette.status.success,
            light: tokens.colorPalette.status.successBg,
            contrastText: tokens.colorPalette.neutral.white,
        },
        warning: {
            main: tokens.colorPalette.status.warning,
            light: tokens.colorPalette.status.warningBg,
            contrastText: tokens.colorPalette.neutral.white,
        },
        error: {
            main: tokens.colorPalette.status.error,
            light: tokens.colorPalette.status.errorBg,
            contrastText: tokens.colorPalette.neutral.white,
        },
        grey: {
            50: tokens.colorPalette.neutral.gray50,
            100: tokens.colorPalette.neutral.gray100,
            200: tokens.colorPalette.neutral.gray200,
            300: tokens.colorPalette.neutral.gray300,
            400: tokens.colorPalette.neutral.gray400,
            500: tokens.colorPalette.neutral.gray500,
            600: tokens.colorPalette.neutral.gray600,
            700: tokens.colorPalette.neutral.gray700,
            800: tokens.colorPalette.neutral.gray800,
            900: tokens.colorPalette.neutral.gray900,
        },
        background: {
            default: tokens.colorPalette.neutral.gray50,
            paper: tokens.colorPalette.neutral.white,
        },
        text: {
            primary: tokens.typography.headings.h1.color,
            secondary: tokens.typography.body.color,
        },
    },
    typography: {
        fontFamily: tokens.typography.fontFamily,
        h1: {
            fontSize: tokens.typography.headings.h1.fontSize,
            fontWeight: tokens.typography.headings.h1.fontWeight,
            lineHeight: tokens.typography.headings.h1.lineHeight,
            color: tokens.typography.headings.h1.color,
        },
        h2: {
            fontSize: tokens.typography.headings.h2.fontSize,
            fontWeight: tokens.typography.headings.h2.fontWeight,
            lineHeight: tokens.typography.headings.h2.lineHeight,
            color: tokens.typography.headings.h2.color,
        },
        h3: {
            fontSize: tokens.typography.headings.h3.fontSize,
            fontWeight: tokens.typography.headings.h3.fontWeight,
            lineHeight: tokens.typography.headings.h3.lineHeight,
            color: tokens.typography.headings.h3.color,
        },
        h4: {
            fontSize: tokens.typography.headings.h4.fontSize,
            fontWeight: tokens.typography.headings.h4.fontWeight,
            lineHeight: tokens.typography.headings.h4.lineHeight,
            color: tokens.typography.headings.h4.color,
        },
        body1: {
            fontSize: tokens.typography.body.fontSize,
            fontWeight: tokens.typography.body.fontWeight,
            lineHeight: tokens.typography.body.lineHeight,
            color: tokens.typography.body.color,
        },
        caption: {
            fontSize: tokens.typography.caption.fontSize,
            fontWeight: tokens.typography.caption.fontWeight,
            lineHeight: tokens.typography.caption.lineHeight,
            color: tokens.typography.caption.color,
        },
    },
    shape: {
        borderRadius: parseFloat(tokens.borderRadius.medium),
    },
    spacing: 8, // 8px base unit (standard Material-UI spacing)
    shadows: [
        'none',
        tokens.shadows.card,
        tokens.shadows.cardHover,
        tokens.shadows.button,
        tokens.shadows.buttonHover,
        tokens.shadows.modal,
        tokens.shadows.header,
        // Add more shadows as needed
        '0 8px 10px -5px rgba(0,0,0,0.2)',
        '0 16px 24px 2px rgba(0,0,0,0.14)',
        '0 6px 30px 5px rgba(0,0,0,0.12)',
        // Fill remaining shadow array to meet Material-UI requirements
        ...Array(15).fill('0 8px 10px -5px rgba(0,0,0,0.2)'),
    ] as any,
    components: {
        // Customize Material-UI components to match our design system
        MuiButton: {
            styleOverrides: {
                root: {
                    textTransform: 'none', // Disable uppercase transform
                    borderRadius: tokens.borderRadius.large,
                    fontWeight: 600,
                    transition: tokens.animations.transitions.all,
                },
                contained: {
                    background: tokens.gradients.primaryButton,
                    boxShadow: tokens.shadows.button,
                    '&:hover': {
                        background: tokens.gradients.primaryButtonHover,
                        boxShadow: tokens.shadows.buttonHover,
                        transform: 'translateY(-2px)',
                    },
                    '&:disabled': {
                        background: tokens.colorPalette.neutral.gray400,
                        color: tokens.colorPalette.neutral.white,
                    },
                },
            },
        },
        MuiCard: {
            styleOverrides: {
                root: {
                    borderRadius: tokens.borderRadius.large,
                    border: `1px solid ${tokens.colorPalette.neutral.gray200}`,
                    boxShadow: tokens.shadows.card,
                    '&:hover': {
                        boxShadow: tokens.shadows.cardHover,
                    },
                },
            },
        },
        MuiPaper: {
            styleOverrides: {
                root: {
                    borderRadius: tokens.borderRadius.large,
                },
            },
        },
    },
}); 