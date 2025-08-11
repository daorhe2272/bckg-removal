import { Box, Button, Card } from '@mui/material';
import { styled } from '@mui/material/styles';
import { tokens } from '../../utils/designTokens';

// Primary Button with gradient and hover effects
export const PrimaryButton = styled(Button)({
    background: tokens.gradients.primaryButton,
    color: tokens.elementStyling.buttons.primary.text,
    borderRadius: tokens.elementStyling.buttons.primary.borderRadius,
    boxShadow: tokens.elementStyling.buttons.primary.shadow,
    padding: tokens.elementStyling.buttons.primary.padding,
    fontSize: tokens.elementStyling.buttons.primary.fontSize,
    fontWeight: tokens.elementStyling.buttons.primary.fontWeight,
    border: 'none',
    textTransform: 'none',
    transition: tokens.animations.transitions.all,

    '&:hover': {
        background: tokens.gradients.primaryButtonHover,
        boxShadow: tokens.elementStyling.buttons.primary.hover.shadow,
        transform: tokens.elementStyling.buttons.primary.hover.transform,
    },

    '&:disabled': {
        background: tokens.elementStyling.buttons.primary.disabled.background,
        cursor: tokens.elementStyling.buttons.primary.disabled.cursor,
        boxShadow: tokens.elementStyling.buttons.primary.disabled.shadow,
        transform: tokens.elementStyling.buttons.primary.disabled.transform,
    },
});

// Secondary Button
export const SecondaryButton = styled(Button)({
    background: tokens.elementStyling.buttons.secondary.background,
    color: tokens.elementStyling.buttons.secondary.text,
    borderRadius: tokens.elementStyling.buttons.secondary.borderRadius,
    padding: tokens.elementStyling.buttons.secondary.padding,
    fontSize: tokens.elementStyling.buttons.secondary.fontSize,
    fontWeight: tokens.elementStyling.buttons.secondary.fontWeight,
    border: 'none',
    textTransform: 'none',
    transition: tokens.animations.transitions.all,

    '&:hover': {
        background: tokens.elementStyling.buttons.secondary.hover.background,
    },
});

// Icon Button
export const IconButton = styled(Button)({
    background: tokens.elementStyling.buttons.icon.background,
    border: tokens.elementStyling.buttons.icon.border,
    borderRadius: tokens.elementStyling.buttons.icon.borderRadius,
    padding: tokens.elementStyling.buttons.icon.padding,
    minWidth: 'auto',
    color: tokens.elementStyling.buttons.icon.iconColor,
    transition: tokens.animations.transitions.all,

    '&:hover': {
        background: tokens.elementStyling.buttons.icon.hover.background,
        border: tokens.elementStyling.buttons.icon.hover.border,
    },
});

// Styled Card with design system tokens
export const StyledCard = styled(Card)({
    background: tokens.elementStyling.cards.background,
    border: tokens.elementStyling.cards.border,
    borderRadius: tokens.elementStyling.cards.borderRadius,
    boxShadow: tokens.elementStyling.cards.shadow,
    padding: tokens.elementStyling.cards.padding,
    transition: tokens.animations.transitions.all,

    '&:hover': {
        boxShadow: tokens.elementStyling.cards.hover.shadow,
    },
});

// Image Upload Container
export const ImageUploadContainer = styled(Box)(({ isDragOver = false }: { isDragOver?: boolean }) => ({
    background: isDragOver
        ? tokens.elementStyling.imageUpload.container.dragOver.background
        : tokens.elementStyling.imageUpload.container.background,
    border: isDragOver
        ? tokens.elementStyling.imageUpload.container.dragOver.border
        : tokens.elementStyling.imageUpload.container.border,
    borderRadius: tokens.elementStyling.imageUpload.container.borderRadius,
    padding: tokens.elementStyling.imageUpload.container.padding,
    textAlign: 'center' as const,
    cursor: 'pointer',
    transition: tokens.animations.transitions.all,

    '&:hover': {
        background: tokens.elementStyling.imageUpload.container.hover.background,
        border: tokens.elementStyling.imageUpload.container.hover.border,
    },
}));

// Image Preview Container
export const ImagePreviewContainer = styled(Box)({
    background: tokens.elementStyling.imageUpload.preview.background,
    border: tokens.elementStyling.imageUpload.preview.border,
    borderRadius: tokens.elementStyling.imageUpload.preview.borderRadius,
    boxShadow: tokens.elementStyling.imageUpload.preview.shadow,
    overflow: 'hidden',
    position: 'relative',
});

// Model Selector Container
export const ModelSelectorContainer = styled(Box)(({ isSelected = false }: { isSelected?: boolean }) => ({
    background: isSelected
        ? tokens.elementStyling.modelSelector.container.selected.background
        : tokens.elementStyling.modelSelector.container.background,
    border: isSelected
        ? tokens.elementStyling.modelSelector.container.selected.border
        : tokens.elementStyling.modelSelector.container.border,
    borderRadius: tokens.elementStyling.modelSelector.container.borderRadius,
    padding: tokens.elementStyling.modelSelector.container.padding,
    cursor: 'pointer',
    transition: tokens.animations.transitions.all,
    boxShadow: isSelected ? tokens.elementStyling.modelSelector.container.selected.shadow : 'none',

    '&:hover': {
        border: tokens.elementStyling.modelSelector.container.hover.border,
        background: tokens.elementStyling.modelSelector.container.hover.background,
    },
}));

// Status Badge Components
export const SuccessBadge = styled(Box)({
    background: tokens.elementStyling.modelSelector.badges.excellent.background,
    color: tokens.elementStyling.modelSelector.badges.excellent.text,
    borderRadius: tokens.elementStyling.modelSelector.badges.excellent.borderRadius,
    padding: `${tokens.spacing.xs} ${tokens.spacing.sm}`,
    fontSize: tokens.typography.caption.fontSize,
    fontWeight: 500,
    display: 'inline-flex',
    alignItems: 'center',
});

export const BetterBadge = styled(Box)({
    background: tokens.elementStyling.modelSelector.badges.better.background,
    color: tokens.elementStyling.modelSelector.badges.better.text,
    borderRadius: tokens.elementStyling.modelSelector.badges.better.borderRadius,
    padding: `${tokens.spacing.xs} ${tokens.spacing.sm}`,
    fontSize: tokens.typography.caption.fontSize,
    fontWeight: 500,
    display: 'inline-flex',
    alignItems: 'center',
});

export const GoodBadge = styled(Box)({
    background: tokens.elementStyling.modelSelector.badges.good.background,
    color: tokens.elementStyling.modelSelector.badges.good.text,
    borderRadius: tokens.elementStyling.modelSelector.badges.good.borderRadius,
    padding: `${tokens.spacing.xs} ${tokens.spacing.sm}`,
    fontSize: tokens.typography.caption.fontSize,
    fontWeight: 500,
    display: 'inline-flex',
    alignItems: 'center',
});

// Processing Status Components
export const ProcessingStatusContainer = styled(Box)(({ status }: { status: 'success' | 'error' | 'processing' }) => {
    const statusStyles = tokens.elementStyling.processingStatus[status];
    return {
        background: statusStyles.background,
        border: statusStyles.border,
        borderRadius: statusStyles.borderRadius,
        padding: tokens.spacing.lg,
        display: 'flex',
        alignItems: 'center',
        gap: tokens.spacing.md,
    };
});

// Feature Card with gradient background
export const FeatureCard = styled(Card)({
    background: tokens.elementStyling.featureCard.background,
    border: tokens.elementStyling.featureCard.border,
    borderRadius: tokens.elementStyling.featureCard.borderRadius,
    padding: tokens.elementStyling.featureCard.padding,
    transition: tokens.animations.transitions.all,

    '&:hover': {
        transform: 'translateY(-2px)',
        boxShadow: tokens.shadows.cardHover,
    },
});

// Header Container
export const HeaderContainer = styled(Box)({
    background: tokens.elementStyling.header.background,
    borderBottom: tokens.elementStyling.header.border.bottom,
    boxShadow: tokens.elementStyling.header.shadow,
    padding: `${tokens.spacing.lg} ${tokens.spacing.xl}`,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
});

// Logo Container with gradient
export const LogoContainer = styled(Box)({
    background: tokens.elementStyling.header.logo.background,
    borderRadius: tokens.elementStyling.header.logo.borderRadius,
    padding: tokens.spacing.md,
    display: 'flex',
    alignItems: 'center',
    gap: tokens.spacing.sm,
    color: tokens.elementStyling.header.logo.iconColor,
    fontWeight: 600,
    fontSize: tokens.typography.headings.h3.fontSize,
});

// Transparency Pattern Background (for PNG previews)
export const TransparencyPatternBox = styled(Box)({
    position: 'relative',
    '&::before': {
        content: '""',
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        background: tokens.elementStyling.imageComparison.transparencyPattern.background,
        backgroundSize: tokens.elementStyling.imageComparison.transparencyPattern.backgroundSize,
        backgroundPosition: tokens.elementStyling.imageComparison.transparencyPattern.backgroundPosition,
        opacity: tokens.elementStyling.imageComparison.transparencyPattern.opacity,
        zIndex: 0,
    },
    '& > *': {
        position: 'relative',
        zIndex: 1,
    },
}); 