// Import design tokens from the root DESIGN.json file
// Note: Adjust this path based on your project structure
const designSystemData = {
    designSystem: {
        colorPalette: {
            primary: {
                blue500: "#3B82F6",
                blue600: "#2563EB",
                blue700: "#1D4ED8",
                blue50: "#EFF6FF",
                blue100: "#DBEAFE"
            },
            secondary: {
                indigo500: "#6366F1",
                indigo600: "#4F46E5",
                indigo50: "#EEF2FF"
            },
            neutral: {
                white: "#FFFFFF",
                gray50: "#F9FAFB",
                gray100: "#F3F4F6",
                gray200: "#E5E7EB",
                gray300: "#D1D5DB",
                gray400: "#9CA3AF",
                gray500: "#6B7280",
                gray600: "#4B5563",
                gray700: "#374151",
                gray800: "#1F2937",
                gray900: "#111827"
            },
            status: {
                success: "#10B981",
                successBg: "#ECFDF5",
                successBorder: "#D1FAE5",
                warning: "#F59E0B",
                warningBg: "#FFFBEB",
                warningBorder: "#FDE68A",
                error: "#EF4444",
                errorBg: "#FEF2F2",
                errorBorder: "#FECACA"
            }
        },
        gradients: {
            primaryButton: "linear-gradient(135deg, #3B82F6 0%, #2563EB 100%)",
            primaryButtonHover: "linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%)",
            logoBackground: "linear-gradient(135deg, #3B82F6 0%, #2563EB 100%)",
            featureCard: "linear-gradient(135deg, #EFF6FF 0%, #EEF2FF 100%)",
            processingOverlay: "linear-gradient(135deg, #3B82F6 0%, #2563EB 100%)"
        },
        shadows: {
            card: "0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)",
            cardHover: "0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)",
            button: "0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)",
            buttonHover: "0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)",
            modal: "0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)",
            header: "0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)"
        },
        borderRadius: {
            small: "0.375rem",
            medium: "0.5rem",
            large: "0.75rem",
            xl: "0.75rem",
            full: "9999px"
        },
        spacing: {
            xs: "0.25rem",
            sm: "0.5rem",
            md: "0.75rem",
            lg: "1rem",
            xl: "1.25rem",
            "2xl": "1.5rem",
            "3xl": "1.75rem",
            "4xl": "2rem",
            "6xl": "3rem",
            "8xl": "4rem"
        },
        typography: {
            fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            headings: {
                h1: {
                    fontSize: "1.5rem",
                    fontWeight: "600",
                    lineHeight: "1.2",
                    color: "#111827"
                },
                h2: {
                    fontSize: "1.25rem",
                    fontWeight: "600",
                    lineHeight: "1.25",
                    color: "#111827"
                },
                h3: {
                    fontSize: "1.125rem",
                    fontWeight: "500",
                    lineHeight: "1.3",
                    color: "#111827"
                },
                h4: {
                    fontSize: "1rem",
                    fontWeight: "500",
                    lineHeight: "1.4",
                    color: "#111827"
                }
            },
            body: {
                fontSize: "0.875rem",
                fontWeight: "400",
                lineHeight: "1.5",
                color: "#4B5563"
            },
            caption: {
                fontSize: "0.75rem",
                fontWeight: "400",
                lineHeight: "1.4",
                color: "#6B7280"
            }
        },
        elementStyling: {
            header: {
                background: "#FFFFFF",
                border: {
                    bottom: "1px solid #E5E7EB"
                },
                shadow: "0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)",
                logo: {
                    background: "linear-gradient(135deg, #3B82F6 0%, #2563EB 100%)",
                    iconColor: "#FFFFFF",
                    borderRadius: "0.75rem"
                },
                statusIndicator: {
                    processing: "#F59E0B",
                    ready: "#10B981",
                    background: "#EFF6FF",
                    borderRadius: "9999px"
                }
            },
            cards: {
                background: "#FFFFFF",
                border: "1px solid #E5E7EB",
                borderRadius: "0.75rem",
                shadow: "0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)",
                padding: "1.5rem",
                hover: {
                    shadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)"
                }
            },
            buttons: {
                primary: {
                    background: "linear-gradient(135deg, #3B82F6 0%, #2563EB 100%)",
                    text: "#FFFFFF",
                    borderRadius: "0.75rem",
                    shadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)",
                    padding: "1rem 2rem",
                    fontSize: "1rem",
                    fontWeight: "600",
                    hover: {
                        background: "linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%)",
                        shadow: "0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)",
                        transform: "translateY(-0.125rem)"
                    },
                    disabled: {
                        background: "#9CA3AF",
                        cursor: "not-allowed",
                        shadow: "none",
                        transform: "none"
                    }
                },
                secondary: {
                    background: "#F3F4F6",
                    text: "#4B5563",
                    borderRadius: "0.5rem",
                    padding: "0.75rem 1rem",
                    fontSize: "0.875rem",
                    fontWeight: "500",
                    hover: {
                        background: "#E5E7EB"
                    }
                },
                icon: {
                    background: "#FFFFFF",
                    border: "1px solid #E5E7EB",
                    borderRadius: "0.5rem",
                    padding: "0.375rem",
                    iconColor: "#4B5563",
                    hover: {
                        background: "#F9FAFB",
                        border: "1px solid #D1D5DB"
                    }
                }
            },
            imageUpload: {
                container: {
                    background: "#F9FAFB",
                    border: "2px dashed #D1D5DB",
                    borderRadius: "0.75rem",
                    padding: "2rem",
                    hover: {
                        background: "#F3F4F6",
                        border: "2px dashed #9CA3AF"
                    },
                    dragOver: {
                        background: "#EFF6FF",
                        border: "2px dashed #3B82F6"
                    }
                },
                icon: {
                    background: "#E5E7EB",
                    iconColor: "#6B7280",
                    borderRadius: "9999px",
                    dragOver: {
                        background: "#DBEAFE",
                        iconColor: "#2563EB"
                    }
                },
                preview: {
                    background: "#FFFFFF",
                    border: "2px solid #E5E7EB",
                    borderRadius: "0.75rem",
                    shadow: "0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)"
                }
            },
            modelSelector: {
                container: {
                    background: "#FFFFFF",
                    border: "2px solid #E5E7EB",
                    borderRadius: "0.75rem",
                    padding: "1rem",
                    hover: {
                        border: "2px solid #D1D5DB",
                        background: "#F9FAFB"
                    },
                    selected: {
                        border: "2px solid #3B82F6",
                        background: "#EFF6FF",
                        shadow: "0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)"
                    }
                },
                radioButton: {
                    border: "2px solid #D1D5DB",
                    background: "#FFFFFF",
                    borderRadius: "9999px",
                    selected: {
                        border: "2px solid #3B82F6",
                        background: "#3B82F6",
                        innerDot: "#FFFFFF"
                    }
                },
                badges: {
                    excellent: {
                        background: "#ECFDF5",
                        text: "#059669",
                        borderRadius: "9999px"
                    },
                    better: {
                        background: "#EEF2FF",
                        text: "#7C3AED",
                        borderRadius: "9999px"
                    },
                    good: {
                        background: "#EFF6FF",
                        text: "#2563EB",
                        borderRadius: "9999px"
                    }
                }
            },
            processingStatus: {
                success: {
                    background: "#ECFDF5",
                    border: "1px solid #D1FAE5",
                    borderRadius: "0.75rem",
                    iconColor: "#10B981",
                    textColor: "#065F46"
                },
                error: {
                    background: "#FEF2F2",
                    border: "1px solid #FECACA",
                    borderRadius: "0.75rem",
                    iconColor: "#EF4444",
                    textColor: "#991B1B"
                },
                processing: {
                    background: "#EFF6FF",
                    border: "1px solid #DBEAFE",
                    borderRadius: "0.75rem",
                    iconColor: "#3B82F6",
                    textColor: "#1E40AF",
                    progressBar: {
                        background: "#DBEAFE",
                        fill: "#3B82F6",
                        borderRadius: "9999px"
                    }
                }
            },
            imageComparison: {
                container: {
                    background: "#FFFFFF",
                    border: "1px solid #E5E7EB",
                    borderRadius: "0.75rem",
                    shadow: "0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)"
                },
                transparencyPattern: {
                    background: "linear-gradient(45deg, #000 25%, transparent 25%), linear-gradient(-45deg, #000 25%, transparent 25%), linear-gradient(45deg, transparent 75%, #000 75%), linear-gradient(-45deg, transparent 75%, #000 75%)",
                    backgroundSize: "20px 20px",
                    backgroundPosition: "0 0, 0 10px, 10px -10px, -10px 0px",
                    opacity: "0.1"
                }
            },
            loadingOverlay: {
                backdrop: {
                    background: "rgba(0, 0, 0, 0.5)",
                    backdropFilter: "blur(4px)"
                },
                modal: {
                    background: "#FFFFFF",
                    borderRadius: "0.75rem",
                    shadow: "0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)",
                    padding: "2rem"
                },
                spinner: {
                    background: "linear-gradient(135deg, #3B82F6 0%, #2563EB 100%)",
                    borderRadius: "9999px",
                    iconColor: "#FFFFFF"
                },
                progressBar: {
                    background: "#E5E7EB",
                    fill: "linear-gradient(135deg, #3B82F6 0%, #2563EB 100%)",
                    borderRadius: "9999px"
                }
            },
            icons: {
                navigation: {
                    color: "#4B5563",
                    size: "1.25rem",
                    hover: {
                        color: "#2563EB"
                    }
                },
                action: {
                    color: "#FFFFFF",
                    size: "1.25rem",
                    container: {
                        background: "transparent"
                    }
                },
                status: {
                    processing: "#F59E0B",
                    success: "#10B981",
                    error: "#EF4444",
                    size: "1.25rem"
                },
                feature: {
                    color: "#3B82F6",
                    size: "1rem"
                }
            },
            featureCard: {
                background: "linear-gradient(135deg, #EFF6FF 0%, #EEF2FF 100%)",
                border: "1px solid #DBEAFE",
                borderRadius: "0.75rem",
                padding: "1.5rem",
                bulletPoints: {
                    color: "#3B82F6"
                }
            }
        },
        animations: {
            hover: {
                duration: "200ms",
                easing: "ease-in-out"
            },
            loading: {
                spin: "animate-spin",
                pulse: "animate-pulse"
            },
            transitions: {
                all: "transition-all duration-200",
                colors: "transition-colors duration-200",
                transform: "transition-transform duration-200"
            }
        }
    }
};

export const tokens = designSystemData.designSystem;

// Helper functions to access design tokens
export const getColor = (path: string): string => {
    const pathArray = path.split('.');
    let current: Record<string, unknown> = tokens;

    for (const key of pathArray) {
        current = current[key] as Record<string, unknown>;
        if (current === undefined) {
            console.warn(`Design token not found: ${path}`);
            return '#000000'; // fallback color
        }
    }

    return current as string;
};

export const getShadow = (variant: keyof typeof tokens.shadows): string => {
    return tokens.shadows[variant];
};

export const getSpacing = (size: keyof typeof tokens.spacing): string => {
    return tokens.spacing[size];
};

export const getBorderRadius = (size: keyof typeof tokens.borderRadius): string => {
    return tokens.borderRadius[size];
};

export const getGradient = (variant: keyof typeof tokens.gradients): string => {
    return tokens.gradients[variant];
};

// Export commonly used token groups
export const colors = tokens.colorPalette;
export const shadows = tokens.shadows;
export const spacing = tokens.spacing;
export const gradients = tokens.gradients;
export const typography = tokens.typography;
export const elementStyling = tokens.elementStyling; 