import { Box, FormControl, InputLabel, MenuItem, Select } from '@mui/material'
import React from 'react'
import type { ModelInfo, ModelSelectorProps } from '../../types/api'

export const ModelSelector: React.FC<ModelSelectorProps & { models: ModelInfo[] }> = ({ models, selectedModel, onModelChange, disabled }) => {
    return (
        <Box sx={{ mb: 2 }} data-cy="model-selector">
            <FormControl fullWidth size="small">
                <InputLabel id="model-select-label">Modelo</InputLabel>
                <Select
                    labelId="model-select-label"
                    value={selectedModel}
                    label="Modelo"
                    onChange={(e) => onModelChange(String(e.target.value))}
                    disabled={disabled}
                >
                    {models.map((m) => (
                        <MenuItem key={m.id} value={m.id} data-cy={`model-option-${m.id}`}>
                            {m.name}
                        </MenuItem>
                    ))}
                </Select>
            </FormControl>
        </Box>
    )
}


