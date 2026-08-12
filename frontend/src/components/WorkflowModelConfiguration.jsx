import { useEffect } from 'react';
import ModelConfiguration, {
  THINKING_EFFORTS,
  modelConfigurationForCatalog,
  modelConfigurationIsValid,
} from './ModelConfiguration.jsx';
import { thinkingEffortForModelChange, thinkingEffortsForModel } from '../lib/modelProviders.js';
import {
  enableModelOverrides,
  modelOverridesDraft,
  modelSelectionDraft,
  normalizeModelOverrides,
  resolvedModelConfiguration,
} from '../lib/modelOverrides.js';

export function workflowModelConfigurationForCatalog(current, providers, catalog) {
  const base = modelConfigurationForCatalog(current, providers, catalog);
  const postProcessingEfforts = thinkingEffortsForModel(
    catalog,
    base.model_provider,
    base.model,
    THINKING_EFFORTS,
    base.harness
  );
  return {
    ...base,
    post_processing_thinking_effort: thinkingEffortForModelChange(
      current?.post_processing_thinking_effort ?? current?.postProcessingThinkingEffort ?? base.thinking_effort,
      postProcessingEfforts
    ),
    model_overrides: normalizeModelOverrides(current?.model_overrides ?? current?.modelOverrides, (configuration) =>
      modelConfigurationForCatalog(configuration, providers, catalog)
    ),
  };
}

export function workflowModelConfigurationIsValid(value, depths, providers, catalog) {
  if (!modelConfigurationIsValid(value, providers, catalog)) return false;
  const postProcessingEfforts = thinkingEffortsForModel(
    catalog,
    value.model_provider,
    value.model,
    THINKING_EFFORTS,
    value.harness
  );
  const postProcessingThinkingEffort =
    value?.post_processing_thinking_effort ?? value?.postProcessingThinkingEffort ?? value?.thinking_effort;
  if (!postProcessingEfforts.includes(postProcessingThinkingEffort)) return false;
  const overrides = modelOverridesDraft(value?.model_overrides ?? value?.modelOverrides);
  const allowedDepths = new Set(depths.map(String));
  if (Object.keys(overrides).some((depth) => !allowedDepths.has(depth))) return false;
  return depths.every((depth) =>
    modelConfigurationIsValid(resolvedModelConfiguration(value, depth), providers, catalog)
  );
}

function modeButtonStyle(active, disabled) {
  return {
    minHeight: 34,
    padding: '0 12px',
    border: `1px solid ${active ? 'var(--accent)' : 'var(--border)'}`,
    borderRadius: 8,
    background: active ? 'var(--accent-subtle)' : 'var(--surface)',
    color: disabled ? 'var(--text-3)' : active ? 'var(--accent)' : 'var(--text-2)',
    fontSize: 12,
    fontWeight: 600,
    cursor: disabled ? 'not-allowed' : 'pointer',
  };
}

export default function WorkflowModelConfiguration({
  value,
  onChange,
  depths,
  depthChips = [],
  providers,
  catalog,
  catalogError,
  disabled = false,
}) {
  const overrides = modelOverridesDraft(value?.model_overrides ?? value?.modelOverrides);
  const customized = Object.keys(overrides).length > 0;
  const canCustomize = depths.length > 0;
  const depthLabels = new Map(depthChips.map((chip) => [chip.depth, chip]));
  const postProcessingEfforts = thinkingEffortsForModel(
    catalog,
    value.model_provider,
    value.model,
    THINKING_EFFORTS,
    value.harness
  );
  const postProcessingThinkingEffort =
    value?.post_processing_thinking_effort ?? value?.postProcessingThinkingEffort ?? value?.thinking_effort;

  useEffect(() => {
    const normalizedPostProcessingEffort = thinkingEffortForModelChange(
      postProcessingThinkingEffort,
      postProcessingEfforts
    );

    if (
      normalizedPostProcessingEffort &&
      normalizedPostProcessingEffort !== postProcessingThinkingEffort
    ) {
      onChange({
        ...value,
        post_processing_thinking_effort: normalizedPostProcessingEffort,
      });
    }
  }, [
    postProcessingThinkingEffort,
    postProcessingEfforts,
    value,
    onChange,
  ]);

  const useSingleModel = () => onChange({ ...value, model_overrides: {} });
  const customizeByDepth = () =>
    onChange({
      ...value,
      model_overrides: enableModelOverrides(depths, value, overrides),
    });

  return (
    <>
      <div
        role="group"
        aria-label="Workflow model strategy"
        style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 14 }}
      >
        <button
          type="button"
          aria-pressed={!customized}
          disabled={disabled}
          onClick={useSingleModel}
          style={modeButtonStyle(!customized, disabled)}
        >
          One model for all depths
        </button>
        <button
          type="button"
          aria-pressed={customized}
          disabled={disabled || !canCustomize}
          onClick={customizeByDepth}
          style={modeButtonStyle(customized, disabled || !canCustomize)}
        >
          Customize by depth
        </button>
      </div>

      <div
        style={{
          border: customized ? '1px solid var(--border)' : 0,
          borderRadius: 9,
          padding: customized ? 13 : 0,
          background: customized ? 'var(--surface-2)' : 'transparent',
        }}
      >
        {customized && (
          <div className="mono" style={{ fontSize: 10.5, color: 'var(--text-3)', marginBottom: 9 }}>
            DEFAULT WORKFLOW MODEL
          </div>
        )}
        <ModelConfiguration
          value={value}
          onChange={(configuration) => onChange({ ...value, ...configuration })}
          providers={providers}
          catalog={catalog}
          catalogError={catalogError}
          disabled={disabled}
        />
      </div>

      <label
        style={{
          display: 'block',
          maxWidth: 280,
          marginTop: 12,
          border: '1px solid var(--border)',
          borderRadius: 9,
          padding: 13,
          background: 'var(--surface-2)',
        }}
      >
        <span className="mono" style={{ display: 'block', fontSize: 10.5, color: 'var(--text-3)', marginBottom: 7 }}>
          POST-PROCESSING THINKING EFFORT
        </span>
        <select
          className="mono"
          value={postProcessingThinkingEffort}
          disabled={disabled || postProcessingEfforts.length === 0}
          onChange={(event) => onChange({ ...value, post_processing_thinking_effort: event.target.value })}
          style={{
            width: '100%',
            height: 36,
            padding: '0 10px',
            borderRadius: 7,
            border: '1px solid var(--border)',
            background: 'var(--surface)',
            color: 'var(--text)',
          }}
        >
          {postProcessingEfforts.map((effort) => (
            <option key={effort} value={effort}>
              {effort}
            </option>
          ))}
        </select>
      </label>

      {customized && (
        <div style={{ display: 'grid', gap: 12, marginTop: 12 }}>
          {depths.map((depth) => {
            const chip = depthLabels.get(depth);
            const configuration = resolvedModelConfiguration(value, depth);
            return (
              <div
                key={depth}
                style={{
                  border: '1px solid var(--border)',
                  borderRadius: 9,
                  padding: 13,
                  background: 'var(--surface)',
                }}
              >
                <div
                  className="mono"
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    gap: 10,
                    fontSize: 10.5,
                    color: 'var(--text-3)',
                    marginBottom: 9,
                  }}
                >
                  <span>DEPTH {depth}</span>
                  {chip?.count ? (
                    <span>
                      {chip.count} {chip.count === 1 ? 'step' : 'steps'}
                    </span>
                  ) : null}
                </div>
                <ModelConfiguration
                  value={configuration}
                  onChange={(nextConfiguration) =>
                    onChange({
                      ...value,
                      model_overrides: {
                        ...overrides,
                        [`${depth}`]: modelSelectionDraft(nextConfiguration),
                      },
                    })
                  }
                  providers={providers}
                  catalog={catalog}
                  catalogError={catalogError}
                  disabled={disabled}
                  showAvailabilityHelp={false}
                />
              </div>
            );
          })}
        </div>
      )}

      {!canCustomize && (
        <div style={{ marginTop: 8, color: 'var(--text-3)', fontSize: 11.5 }}>
          Select a workflow to configure models by depth.
        </div>
      )}
    </>
  );
}
