import { useState, useRef} from 'react';
import { useAppContext } from '../utils/app.context';
import { CONFIG_DEFAULT, CONFIG_INFO } from '../Config';
import StorageUtils from '../utils/storage';
import { useModals } from './ModalProvider';
import { classNames, isBoolean, isNumeric, isString } from '../utils/misc';
import {
  BeakerIcon,
  BookmarkIcon,
  ChatBubbleOvalLeftEllipsisIcon,
  Cog6ToothIcon,
  FunnelIcon,
  HandRaisedIcon,
  SquaresPlusIcon,
  TrashIcon,
} from '@heroicons/react/24/outline';
import { OpenInNewTab } from '../utils/common';
import { SettingsPreset } from '../utils/types';
import toast from 'react-hot-toast'

type SettKey = keyof typeof CONFIG_DEFAULT;

const BASIC_KEYS: SettKey[] = [
  'reasoning_format',
  'temperature',
  'top_k',
  'top_p',
  'min_p',
  'max_tokens',
];
const SAMPLER_KEYS: SettKey[] = [
  'dynatemp_range',
  'dynatemp_exponent',
  'typical_p',
  'xtc_probability',
  'xtc_threshold',
  'top_n_sigma'
];
const PENALTY_KEYS: SettKey[] = [
  'repeat_last_n',
  'repeat_penalty',
  'presence_penalty',
  'frequency_penalty',
  'dry_multiplier',
  'dry_base',
  'dry_allowed_length',
  'dry_penalty_last_n',
];

enum SettingInputType {
  SHORT_INPUT,
  LONG_INPUT,
  CHECKBOX,
  CUSTOM,
}

interface SettingFieldInput {
  type: Exclude<SettingInputType, SettingInputType.CUSTOM>;
  label: string | React.ReactElement;
  help?: string | React.ReactElement;
  key: SettKey;
}

interface SettingFieldCustom {
  type: SettingInputType.CUSTOM;
  key: SettKey;
  component:
    | string
    | React.FC<{
        value: string | boolean | number;
        onChange: (value: string) => void;
      }>;
}

interface SettingSection {
  title: React.ReactElement;
  fields: (SettingFieldInput | SettingFieldCustom)[];
}

const ICON_CLASSNAME = 'w-4 h-4 mr-1 inline';

// Presets Component
function PresetsManager({
  currentConfig,
  onLoadPreset,
}: {
  currentConfig: typeof CONFIG_DEFAULT;
  onLoadPreset: (config: typeof CONFIG_DEFAULT) => void;
}) {
  const [presets, setPresets] = useState<SettingsPreset[]>(() =>
    StorageUtils.getPresets()
  );
  const [presetName, setPresetName] = useState('');
  const [selectedPresetId, setSelectedPresetId] = useState<string | null>(null);
  const { showConfirm, showAlert } = useModals();

  const handleSavePreset = async () => {
    if (!presetName.trim()) {
      await showAlert('Please enter a preset name');
      return;
    }

    // Check if preset name already exists
    const existingPreset = presets.find((p) => p.name === presetName.trim());
    if (existingPreset) {
      if (
        await showConfirm(
          `Preset "${presetName}" already exists. Do you want to overwrite it?`
        )
      ) {
        StorageUtils.updatePreset(existingPreset.id, currentConfig);
        setPresets(StorageUtils.getPresets());
        setPresetName('');
        await showAlert('Preset updated successfully');
      }
    } else {
      const newPreset = StorageUtils.savePreset(
        presetName.trim(),
        currentConfig
      );
      setPresets([...presets, newPreset]);
      setPresetName('');
      await showAlert('Preset saved successfully');
    }
  };

  const handleLoadPreset = async (preset: SettingsPreset) => {
    if (
      await showConfirm(
        `Load preset "${preset.name}"? Current settings will be replaced.`
      )
    ) {
      onLoadPreset(preset.config as typeof CONFIG_DEFAULT);
      setSelectedPresetId(preset.id);
    }
  };

  const handleDeletePreset = async (preset: SettingsPreset) => {
    if (await showConfirm(`Delete preset "${preset.name}"?`)) {
      StorageUtils.deletePreset(preset.id);
      setPresets(presets.filter((p) => p.id !== preset.id));
      if (selectedPresetId === preset.id) {
        setSelectedPresetId(null);
      }
    }
  };

  return (
    <div className="space-y-4">
      {/* Save current settings as preset */}
      <div className="form-control">
        <label className="label">
          <span className="label-text">Save current settings as preset</span>
        </label>
        <div className="join">
          <input
            type="text"
            placeholder="Enter preset name"
            className="input input-bordered join-item flex-1"
            value={presetName}
            onChange={(e) => setPresetName(e.target.value)}
            onKeyPress={(e) => {
              if (e.key === 'Enter') {
                handleSavePreset();
              }
            }}
          />
          <button
            className="btn btn-primary join-item"
            onClick={handleSavePreset}
          >
            Save Preset
          </button>
        </div>
      </div>

      {/* List of saved presets */}
      <div className="form-control">
        <label className="label">
          <span className="label-text">Saved presets</span>
        </label>
        {presets.length === 0 ? (
          <div className="alert">
            <span>No presets saved yet</span>
          </div>
        ) : (
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {presets.map((preset) => (
              <div
                key={preset.id}
                className={classNames({
                  'card bg-base-200 p-3': true,
                  'ring-2 ring-primary': selectedPresetId === preset.id,
                })}
              >
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="font-semibold">{preset.name}</h4>
                    <p className="text-sm opacity-70">
                      Created: {new Date(preset.createdAt).toLocaleString()}
                    </p>
                  </div>
                  <div className="flex gap-2">
                    <button
                      className="btn btn-sm btn-primary"
                      onClick={() => handleLoadPreset(preset)}
                    >
                      Load
                    </button>
                    <button
                      className="btn btn-sm btn-error"
                      onClick={() => handleDeletePreset(preset)}
                    >
                      <TrashIcon className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

const SETTING_SECTIONS = (
  localConfig: typeof CONFIG_DEFAULT,
  setLocalConfig: (config: typeof CONFIG_DEFAULT) => void
): SettingSection[] => [
  {
    title: (
      <>
        <BookmarkIcon className={ICON_CLASSNAME} />
        Presets
      </>
    ),
    fields: [
      {
        type: SettingInputType.CUSTOM,
        key: 'custom', // dummy key for presets
        component: () => (
          <PresetsManager
            currentConfig={localConfig}
            onLoadPreset={setLocalConfig}
          />
        ),
      },
    ],
  },
  {
    title: (
      <>
        <Cog6ToothIcon className={ICON_CLASSNAME} />
        General
      </>
    ),
    fields: [
      {
        type: SettingInputType.SHORT_INPUT,
        label: 'API Key',
        key: 'apiKey',
      },
      {
        type: SettingInputType.LONG_INPUT,
        label: 'System Message (will be disabled if left empty)',
        key: 'systemMessage',
      },
      ...BASIC_KEYS.map(
        (key) =>
          ({
            type: SettingInputType.SHORT_INPUT,
            label: key,
            key,
          }) as SettingFieldInput
      ),
      {
        type: SettingInputType.SHORT_INPUT,
        label: 'Paste length to file',
        key: 'pasteLongTextToFileLen',
      },
      {
        type: SettingInputType.CHECKBOX,
        label: 'Parse PDF as image instead of text',
        key: 'pdfAsImage',
      },      
    ],
  },
  {
    title: (
      <>
        <FunnelIcon className={ICON_CLASSNAME} />
        Samplers
      </>
    ),
    fields: [
      {
        type: SettingInputType.SHORT_INPUT,
        label: 'Samplers queue',
        key: 'samplers',
      },
      ...SAMPLER_KEYS.map(
        (key) =>
          ({
            type: SettingInputType.SHORT_INPUT,
            label: key,
            key,
          }) as SettingFieldInput
      ),
    ],
  },
  {
    title: (
      <>
        <HandRaisedIcon className={ICON_CLASSNAME} />
        Penalties
      </>
    ),
    fields: PENALTY_KEYS.map((key) => ({
      type: SettingInputType.SHORT_INPUT,
      label: key,
      key,
    })),
  },
  {
    title: (
      <>
        <ChatBubbleOvalLeftEllipsisIcon className={ICON_CLASSNAME} />
        Reasoning
      </>
    ),
    fields: [
      {
        type: SettingInputType.CHECKBOX,
        label: 'Expand thought process by default when generating messages',
        key: 'showThoughtInProgress',
      },
      {
        type: SettingInputType.CHECKBOX,
        label:
          'Exclude thought process when sending requests to API (Recommended for Reasoning Models like Deepseek R1)',
        key: 'excludeThoughtOnReq',
      },
    ],
  },
  {
    title: (
      <>
        <SquaresPlusIcon className={ICON_CLASSNAME} />
        Advanced
      </>
    ),
    fields: [
      {
        type: SettingInputType.CUSTOM,
        key: 'custom', // dummy key, won't be used
        component: () => {
          const debugImportDemoConv = async () => {
            const res = await fetch('/demo-conversation.json');
            const demoConv = await res.json();
            StorageUtils.remove(demoConv.id);
            for (const msg of demoConv.messages) {
              StorageUtils.appendMsg(demoConv.id, msg, msg.model_name);
            }
          };
          return (
            <button className="btn" onClick={debugImportDemoConv}>
              (debug) Import demo conversation
            </button>
          );
        },
      },
      {
        type: SettingInputType.CUSTOM,
        key: 'custom', // dummy key, won't be used
        component: () => {
          const exportDB = async () => {
            const blob = await StorageUtils.exportDB();
            const a = document.createElement('a');
            document.body.appendChild(a);
            a.href = URL.createObjectURL(blob);
            document.body.appendChild(a);
            a.download = `llamawebui_dump.json`;
            a.click();
            document.body.removeChild(a);
          };
          return (
            <button className="btn" onClick={exportDB}>
              Export conversation database
            </button>
          );
        },
      },
     {
        type: SettingInputType.CUSTOM,
        key: 'custom', // dummy key, won't be used
        component: () => {
          const importDB = async (e: React.ChangeEvent<HTMLInputElement>) => {
            console.log(e);
            if (!e.target.files) {
              toast.error('Target.files cant be null');
              throw new Error('e.target.files cant be null');
            }
            if (e.target.files.length != 1)
            {
              toast.error(
                'Number of selected files for DB import must be 1 but was ' +
                  e.target.files.length +
                  '.');
              throw new Error(
                'Number of selected files for DB import must be 1 but was ' +
                  e.target.files.length +
                  '.'
              );
            }
            const file = e.target.files[0];
            try {
              if (!file) throw new Error('No DB found to import.');
              console.log('Importing DB ' + file.name);
              await StorageUtils.importDB(file);
              toast.success('Import complete')
              window.location.reload();
            } catch (error) {
              //console.error('' + error);
              toast.error('' + error);
            }
          };
          return (
            <div>
              <label
                htmlFor="db-import"
                className="btn"
                role="button"
                tabIndex={0}
              >
                {' '}
                Reset and import conversation database{' '}
              </label>
              <input
                id="db-import"
                type="file"
                accept=".json"
                className="file-upload"
                onInput={importDB}
                hidden
              />
            </div>
          );
        },
      },

      {
        type: SettingInputType.CHECKBOX,
        label: 'Show generation stats (model name, context size, prompt and token per second)',
        key: 'showTokensPerSecond',
      },
      {
        type: SettingInputType.CHECKBOX,
        label: 'Use server defaults for parameters (skip sending temp, top_k, top_p, min_p, typical p from WebUI)',
        key: 'useServerDefaults',
      },
      {
        type: SettingInputType.LONG_INPUT,
        label: (
          <>
            Custom JSON config (For more info, refer to{' '}
            <OpenInNewTab href="https://github.com/ikawrakow/ik_llama.cpp/tree/main/examples/server/README.md">
              server documentation
            </OpenInNewTab>
            )
          </>
        ),
        key: 'custom',
      },
    ],
  },
  {
    title: (
      <>
        <BeakerIcon className={ICON_CLASSNAME} />
        Experimental
      </>
    ),
    fields: [
      {
        type: SettingInputType.CUSTOM,
        key: 'custom', // dummy key, won't be used
        component: () => (
          <>
            <p className="mb-8">
              Experimental features are not guaranteed to work correctly.
              <br />
              <br />
              If you encounter any problems, create a{' '}
              <OpenInNewTab href="https://github.com/ikawrakow/ik_llama.cpp/issues/new?template=019-bug-misc.yml">
                Bug (misc.)
              </OpenInNewTab>{' '}
              report on Github. Please also specify <b>webui/experimental</b> on
              the report title and include screenshots.
              <br />
              <br />
              Some features may require packages downloaded from CDN, so they
              need internet connection.
            </p>
          </>
        ),
      },
      {
        type: SettingInputType.CHECKBOX,
        label: (
          <>
            <b>Enable Python interpreter</b>
            <br />
            <small className="text-xs">
              This feature uses{' '}
              <OpenInNewTab href="https://pyodide.org">pyodide</OpenInNewTab>,
              downloaded from CDN. To use this feature, ask the LLM to generate
              Python code inside a Markdown code block. You will see a "Run"
              button on the code block, near the "Copy" button.
            </small>
          </>
        ),
        key: 'pyIntepreterEnabled',
      },
    ],
  },
  
];

export default function SettingDialog({
  show,
  onClose,
}: {
  show: boolean;
  onClose: () => void;
}) {
  const { config, saveConfig } = useAppContext();
  const [sectionIdx, setSectionIdx] = useState(0);

  // clone the config object to prevent direct mutation
  const [localConfig, setLocalConfig] = useState<typeof CONFIG_DEFAULT>(
    JSON.parse(JSON.stringify(config))
  );

    // Generate sections with access to local state
  const SETTING_SECTIONS_GENERATED = SETTING_SECTIONS(
    localConfig,
    setLocalConfig
  );

  const resetConfig = () => {
    if (window.confirm('Are you sure you want to reset all settings?')) {
      setLocalConfig(CONFIG_DEFAULT);
    }
  };

  function isValidKey(key: string): key is SettKey {
    return key in CONFIG_DEFAULT;
  }


  const handleSave = () => {
    // copy the local config to prevent direct mutation
    const newConfig: typeof CONFIG_DEFAULT = { 
      ...CONFIG_DEFAULT,
      ...JSON.parse(
      JSON.stringify(localConfig)
    )};
    // validate the config
    for (const key in newConfig) {
      if (!isValidKey(key)) {
        console.log(`Unknown default type for key ${key}`);
        continue;
      }
      const value = newConfig[key as SettKey];
      const mustBeBoolean = isBoolean(CONFIG_DEFAULT[key as SettKey]);
      const mustBeString = isString(CONFIG_DEFAULT[key as SettKey]);
      const mustBeNumeric = isNumeric(CONFIG_DEFAULT[key as SettKey]);
      if (mustBeString) {
        if (!isString(value)) {
          alert(`Value for ${key} must be string`);
          return;
        }
      } else if (mustBeNumeric) {
        const trimmedValue = value.toString().trim();
        const numVal = Number(trimmedValue);
        if (isNaN(numVal) || !isNumeric(numVal) || trimmedValue.length === 0) {
          alert(`Value for ${key} must be numeric`);
          return;
        }
        // force conversion to number
        // @ts-expect-error this is safe
        newConfig[key] = numVal;
      } else if (mustBeBoolean) {
        if (!isBoolean(value)) {
          alert(`Value for ${key} must be boolean`);
          return;
        }
      } else {
        //console.error(`Unknown default type for key ${key}`);
        toast.error(`Unknown default type for key ${key}`);
      }
    }
    saveConfig(newConfig);
    onClose();
  };

  const onChange = (key: SettKey) => (value: string | boolean) => {
    // note: we do not perform validation here, because we may get incomplete value as user is still typing it
    setLocalConfig({ ...localConfig, [key]: value });
  };

  const detailsRef = useRef<HTMLDetailsElement>(null);  // <-- Add this line
  return (
    <dialog className={classNames({ modal: true, 'modal-open': show })}>
      <div className="modal-box w-11/12 max-w-3xl">
        <h3 className="text-lg font-bold mb-6">Settings</h3>
        <div className="flex flex-col md:flex-row h-[calc(90vh-12rem)]">
          {/* Left panel, showing sections - Desktop version */}
          <div className="hidden md:flex flex-col items-stretch pr-4 mr-4 border-r-2 border-base-200">
            {SETTING_SECTIONS_GENERATED.map((section, idx) => (
              <div
                key={idx}
                className={classNames({
                  'btn btn-ghost justify-start font-normal w-44 mb-1': true,
                  'btn-active': sectionIdx === idx,
                })}
                onClick={() => setSectionIdx(idx)}
                dir="auto"
              >
                {section.title}
              </div>
            ))}
          </div>

          {/* Left panel, showing sections - Mobile version */}
          <div className="md:hidden flex flex-row gap-2 mb-4">
            <details className="dropdown" ref={detailsRef}>
              <summary className="btn bt-sm w-full m-1">
                {SETTING_SECTIONS_GENERATED[sectionIdx].title}
              </summary>
              <ul className="menu dropdown-content bg-base-100 rounded-box z-[1] w-52 p-2 shadow">
                {SETTING_SECTIONS_GENERATED.map((section, idx) => (
                  <div
                    key={idx}
                    className={classNames({
                      'btn btn-ghost justify-start font-normal': true,
                      'btn-active': sectionIdx === idx,
                    })}
                    onClick={() => {setSectionIdx(idx);
                      detailsRef.current?.removeAttribute('open');
                    }}
                    dir="auto"
                  >
                    {section.title}
                  </div>
                ))}
              </ul>
            </details>
          </div>

          {/* Right panel, showing setting fields */}
          <div className="grow overflow-y-auto px-4">
            {SETTING_SECTIONS_GENERATED[sectionIdx].fields.map((field, idx) => {
              const key = `${sectionIdx}-${idx}`;
              if (field.type === SettingInputType.SHORT_INPUT) {
                return (
                  <SettingsModalShortInput
                    key={key}
                    configKey={field.key}
                    value={localConfig[field.key]}
                    onChange={onChange(field.key)}
                    label={field.label as string}
                  />
                );
              } else if (field.type === SettingInputType.LONG_INPUT) {
                return (
                  <SettingsModalLongInput
                    key={key}
                    configKey={field.key}
                    value={localConfig[field.key].toString()}
                    onChange={onChange(field.key)}
                    label={field.label as string}
                  />
                );
              } else if (field.type === SettingInputType.CHECKBOX) {
                return (
                  <SettingsModalCheckbox
                    key={key}
                    configKey={field.key}
                    value={!!localConfig[field.key]}
                    onChange={onChange(field.key)}
                    label={field.label as string}
                  />
                );
              } else if (field.type === SettingInputType.CUSTOM) {
                return (
                  <div key={key} className="mb-2">
                    {typeof field.component === 'string'
                      ? field.component
                      : field.component({
                          value: localConfig[field.key],
                          onChange: onChange(field.key),
                        })}
                  </div>
                );
              }
            })}

            <p className="opacity-40 mb-6 text-sm mt-8">
              Settings are saved in browser's localStorage
            </p>
          </div>
        </div>

        <div className="modal-action">
          <button className="btn" onClick={resetConfig}>
            Reset to default
          </button>
          <button className="btn" onClick={onClose}>
            Close
          </button>
          <button className="btn btn-primary" onClick={handleSave}>
            Save
          </button>
        </div>
      </div>
    </dialog>
  );
}

function SettingsModalLongInput({
  configKey,
  value,
  onChange,
  label,
}: {
  configKey: SettKey;
  value: string;
  onChange: (value: string) => void;
  label?: string;
}) {
  return (
    <label className="form-control mb-2">
      <div className="label inline">{label || configKey}</div>
      <textarea
        className="textarea textarea-bordered h-24"
          placeholder={`Default: ${CONFIG_DEFAULT[configKey] || 'none'}`}
          value={value}
          onChange={(e) => onChange(e.target.value)}
        />
      </label>
  );
}

function SettingsModalShortInput({
  configKey,
  value,
  onChange,
  label,
}: {
  configKey: SettKey;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  value: any;
  onChange: (value: string) => void;
  label?: string;
}) {
  const helpMsg = CONFIG_INFO[configKey];

  return (
    <>
      {/* on mobile, we simply show the help message here */}
      {helpMsg && (
        <div className="block md:hidden mb-1">
          <b>{label || configKey}</b>
          <br />
          <p className="text-xs whitespace-normal">{helpMsg}</p>
        </div>
      )}
      <label className="input input-bordered join-item grow flex items-center gap-2 mb-2">
        <div className="dropdown dropdown-hover">
          <div tabIndex={0} role="button" className="font-bold hidden md:block">
            {label || configKey}
          </div>
          {helpMsg && (
            <div className="dropdown-content menu bg-base-100 rounded-box z-10 w-64 p-2 shadow mt-4 whitespace-normal break-words">
              {helpMsg}
            </div>
          )}
        </div>
        <input
          type="text"
          className="grow"
          placeholder={`Default: ${CONFIG_DEFAULT[configKey] || 'none'}`}
          value={value}
          onChange={(e) => onChange(e.target.value)}
        />
      </label>
    </>
  );
}

function SettingsModalCheckbox({
  configKey,
  value,
  onChange,
  label,
}: {
  configKey: SettKey;
  value: boolean;
  onChange: (value: boolean) => void;
  label: string;
}) {
  return (
    <div className="flex flex-row items-center mb-2">
      <input
        type="checkbox"
        className="toggle"
        checked={value}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span className="ml-4">{label || configKey}</span>
    </div>
  );
}
