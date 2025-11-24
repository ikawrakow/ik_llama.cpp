import { useEffect, useState } from 'react';
import StorageUtils from '../utils/storage';
import { useAppContext } from '../utils/app.context';
import { classNames } from '../utils/misc';
import daisyuiThemes from 'daisyui/theme/object';
import { THEMES } from '../Config';
import { useNavigate } from 'react-router';
import toast from 'react-hot-toast';
import { useModals } from './ModalProvider';
import {
  ArrowUpTrayIcon,
  ArrowDownTrayIcon,
  PencilIcon,
  TrashIcon,
  MoonIcon,
} from '@heroicons/react/24/outline';

export default function Header() {
  const navigate = useNavigate();
  const [selectedTheme, setSelectedTheme] = useState(StorageUtils.getTheme());
  const { setShowSettings } = useAppContext();

  const setTheme = (theme: string) => {
    StorageUtils.setTheme(theme);
    setSelectedTheme(theme);
  };

  useEffect(() => {
    document.body.setAttribute('data-theme', selectedTheme);
    document.body.setAttribute(
      'data-color-scheme',
      daisyuiThemes[selectedTheme]?.['color-scheme'] ?? 'auto'
    );
  }, [selectedTheme]);

  const {showPrompt } = useModals();
  const { isGenerating, viewingChat } = useAppContext();
  const isCurrConvGenerating = isGenerating(viewingChat?.conv.id ?? '');

  // remove conversation
  const removeConversation = () => {
    if (isCurrConvGenerating || !viewingChat) return;
    const convId = viewingChat?.conv.id;
    if (window.confirm('Are you sure to delete this conversation?')) {
      StorageUtils.remove(convId);
      navigate('/');
    }
  };
  
  // rename conversation 
  async function renameConversation() {  
	if (isGenerating(viewingChat?.conv.id ?? '')) {
	  toast.error(
		'Cannot rename conversation while generating'
	  );
	  return;
	}
	const newName = await showPrompt(
	  'Enter new name for the conversation',
	  viewingChat?.conv.name
	);
	if (newName && newName.trim().length > 0) {
	  StorageUtils.updateConversationName(viewingChat?.conv.id ?? '', newName);
	}
  };
  
  // at the top of your file, alongside ConversationExport:
  async function importConversation() {  
    const importedConv = await StorageUtils.importConversationFromFile();
  if (importedConv) {
    console.log('Successfully imported:', importedConv.name);
    // Refresh UI or navigate to conversation
    navigate(`/chat/${importedConv.id}`);
  }
  };

  const downloadConversation = () => {
    if (isCurrConvGenerating || !viewingChat) return;
    const convId = viewingChat?.conv.id;

    // Get the current system message from config
    const systemMessage = StorageUtils.getConfig().systemMessage;
    
    // Clone the viewingChat object to avoid modifying the original
    const exportData = {
      conv: { ...viewingChat.conv },
      messages: viewingChat.messages.map(msg => ({ ...msg }))
    };
    
    // Find the root message and update its content
    const rootMessage = exportData.messages.find(m => m.type === 'root');
    if (rootMessage) {
      rootMessage.content = systemMessage;
    }

    const conversationJson = JSON.stringify(exportData, null, 2);
    const blob = new Blob([conversationJson], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `conversation_${convId}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex flex-row items-center pt-6 pb-6 sticky top-0 z-10 bg-base-100">
      {/* open sidebar button */}
      <label htmlFor="toggle-drawer" className="btn btn-ghost lg:hidden">
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="16"
          height="16"
          fill="currentColor"
          className="bi bi-list"
          viewBox="0 0 16 16"
        >
          <path
            fillRule="evenodd"
            d="M2.5 12a.5.5 0 0 1 .5-.5h10a.5.5 0 0 1 0 1H3a.5.5 0 0 1-.5-.5m0-4a.5.5 0 0 1 .5-.5h10a.5.5 0 0 1 0 1H3a.5.5 0 0 1-.5-.5m0-4a.5.5 0 0 1 .5-.5h10a.5.5 0 0 1 0 1H3a.5.5 0 0 1-.5-.5"
          />
        </svg>
      </label>

      <div className="grow text-2xl font-bold ml-2">ik_llama.cpp</div>

      {/* action buttons (top right) */}
      <div className="flex items-center">
	  {/* start */ }
	    {/*viewingChat && */ /* show options for new conversation as well */
         (
          <div className="dropdown dropdown-end">
            {/* "..." button */}

            <button
              tabIndex={0}
              role="button"
              className="btn m-1"
              disabled={isCurrConvGenerating}
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="16"
                height="16"
                fill="currentColor"
                className="bi bi-three-dots-vertical"
                viewBox="0 0 16 16"
              >
                <path d="M9.5 13a1.5 1.5 0 1 1-3 0 1.5 1.5 0 0 1 3 0m0-5a1.5 1.5 0 1 1-3 0 1.5 1.5 0 0 1 3 0m0-5a1.5 1.5 0 1 1-3 0 1.5 1.5 0 0 1 3 0" />
              </svg>
            </button>
            {/* dropdown menu */}
            <ul
              tabIndex={0}
              className="dropdown-content menu bg-base-100 rounded-box z-[1] w-52 p-2 shadow"
            >
              {/* Always show Upload when viewingChat is false */}
              {!viewingChat && (
                <li onClick={importConversation}>
                  <a>
				  <ArrowUpTrayIcon className="w-4 h-4" />
				  Upload
				  </a>
                </li>
              )}

              {/* Show all three when viewingChat is true */}
              {viewingChat && (
                <>
                  <li onClick={importConversation}>
                    <a>				  
					<ArrowUpTrayIcon className="w-4 h-4" />
					Upload
					</a>
                  </li>
				  <li onClick={renameConversation} tabIndex={0}>
					<a>
					  <PencilIcon className="w-4 h-4" />
					  Rename
					</a>
				  </li>
                  <li onClick={downloadConversation}>
                    <a>
					<ArrowDownTrayIcon className="w-4 h-4" />
					Download
					</a>
                  </li>
                  <li className="text-error" onClick={removeConversation}>
                    <a>
					<TrashIcon className="w-4 h-4" />
					Delete
					</a>
                  </li>
                </>
              )}
            </ul>
          </div>
        )}

        <div className="tooltip tooltip-bottom" data-tip="Settings">
          <button className="btn" onClick={() => setShowSettings(true)}>
            {/* settings button */}
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="16"
              height="16"
              fill="currentColor"
              className="bi bi-gear"
              viewBox="0 0 16 16"
            >
              <path d="M8 4.754a3.246 3.246 0 1 0 0 6.492 3.246 3.246 0 0 0 0-6.492M5.754 8a2.246 2.246 0 1 1 4.492 0 2.246 2.246 0 0 1-4.492 0" />
              <path d="M9.796 1.343c-.527-1.79-3.065-1.79-3.592 0l-.094.319a.873.873 0 0 1-1.255.52l-.292-.16c-1.64-.892-3.433.902-2.54 2.541l.159.292a.873.873 0 0 1-.52 1.255l-.319.094c-1.79.527-1.79 3.065 0 3.592l.319.094a.873.873 0 0 1 .52 1.255l-.16.292c-.892 1.64.901 3.434 2.541 2.54l.292-.159a.873.873 0 0 1 1.255.52l.094.319c.527 1.79 3.065 1.79 3.592 0l.094-.319a.873.873 0 0 1 1.255-.52l.292.16c1.64.893 3.434-.902 2.54-2.541l-.159-.292a.873.873 0 0 1 .52-1.255l.319-.094c1.79-.527 1.79-3.065 0-3.592l-.319-.094a.873.873 0 0 1-.52-1.255l.16-.292c.893-1.64-.902-3.433-2.541-2.54l-.292.159a.873.873 0 0 1-1.255-.52zm-2.633.283c.246-.835 1.428-.835 1.674 0l.094.319a1.873 1.873 0 0 0 2.693 1.115l.291-.16c.764-.415 1.6.42 1.184 1.185l-.159.292a1.873 1.873 0 0 0 1.116 2.692l.318.094c.835.246.835 1.428 0 1.674l-.319.094a1.873 1.873 0 0 0-1.115 2.693l.16.291c.415.764-.42 1.6-1.185 1.184l-.291-.159a1.873 1.873 0 0 0-2.693 1.116l-.094.318c-.246.835-1.428.835-1.674 0l-.094-.319a1.873 1.873 0 0 0-2.692-1.115l-.292.16c-.764.415-1.6-.42-1.184-1.185l.159-.291A1.873 1.873 0 0 0 1.945 8.93l-.319-.094c-.835-.246-.835-1.428 0-1.674l.319-.094A1.873 1.873 0 0 0 3.06 4.377l-.16-.292c-.415-.764.42-1.6 1.185-1.184l.292.159a1.873 1.873 0 0 0 2.692-1.115z" />
            </svg>
          </button>
        </div>

        {/* theme controller is copied from https://daisyui.com/components/theme-controller/ */}
        <div className="tooltip tooltip-bottom" data-tip="Themes">
          <div className="dropdown dropdown-end dropdown-bottom">
            <div tabIndex={0} role="button" className="btn m-1">
              <MoonIcon className="w-5 h-5" />
            </div>
            <ul
              tabIndex={0}
              className="dropdown-content bg-base-300 rounded-box z-[1] w-52 p-2 shadow-2xl h-80 overflow-y-auto"
            >
              <li>
                <button
                  className={classNames({
                    'btn btn-sm btn-block btn-ghost justify-start': true,
                    'btn-active': selectedTheme === 'auto',
                  })}
                  onClick={() => setTheme('auto')}
                >
                  auto
                </button>
              </li>
              {THEMES.map((theme) => (
                <li key={theme}>
                  <input
                    type="radio"
                    name="theme-dropdown"
                    className="theme-controller btn btn-sm btn-block btn-ghost justify-start"
                    aria-label={theme}
                    value={theme}
                    checked={selectedTheme === theme}
                    onChange={(e) => e.target.checked && setTheme(theme)}
                  />
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
