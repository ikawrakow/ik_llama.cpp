import { useEffect, useMemo, useState } from 'react';
import { classNames } from '../utils/misc';
import { Conversation } from '../utils/types';
import StorageUtils from '../utils/storage';
import { useNavigate, useParams } from 'react-router';
import {
  ArrowUpTrayIcon,
  ArrowDownTrayIcon,
  EllipsisVerticalIcon,
  PencilIcon,
  PencilSquareIcon,
  TrashIcon,
  XMarkIcon,
} from '@heroicons/react/24/outline';
import { BtnWithTooltips } from '../utils/common';
import { useAppContext } from '../utils/app.context';
import toast from 'react-hot-toast';
import { useModals } from './ModalProvider';
import {DateTime} from 'luxon'

  // at the top of your file, alongside ConversationExport:
  async function importConversation() {  
    const importedConv = await StorageUtils.importConversationFromFile();
  if (importedConv) {
    console.log('Successfully imported:', importedConv.name);
    // Refresh UI or navigate to conversation
  }
  };
export default function Sidebar() {
  const params = useParams();
  const navigate = useNavigate();

  const { isGenerating, viewingChat } = useAppContext();

  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [currConv, setCurrConv] = useState<Conversation | null>(null);

  useEffect(() => {
    StorageUtils.getOneConversation(params.convId ?? '').then(setCurrConv);
  }, [params.convId]);

  useEffect(() => {
    const handleConversationChange = async () => {
      setConversations(await StorageUtils.getAllConversations());
    };
    StorageUtils.onConversationChanged(handleConversationChange);
    handleConversationChange();
    return () => {
      StorageUtils.offConversationChanged(handleConversationChange);
    };
  }, []);
  const { showConfirm, showPrompt } = useModals();

  const groupedConv = useMemo(
    () => groupConversationsByDate(conversations),
    [conversations]
  );


  
  return (
    <>
      <input
        id="toggle-drawer"
        type="checkbox"
        className="drawer-toggle"
        aria-label="Toggle sidebar"
        defaultChecked
      />

      <div
        className="drawer-side h-screen lg:h-screen z-50 lg:max-w-64"
        role="complementary"
        aria-label="Sidebar"
        tabIndex={0}
      >
        <label
          htmlFor="toggle-drawer"
          aria-label="Close sidebar"
          className="drawer-overlay"
        ></label>

        <a
          href="#main-scroll"
          className="absolute -left-80 top-0 w-1 h-1 overflow-hidden"
        >
          Skip to main content
        </a>

        <div className="flex flex-col bg-base-200 min-h-full max-w-64 py-4 px-4">
          <div className="flex flex-row items-center justify-between mb-4 mt-4">
            <h2 className="font-bold ml-4" role="heading">
              Conversations
            </h2>

            {/* close sidebar button */}
            <label
              htmlFor="toggle-drawer"
              className="btn btn-ghost lg:hidden"
              aria-label="Close sidebar"
              role="button"
              tabIndex={0}
            >
              <XMarkIcon className="w-5 h-5" />
            </label>
          </div>

          {/* new conversation button */}
          <button
            className={classNames({
              'btn btn-ghost justify-start px-2': true,
              'btn-soft': !currConv,
            })}
            onClick={() => navigate('/')}
            aria-label="New conversation"
          >
            <PencilSquareIcon className="w-5 h-5" />
            New Conversations
          </button>

          {/* list of conversations */}
          {groupedConv.map((group, i) => (
            <div key={i} role="group">
              {/* group name (by date) */}
              {group.title ? (
                // we use btn class here to make sure that the padding/margin are aligned with the other items
                <b
                  className="btn btn-ghost btn-xs bg-none btn-disabled block text-xs text-base-content text-start px-2 mb-0 mt-6 font-bold"
                  role="note"
                  aria-description={group.title}
                  tabIndex={0}
                >
                  {group.title}
                </b>
              ) : (
                <div className="h-2" />
              )}

              {group.conversations.map((conv) => (
                <ConversationItem
                  key={conv.id}
                  conv={conv}
                  isCurrConv={currConv?.id === conv.id}
                  onSelect={() => {
                    navigate(`/chat/${conv.id}`);
                  }}
                  onDelete={async () => {
                    if (isGenerating(conv.id)) {
                      toast.error(
                        'Cannot delete conversation while generating'
                      );
                      return;
                    }
                    if (
                      await showConfirm(
                        'Are you sure to delete this conversation?'
                      )
                    ) {
                      toast.success('Conversation deleted');
                      StorageUtils.remove(conv.id);
                      navigate('/');
                    }
                  }}
  
                  onDownload={() => {
                    if (isGenerating(conv.id)) {
                      toast.error(
                        'Cannot download conversation while generating'
                      );
                      return;
                    }
					// Get the current system message from config
					const systemMessage = StorageUtils.getConfig().systemMessage;
					
					// Clone the viewingChat object to avoid modifying the original
					const exportData = {
					  conv: { ...viewingChat?.conv },
					  messages: viewingChat?.messages.map(msg => ({ ...msg }))
					};
					
					// Find the root message and update its content
					const rootMessage = exportData.messages?.find(m => m.type === 'root');
					if (rootMessage) {
					  rootMessage.content = systemMessage;
					}


					const conversationJson = JSON.stringify(exportData, null, 2);
                    // const conversationJson = JSON.stringify(conv, null, 2);
                    const blob = new Blob([conversationJson], {
                      type: 'application/json',
                    });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `conversation_${conv.id}.json`;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);
                  }}
                  onRename={async () => {
                    if (isGenerating(conv.id)) {
                      toast.error(
                        'Cannot rename conversation while generating'
                      );
                      return;
                    }
                    const newName = await showPrompt(
                      'Enter new name for the conversation',
                      conv.name
                    );
                    if (newName && newName.trim().length > 0) {
                      StorageUtils.updateConversationName(conv.id, newName);
                    }
                  }}
                />
              ))}
            </div>
          ))}
          <div className="text-center text-xs opacity-40 mt-auto mx-4 pt-8">
            Conversations are saved to browser's IndexedDB
          </div>
        </div>
      </div>
    </>
  );
}

function ConversationItem({
  conv,
  isCurrConv,
  onSelect,
  onDelete,
  onDownload,
  onRename,
}: {
  conv: Conversation;
  isCurrConv: boolean;
  onSelect: () => void;
  onDelete: () => void;
  onDownload: () => void;
  onRename: () => void;
}) {
  return (
    <div
      role="menuitem"
      tabIndex={0}
      aria-label={conv.name}
      className={classNames({
        'group flex flex-row btn btn-ghost justify-start items-center font-normal px-2 h-9':
          true,
        'btn-soft': isCurrConv,
      })}
	  onClick={onSelect}
    >
      <button
        key={conv.id}
        className="w-full overflow-hidden truncate text-start"        
        dir="auto"
      >
        {conv.name}
      </button>
      <div className="dropdown dropdown-end h-5">
        <BtnWithTooltips
          // on mobile, we always show the ellipsis icon
          // on desktop, we only show it when the user hovers over the conversation item
          // we use opacity instead of hidden to avoid layout shift
          className="cursor-pointer opacity-100 xl:opacity-0 group-hover:opacity-100"
          onClick={() => {}}
          tooltipsContent="More"
        >
          <EllipsisVerticalIcon className="w-5 h-5" />
        </BtnWithTooltips>
        {/* dropdown menu */}
        <ul
          aria-label="More options"
          tabIndex={0}
          className="dropdown-content menu bg-base-100 rounded-box z-[1] p-2 shadow"
        >
		{/* Always show Upload when viewingChat is false */}

		  <li onClick={importConversation}>
			  <a>
			  <ArrowUpTrayIcon className="w-4 h-4" />
			  Upload
			  </a>
			</li>
          <li onClick={onRename} tabIndex={0}>
            <a>
              <PencilIcon className="w-4 h-4" />
              Rename
            </a>
          </li>
          <li onClick={onDownload} tabIndex={0}>
            <a>
              <ArrowDownTrayIcon className="w-4 h-4" />
              Download
            </a>
          </li>
          <li className="text-error" onClick={onDelete} tabIndex={0}>
            <a>
              <TrashIcon className="w-4 h-4" />
              Delete
            </a>
          </li>
        </ul>
      </div>
    </div>
  );
}

// WARN: vibe code below

export interface GroupedConversations {
  title?: string;
  conversations: Conversation[];
}

// TODO @ngxson : add test for this function
// Group conversations by date
// - Yesterday
// - "Previous 7 Days"
// - "Previous 30 Days"
// - "Month Year" (e.g., "April 2023")
export function groupConversationsByDate(
  conversations: Conversation[]
): GroupedConversations[] {
  
  const today=DateTime.now().startOf('day');
  const yesterday = today.minus({ days: 1 });

  const yesterday2 = today.minus({ days: 2});

  const sevenDaysAgo = today.minus({ days: 7 });

  const thirtyDaysAgo = today.minus({ days: 30 });
  const groups: { [key: string]: Conversation[] } = {
    Today: [],
    Yesterday: [],
    'Previous 2 Days': [],
    'Previous 7 Days': [],
    'Previous 30 Days': [],
  };
  const monthlyGroups: { [key: string]: Conversation[] } = {}; // Key format: "Month Year" e.g., "April 2023"

  // Sort conversations by lastModified date in descending order (newest first)
  // This helps when adding to groups, but the final output order of groups is fixed.
  const sortedConversations = [...conversations].sort(
    (a, b) => b.lastModified - a.lastModified
  );

  for (const conv of sortedConversations) {
    const convDate=DateTime.fromMillis(conv.lastModified).setZone('America/Chicago');
    if (convDate >= today) {
      groups['Today'].push(conv);
    } else if (convDate >= yesterday) {
      groups['Yesterday'].push(conv);
    } else if (convDate >= yesterday2) {
      groups['Previous 2 Days'].push(conv);
    } else if (convDate >= sevenDaysAgo) {
      groups['Previous 7 Days'].push(conv);
    } else if (convDate >= thirtyDaysAgo) {
      groups['Previous 30 Days'].push(conv);
    } else {
      const monthName = convDate.monthLong;
      const year = convDate.year;
      const monthYearKey = `${monthName} ${year}`;
      if (!monthlyGroups[monthYearKey]) {
        monthlyGroups[monthYearKey] = [];
      }
      monthlyGroups[monthYearKey].push(conv);
    }
  }

  const result: GroupedConversations[] = [];

  if (groups['Today'].length > 0) {
    result.push({
      title: undefined, // no title for Today
      conversations: groups['Today'],
    });
  }
const timeRanges = [
  { key: 'Yesterday',       display:  'Yesterday'},
  { key: 'Previous 2 Days', display:  'Previous 2 Days'},
  { key: 'Previous 7 Days', display:  'Previous 7 Days' },
  { key: 'Previous 30 Days', display: 'Previous 30 Days' },

  // Add more ranges here if needed, e.g., 'Previous 90 Days'
];

for (const range of timeRanges) {
  if (groups[range.key]?.length > 0) {
    result.push({
      title: range.display,
      conversations: groups[range.key]
    });
  }
}

  // Sort monthly groups by date (most recent month first)
  const sortedMonthKeys = Object.keys(monthlyGroups).sort((a, b) => {
    const dateA = new Date(a); // "Month Year" can be parsed by Date constructor
    const dateB = new Date(b);
    return dateB.getTime() - dateA.getTime();
  });

  for (const monthKey of sortedMonthKeys) {
    if (monthlyGroups[monthKey].length > 0) {
      result.push({ title: monthKey, conversations: monthlyGroups[monthKey] });
    }
  }

  return result;
}
