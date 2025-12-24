import { useEffect, useMemo, useRef, useState } from 'react';
import toast from 'react-hot-toast';
import { CallbackGeneratedChunk, useAppContext } from '../utils/app.context';
import ChatMessage from './ChatMessage';
import { CanvasType, Message, PendingMessage } from '../utils/types';
import { classNames, cleanCurrentUrl } from '../utils/misc';
import CanvasPyInterpreter from './CanvasPyInterpreter';
import StorageUtils from '../utils/storage';
import { useVSCodeContext } from '../utils/llama-vscode';
import { useChatTextarea, ChatTextareaApi } from './useChatTextarea.ts';
import { scrollToBottom, useChatScroll } from './useChatScroll.tsx';
import {
  ArrowUpIcon,
  StopIcon,
  PaperClipIcon,
} from '@heroicons/react/24/solid';
import {
  ChatExtraContextApi,
  useChatExtraContext,
} from './useChatExtraContext.tsx';
import Dropzone from 'react-dropzone';
import ChatInputExtraContextItem from './ChatInputExtraContextItem.tsx';
/**
 * A message display is a message node with additional information for rendering.
 * For example, siblings of the message node are stored as their last node (aka leaf node).
 */
export interface MessageDisplay {
  msg: Message | PendingMessage;
  siblingLeafNodeIds: Message['id'][];
  siblingCurrIdx: number;
  isPending?: boolean;
}

/**
 * If the current URL contains "?m=...", prefill the message input with the value.
 * If the current URL contains "?q=...", prefill and SEND the message.
 */
const prefilledMsg = {
  content() {
    const url = new URL(window.location.href);
    return url.searchParams.get('m') ?? url.searchParams.get('q') ?? '';
  },
  shouldSend() {
    const url = new URL(window.location.href);
    return url.searchParams.has('q');
  },
  clear() {
    cleanCurrentUrl(['m', 'q']);
  },
};

function getListMessageDisplay(
  msgs: Readonly<Message[]>,
  leafNodeId: Message['id']
): MessageDisplay[] {
  const currNodes = StorageUtils.filterByLeafNodeId(msgs, leafNodeId, true);
  const res: MessageDisplay[] = [];
  const nodeMap = new Map<Message['id'], Message>();
  for (const msg of msgs) {
    nodeMap.set(msg.id, msg);
  }
  // find leaf node from a message node
  const findLeafNode = (msgId: Message['id']): Message['id'] => {
    let currNode: Message | undefined = nodeMap.get(msgId);
    while (currNode) {
      if (currNode.children.length === 0) break;
      currNode = nodeMap.get(currNode.children.at(-1) ?? -1);
    }
    return currNode?.id ?? -1;
  };
  // traverse the current nodes
  for (const msg of currNodes) {
    const parentNode = nodeMap.get(msg.parent ?? -1);
    if (!parentNode) continue;
    const siblings = parentNode.children;
    if (msg.type !== 'root') {
      res.push({
        msg,
        siblingLeafNodeIds: siblings.map(findLeafNode),
        siblingCurrIdx: siblings.indexOf(msg.id),
      });
    }
  }
  return res;
}


export default function ChatScreen() {
  const {
    viewingChat,
    sendMessage,
    isGenerating,
    stopGenerating,
    pendingMessages,
    canvasData,
    replaceMessageAndGenerate,
    continueMessageAndGenerate,
  } = useAppContext();

  const textarea: ChatTextareaApi = useChatTextarea(prefilledMsg.content());

  const extraContext = useChatExtraContext();
  useVSCodeContext(textarea, extraContext);
  
  const msgListRef = useRef<HTMLDivElement>(null);
  useChatScroll(msgListRef);
  // TODO: improve this when we have "upload file" feature
  // keep track of leaf node for rendering
  const [currNodeId, setCurrNodeId] = useState<number>(-1);
  const messages: MessageDisplay[] = useMemo(() => {
    if (!viewingChat) return [];
    else return getListMessageDisplay(viewingChat.messages, currNodeId);
  }, [currNodeId, viewingChat]);

  const currConvId = viewingChat?.conv.id ?? null;
  const pendingMsg: PendingMessage | undefined =
    pendingMessages[currConvId ?? ''];

  useEffect(() => {
    // reset to latest node when conversation changes
    setCurrNodeId(-1);
    // scroll to bottom when conversation changes
    // scrollToBottom(false, 1);
  }, [currConvId]);

  const onChunk: CallbackGeneratedChunk = (currLeafNodeId?: Message['id']) => {
    if (currLeafNodeId) {
      setCurrNodeId(currLeafNodeId);
    }
    //useChatScroll will handle the auto scroll
  };

  const sendNewMessage = async () => {
  
  const lastInpMsg = textarea.value();
  try {
    const generate = isGenerating(currConvId ?? '');
    console.log('IsGenerating', generate);
    if (lastInpMsg.trim().length === 0 || generate)
      return;

    textarea.setValue('');
    setCurrNodeId(-1);
    scrollToBottom(false);
    // get the last message node
    const lastMsgNodeId = messages.at(-1)?.msg.id ?? null;
    const successSendMsg=await sendMessage(
        currConvId,
        lastMsgNodeId,
        lastInpMsg,
        extraContext.items,
        onChunk
      );
    console.log('Send msg success:', successSendMsg);
    if (!successSendMsg)
    {
      // restore the input message if failed
      textarea.setValue(lastInpMsg);
    }
    // OK
    extraContext.clearItems();
    } 
     catch (err) {
        //console.error('Error sending message:', error);
        toast.error(err instanceof Error ? err.message : String(err));
        textarea.setValue(lastInpMsg); // Restore input on error
    }
  };

  const handleEditMessage = async (msg: Message, content: string) => {
    if (!viewingChat) return;
    setCurrNodeId(msg.id);
    scrollToBottom(false);
    await replaceMessageAndGenerate(
      viewingChat.conv.id,
      msg.parent,
      content,
      msg.extra,
      onChunk
    );
    setCurrNodeId(-1);
    scrollToBottom(false);

  };

  const handleRegenerateMessage = async (msg: Message) => {
    if (!viewingChat) return;
    setCurrNodeId(msg.parent);
    scrollToBottom(false);
    await replaceMessageAndGenerate(
      viewingChat.conv.id,
      msg.parent,
      null,
      msg.extra,
      onChunk
    );
    setCurrNodeId(-1);
    scrollToBottom(false);

  };

  const handleContinueMessage = async (msg: Message, content: string) => {
    if (!viewingChat || !continueMessageAndGenerate) return;
    setCurrNodeId(msg.id);
    scrollToBottom(false);
    await continueMessageAndGenerate(
      viewingChat.conv.id,
      msg.id,
      content,
      onChunk
    );
    setCurrNodeId(-1);
    scrollToBottom(false);

  };

  const hasCanvas = !!canvasData;

  useEffect(() => {
    if (prefilledMsg.shouldSend()) {
      // send the prefilled message if needed
      sendNewMessage();
    } else {
      // otherwise, focus on the input
      textarea.focus();
    }
    prefilledMsg.clear();
    // no need to keep track of sendNewMessage
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [textarea.ref]);

  // due to some timing issues of StorageUtils.appendMsg(), we need to make sure the pendingMsg is not duplicated upon rendering (i.e. appears once in the saved conversation and once in the pendingMsg)
  const pendingMsgDisplay: MessageDisplay[] =
    pendingMsg && !messages.some((m) => m.msg.id === pendingMsg.id) // Only show if pendingMsg is not an existing message being continued
      ? [
          {
            msg: pendingMsg,
            siblingLeafNodeIds: [],
            siblingCurrIdx: 0,
            isPending: true,
          },
        ]
      : [];

  return (
    <div
      className={classNames({
        'grid lg:gap-8 grow transition-[300ms]': true,
        'grid-cols-[1fr_0fr] lg:grid-cols-[1fr_1fr]': hasCanvas, // adapted for mobile
        'grid-cols-[1fr_0fr]': !hasCanvas,
      })}
    >
      <div
        className={classNames({
          'flex flex-col w-full lg:w-[75vw] lg:mx-auto': true, // Changed here
          'hidden lg:flex': hasCanvas, // adapted for mobile
          flex: !hasCanvas,
        })}
      >
	  <div className="flex items-center justify-center">
		{viewingChat?.conv.model_name}
	  </div>
        {/* chat messages */}
        <div id="messages-list" className="grow" ref={msgListRef}>
          <div className="mt-auto flex justify-center">
            {/* placeholder to shift the message to the bottom */}
            <div>
            {viewingChat ? '' : ''}
            </div>
            {viewingChat==null && (
              <div className="w-full max-w-2xl px-4">
                <div className="mb-8 text-center" >
                  <p className="text-1xl text-muted-foreground">How can I help you today?</p>
                </div>
              </div>
            )}

          </div>
          {[...messages, ...pendingMsgDisplay].map((msgDisplay) => {
            const actualMsgObject = msgDisplay.msg;
            // Check if the current message from the list is the one actively being generated/continued
            const isThisMessageTheActivePendingOne =
              pendingMsg?.id === actualMsgObject.id;

            return (
              <ChatMessage
                key={actualMsgObject.id}
                // If this message is the active pending one, use the live object from pendingMsg state (which has streamed content).
                // Otherwise, use the version from the messages array (from storage).
                msg={
                  isThisMessageTheActivePendingOne
                    ? pendingMsg
                    : actualMsgObject
                }
                siblingLeafNodeIds={msgDisplay.siblingLeafNodeIds}
                siblingCurrIdx={msgDisplay.siblingCurrIdx}
                onRegenerateMessage={handleRegenerateMessage}
                onEditMessage={handleEditMessage}
                onChangeSibling={setCurrNodeId}
                // A message is pending if it's the actively streaming one OR if it came from pendingMsgDisplay (for new messages)
                isPending={
                  isThisMessageTheActivePendingOne || msgDisplay.isPending
                }
                onContinueMessage={handleContinueMessage}
              />
            );
          })}        
        </div>
        {/* chat input */}
        <ChatInput
          textarea={textarea}
          extraContext={extraContext}
          onSend={sendNewMessage}
          onStop={() => stopGenerating(currConvId ?? '')}
          isGenerating={isGenerating(currConvId ?? '')}
        />
        </div>
      <div className="w-full sticky top-[7em] h-[calc(100vh-9em)]">
        {canvasData?.type === CanvasType.PY_INTERPRETER && (
          <CanvasPyInterpreter />
        )}
      </div>
    </div>
  );
}


function ChatInput({
  textarea,
  extraContext,
  onSend,
  onStop,
  isGenerating,
}: {
  textarea: ChatTextareaApi;
  extraContext: ChatExtraContextApi;
  onSend: () => void;
  onStop: () => void;
  isGenerating: boolean;
}) {
  const { config } = useAppContext();
  const [isDrag, setIsDrag] = useState(false);

  return (
    <div
      role="group"
      aria-label="Chat input"
      className={classNames({
        'flex items-end pt-8 pb-6 sticky bottom-0 bg-base-100': true,
        'opacity-50': isDrag, // simply visual feedback to inform user that the file will be accepted
      })}
    >
      <Dropzone
        noClick
        onDrop={(files: File[]) => {
          setIsDrag(false);
          extraContext.onFileAdded(files);
        }}
        onDragEnter={() => setIsDrag(true)}
        onDragLeave={() => setIsDrag(false)}
        multiple={true}
      >
        {({ getRootProps, getInputProps }) => (
          <div
            className="flex flex-col rounded-xl border-1 border-base-content/30 p-3 w-full"
            // when a file is pasted to the input, we handle it here
            // if a text is pasted, and if it is long text, we will convert it to a file
            onPasteCapture={(e: React.ClipboardEvent<HTMLInputElement>) => {
              const text = e.clipboardData.getData('text/plain');
              if (
                text.length > 0 &&
                config.pasteLongTextToFileLen > 0 &&
                text.length > config.pasteLongTextToFileLen
              ) {
                // if the text is too long, we will convert it to a file
                extraContext.addItems([
                  {
                    type: 'context',
                    name: 'Pasted Content',
                    content: text,
                  },
                ]);
                e.preventDefault();
                return;
              }

              // if a file is pasted, we will handle it here
              const files = Array.from(e.clipboardData.items)
                .filter((item) => item.kind === 'file')
                .map((item) => item.getAsFile())
                .filter((file) => file !== null);

              if (files.length > 0) {
                e.preventDefault();
                extraContext.onFileAdded(files);
              }
            }}
            {...getRootProps()}
          >
            {!isGenerating && (
              <ChatInputExtraContextItem
                items={extraContext.items}
                removeItem={extraContext.removeItem}
              />
            )}

            <div className="flex flex-row w-full">
              <textarea
                // Default (mobile): Enable vertical resize, overflow auto for scrolling if needed
                // Large screens (lg:): Disable manual resize, apply max-height for autosize limit
                className="text-md outline-none border-none w-full resize-vertical lg:resize-none lg:max-h-48 lg:overflow-y-auto" // Adjust lg:max-h-48 as needed (e.g., lg:max-h-60)
                placeholder="Type a message..."
                ref={textarea.ref}
                onInput={textarea.onInput} // Hook's input handler (will only resize height on lg+ screens)
                onKeyDown={(e) => {
                  if (e.nativeEvent.isComposing || e.keyCode === 229) return;
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    onSend();
                  }
                }}
                id="msg-input"
                dir="auto"
                // Set a base height of 2 rows for mobile views
                // On lg+ screens, the hook will calculate and set the initial height anyway
                rows={2}
              ></textarea>

              {/* buttons area */}
              <div className="flex flex-row gap-2 ml-2">
                <label
                  htmlFor="file-upload"
                  className={classNames({
                    'btn w-8 h-8 p-0 rounded-full': true,
                    'btn-disabled': isGenerating,
                  })}
                  aria-label="Upload file"
                  tabIndex={0}
                  role="button"
                >
                  <PaperClipIcon className="h-5 w-5" />
                </label>
                <input
                  id="file-upload"
                  type="file"
                  disabled={isGenerating}
                  {...getInputProps()}
                  hidden
                />
                {isGenerating ? (
                  <button
                    className="btn btn-neutral w-8 h-8 p-0 rounded-full"
                    onClick={onStop}
                  >
                    <StopIcon className="h-5 w-5" />
                  </button>
                ) : (
                  <button
                    className="btn btn-primary w-8 h-8 p-0 rounded-full"
                    onClick={onSend}
                    aria-label="Send message"
                  >
                    <ArrowUpIcon className="h-5 w-5" />
                  </button>
                )}
              </div>
            </div>
          </div>
        )}
      </Dropzone>
    </div>
  );
}