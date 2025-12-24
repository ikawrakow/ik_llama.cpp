import { useMemo, useState, useRef, useEffect } from 'react';
import { useAppContext } from '../utils/app.context';
import { Message, PendingMessage } from '../utils/types';
import { classNames } from '../utils/misc';
import MarkdownDisplay, { CopyButton } from './MarkdownDisplay';
import { ChevronLeftIcon, ChevronRightIcon,  ArrowPathIcon, PencilSquareIcon } from '@heroicons/react/24/outline';
import ChatInputExtraContextItem from './ChatInputExtraContextItem';
import TextareaAutosize from 'react-textarea-autosize';

interface SplitMessage {
  content: PendingMessage['content'];
  thought?: string;
  isThinking?: boolean;
}

export default function ChatMessage({
  msg,
  siblingLeafNodeIds,
  siblingCurrIdx,
  id,
  onRegenerateMessage,
  onEditMessage,
  onChangeSibling,
  isPending,
  onContinueMessage,
}: {
  msg: Message | PendingMessage;
  siblingLeafNodeIds: Message['id'][];
  siblingCurrIdx: number;
  id?: string;
  onRegenerateMessage(msg: Message): void;
  onEditMessage(msg: Message, content: string): void;
  onContinueMessage(msg: Message, content: string): void;
  onChangeSibling(sibling: Message['id']): void;
  isPending?: boolean;
}) {
  const { viewingChat, config } = useAppContext();
  const [editingContent, setEditingContent] = useState<string | null>(null);  
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (msg.content=== null) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [msg.content]);
  const timings = useMemo(
    () =>
      msg.timings
        ? {
            ...msg.timings,
            prompt_per_second:
              (msg.timings.prompt_n / msg.timings.prompt_ms) * 1000,
            predicted_per_second:
              (msg.timings.predicted_n / msg.timings.predicted_ms) * 1000,
          }
        : null,
    [msg.timings]
  );
  const nextSibling = siblingLeafNodeIds[siblingCurrIdx + 1];
  const prevSibling = siblingLeafNodeIds[siblingCurrIdx - 1];
  // for reasoning model, we split the message into content and thought
  // TODO: implement this as remark/rehype plugin in the future
  const { content, thought, isThinking }: SplitMessage = useMemo(() => {
    if (msg.content === null || msg.role !== 'assistant') {
      return { content: msg.content };
    }      

    const REGEX_THINK_OPEN = /<think>|<\|channel\|>analysis<\|message\|>/;
    const REGEX_THINK_CLOSE = /<\/think>|<\|end\|>/;
    let actualContent = '';
    let thought = '';
    let isThinking = false;
    let thinkSplit = msg.content.split(REGEX_THINK_OPEN, 2);
    actualContent += thinkSplit[0];
    while (thinkSplit[1] !== undefined) {
      // <think> tag found
      thinkSplit = thinkSplit[1].split(REGEX_THINK_CLOSE, 2);
      thought += thinkSplit[0];
      isThinking = true;
      if (thinkSplit[1] !== undefined) {
        // </think> closing tag found
        isThinking = false;
        thinkSplit = thinkSplit[1].split(REGEX_THINK_OPEN, 2);
        actualContent += thinkSplit[0];
      }
    }
    return { content: actualContent, thought, isThinking };
  }, [msg]);

  if (!viewingChat) return null;
  //const model_name = (timings?.model_name ??'')!== '' ? timings?.model_name: viewingChat.conv.model_name;
  const model_name = viewingChat.conv.model_name;
  return (
    <div className="group" 
      id={id}       
      role="group"
      aria-description={`Message from ${msg.role}`}
    >
      <div
        className={classNames({
          chat: true,
          'chat-start': msg.role !== 'user',
          'chat-end': msg.role === 'user',
        })}
      >
        {msg.extra && msg.extra.length > 0 && (
          <ChatInputExtraContextItem items={msg.extra} clickToShow />
        )}

        <div
          className={classNames({
            'chat-bubble chat-bubble-primary': true,
            'chat-bubble-base-300': msg.role !== 'user',
          })}
        >
          {/* textarea for editing message */}
          {editingContent !== null && (
            <>
              <TextareaAutosize
                dir="auto"
                className="textarea textarea-bordered bg-base-100 text-base-content max-w-2xl w-[calc(90vw-8em)]"
                value={editingContent}
                onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setEditingContent(e.target.value)}
                minRows={3}
                maxRows={15}
              />
              <br />
              <button
                className="btn btn-ghost mt-2 mr-2"
                onClick={() => setEditingContent(null)}
              >
                Cancel
              </button>
              <button
                className="btn mt-2"
                onClick={() => {
                  if (msg.content !== null) {
                    setEditingContent(null);
                    if (msg.role === 'user') {
                      onEditMessage(msg as Message, editingContent);
                    } else {
                      onContinueMessage(msg as Message, editingContent);
                    }
                  }
                }}
              >
                Submit
              </button>
            </>
          )}
          {/* not editing content, render message */}
          {editingContent === null && (
            <>
              {content === null ? (
                <>
                  {/* show loading dots for pending message */}
                  <span className="loading loading-dots loading-md"></span>
                </>
              ) : (
                <>
                  {/* render message as markdown */}
                  <div dir="auto">
                    {thought && (
                      <details
                        className="collapse bg-base-200 collapse-arrow mb-4"
                        open={isThinking && config.showThoughtInProgress}
                      >
                        <summary className="collapse-title">
                          {isPending && isThinking ? (
                            <span>
                              <span
                                v-if="isGenerating"
                                className="loading loading-spinner loading-md mr-2"
                                style={{ verticalAlign: 'middle' }}
                              ></span>
                              <b>Thinking</b>
                            </span>
                          ) : (
                            <b>Thought Process</b>
                          )}
                        </summary>
                        <div className="collapse-content">
                          <MarkdownDisplay
                            content={thought}
                            isGenerating={isPending}
                          />
                        </div>
                      </details>
                    )}
                    <MarkdownDisplay
                      content={content}
                      isGenerating={isPending}
                    />
                  </div>
                </>
              )}
              {/* render timings if enabled */}
              {timings && config.showTokensPerSecond && (
                <div className="dropdown dropdown-hover dropdown-top ax-w-[900px] w-full mt-4">
                  <div
                    tabIndex={0}
                    role="button"
                    className="cursor-pointer font-semibold text-sm opacity-60"
                  >
                    <div className="font-bold text-xs">                    
                      {timings.n_ctx>0 && (       
                        <div className="flex justify-between items-center">
                          <span className="whitespace-nowrap">
                            Token: {timings.predicted_per_second.toFixed(1)} t/s | Prompt: {timings.prompt_per_second.toFixed(1)} t/s
                          </span>
                          <span className="hidden lg:block pl-[200px] whitespace-nowrap">
                            Ctx: {timings.predicted_n+timings.prompt_n} / {timings.n_past} / {timings.n_ctx}
                          </span>
                        </div>
                      )}
                      {(timings.n_ctx==null || timings.n_ctx <=0) && (
                      <div>
                        Token: {timings.predicted_per_second.toFixed(1)} t/s | Prompt: {timings.prompt_per_second.toFixed(1)} t/s
                      </div>
                      )}
                    </div>
                  </div>
                  <div className="dropdown-content bg-base-100 z-10 w-64 p-2 shadow mt-4">
                    <p className="text-xs"><b>{model_name}</b></p>
                    <p className="text-sm">
                    <b>Prompt</b>
                    <br />- Tokens: {timings.prompt_n}
                    <br />- Time: {timings.prompt_ms} ms
                    <br />- Speed: {timings.prompt_per_second.toFixed(2)} t/s
                    <br />
                    <b>Generation</b>
                    <br />- Tokens: {timings.predicted_n}
                    <br />- Time: {timings.predicted_ms} ms
                    <br />- Speed: {timings.predicted_per_second.toFixed(2)} t/s
                    <br />
                    <b>Context</b>
                    <br />- n_ctx: {timings.n_ctx}
                    <br />- n_past: {timings.n_past}
                    <br />
                    </p>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </>
          )}
        </div>
      </div>

      {/* actions for each message */}
      {msg.content !== null && !config.showTokensPerSecond && (            
        msg.role === 'assistant' &&(
        <div className="badge border-none outline-none btn-mini show-on-hover mr-2">
          <p className="text-xs">Model: {model_name}</p>
        </div>
        )
      )}
      {msg.content !== null && (
        <div
          className={classNames({
            'flex items-center gap-2 mx-4 mt-2 mb-2': true,
            'flex-row-reverse': msg.role === 'user',
          })}
        >
          {siblingLeafNodeIds && siblingLeafNodeIds.length > 1 && (
            <div className="flex gap-1 items-center opacity-60 text-sm">
              <button
                className={classNames({
                  'btn btn-sm btn-ghost p-1': true,
                  'opacity-20': !prevSibling,
                })}
                onClick={() => prevSibling && onChangeSibling(prevSibling)}
              >
                <ChevronLeftIcon className="h-4 w-4" />
              </button>
              <span>
                {siblingCurrIdx + 1} / {siblingLeafNodeIds.length}
              </span>
              <button
                className={classNames({
                  'btn btn-sm btn-ghost p-1': true,
                  'opacity-20': !nextSibling,
                })}
                onClick={() => nextSibling && onChangeSibling(nextSibling)}
              >
                <ChevronRightIcon className="h-4 w-4" />
              </button>
            </div>
          )}
          {/* user message */}
          {msg.role === 'user' && (
            <button
              className="badge border-none outline-none btn-mini show-on-hover"
              onClick={() => setEditingContent(msg.content)}
              disabled={msg.content === null}
            >
              <PencilSquareIcon className="h-4 w-4" /> Edit
            </button>
          )}
          {/* assistant message */}
          {msg.role === 'assistant' && (
            <>
              {!isPending && (
                <button
                  className="badge border-none outline-none btn-mini show-on-hover mr-2"
                  onClick={() => {
                    if (msg.content !== null) {
                      onRegenerateMessage(msg as Message);
                    }
                  }}
                  disabled={msg.content === null}
                >
                 <ArrowPathIcon className="h-4 w-4" /> Regenerate
                </button>
              )}
              {!isPending && (
                <button
                  className="badge border-none outline-none btn-mini show-on-hover"
                  onClick={() => setEditingContent(msg.content)}
                  disabled={msg.content === null}
                >
                   <PencilSquareIcon className="h-4 w-4" /> Edit
                </button>
              )}
            </>
          )}
          <CopyButton
            className="badge border-none outline-none btn-mini show-on-hover mr-2"
            content={msg.content}
          />
        </div>
      )}
    </div>
  );
}
