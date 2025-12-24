import React, { useEffect } from 'react';
//import { throttle } from '../utils/misc';

let autoScrollPaused = false; 

export const scrollToBottom = (requiresNearBottom: boolean, delay?: number) => {
  const mainScrollElem = document.getElementById('main-scroll');
  if (!mainScrollElem) return;
  const spaceToBottom =
    mainScrollElem.scrollHeight -
    mainScrollElem.scrollTop -
    mainScrollElem.clientHeight;
  if (!requiresNearBottom || spaceToBottom < 100) {
    if (!autoScrollPaused) {
      setTimeout(
        () => mainScrollElem.scrollTo({
          top: mainScrollElem.scrollHeight,
          behavior: 'smooth'
        }),
        delay ?? 80
      );
    }
  }
};

//const scrollToBottomThrottled = throttle(scrollToBottom, 80);

export function useChatScroll(msgListRef: React.RefObject<HTMLDivElement>) {
  useEffect(() => {
    if (!msgListRef.current) return;

    const resizeObserver = new ResizeObserver((_) => {
      // Remove throttle but keep the near-bottom logic
      scrollToBottom(true, 10);
    });

    const mainScrollElem = document.getElementById('main-scroll');
    if (!mainScrollElem) return;

    // Initialize handleWheel event listener to detect user scrolling actions
    const handleWheel = (event: WheelEvent) => {
      if (event.deltaY < 0) {
        // User scrolled up
        autoScrollPaused = true;
      } else {
        // User scrolled down
        autoScrollPaused = false;
      }
    };
    // Add event listener for wheel events
    mainScrollElem.addEventListener('wheel', handleWheel);

    resizeObserver.observe(msgListRef.current);
    // Observe the msgListRef element for size changes
    return () => {
      resizeObserver.disconnect();
      mainScrollElem.removeEventListener('wheel', handleWheel);
    };
  }, [msgListRef]);
}