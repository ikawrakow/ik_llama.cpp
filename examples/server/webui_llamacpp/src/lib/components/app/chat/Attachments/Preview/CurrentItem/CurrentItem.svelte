<script lang="ts">
	import type { ChatAttachmentDisplayItem } from '$lib/types';
	import { Image, Music, Video, FileText, FileIcon } from '@lucide/svelte';
	import PdfPreview from './Pdf.svelte';
	import ImagePreview from './Image.svelte';
	import AudioPreview from './Audio.svelte';
	import VideoPreview from './Video.svelte';
	import TextPreview from './Text.svelte';
	import UnavailablePreview from './Unavailable.svelte';

	interface Props {
		currentItem: ChatAttachmentDisplayItem | null;
		isImage: boolean;
		isAudio: boolean;
		isVideo: boolean;
		isPdf: boolean;
		isText: boolean;
		displayPreview: string | undefined;
		displayTextContent: string | undefined;
		audioSrc: string | null;
		videoSrc: string | null;
		language: string;
		hasVisionModality: boolean;
		activeModelId?: string;
	}

	let {
		currentItem,
		isImage,
		isAudio,
		isVideo,
		isPdf,
		isText,
		displayPreview,
		displayTextContent,
		audioSrc,
		videoSrc,
		language,
		hasVisionModality,
		activeModelId
	}: Props = $props();

	let IconComponent = $derived(
		isImage ? Image : isText || isPdf ? FileText : isAudio ? Music : isVideo ? Video : FileIcon
	);

	let isUnavailable = $derived(
		!isPdf && !isImage && !(isText && displayTextContent) && !isAudio && !isVideo
	);
</script>

{#if currentItem}
	{#key currentItem.id}
		{#if isPdf}
			<PdfPreview
				{currentItem}
				displayName={currentItem.name}
				{displayTextContent}
				{hasVisionModality}
				{activeModelId}
			/>
		{:else if isImage}
			<ImagePreview {currentItem} {displayPreview} />
		{:else if isText && displayTextContent}
			<TextPreview {displayTextContent} {language} />
		{:else if isAudio}
			<AudioPreview {currentItem} {audioSrc} />
		{:else if isVideo}
			<VideoPreview {currentItem} {videoSrc} />
		{:else if isUnavailable}
			<UnavailablePreview {IconComponent} />
		{/if}
	{/key}
{/if}
