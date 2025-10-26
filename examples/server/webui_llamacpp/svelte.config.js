import { mdsvex } from 'mdsvex';
import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	// Consult https://svelte.dev/docs/kit/integrations
	// for more information about preprocessors
	preprocess: [vitePreprocess(), mdsvex()],
	kit: {
		adapter: adapter({
			pages: '../public_llamacpp',
			assets: '../public_llamacpp',
			fallback: 'index_llamacpp.html',
			precompress: false,
			strict: true
		}),
		output: {
			bundleStrategy: 'inline'
		}
	},
	extensions: ['.svelte', '.svx']
};

export default config;
