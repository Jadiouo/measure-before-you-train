// Bundles src/viewer/app.js (+ sim modules) into ONE self-contained dist/index.html (classic script, no modules, no fetch),
// so it opens by double-click from file:// as well as from any static host.
import { build } from 'esbuild';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';

const js = await build({ entryPoints: ['src/viewer/app.js'], bundle: true, minify: true, format: 'iife', target: 'es2019', write: false, legalComments: 'none' });
const code = js.outputFiles[0].text.replace(/<\/script/gi, '<\\/script');
const css = readFileSync('src/viewer/style.css', 'utf8');
const html = readFileSync('index.html', 'utf8').replace('/*CSS*/', () => css).replace('/*JS*/', () => code);
mkdirSync('dist', { recursive: true });
writeFileSync('dist/index.html', html);
console.log(`dist/index.html  ${(html.length / 1024).toFixed(1)} KiB`);
