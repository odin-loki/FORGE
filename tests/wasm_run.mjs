// Run a wasm32-wasi command module under node's WASI (tests/wasm_simd.sh): node tests/wasm_run.mjs <file.wasm> [args]
import { readFile } from 'node:fs/promises';
import { WASI } from 'node:wasi';
import { argv, env, exit } from 'node:process';

const wasi = new WASI({ version: 'preview1', args: argv.slice(2), env, preopens: { '.': '.' }, returnOnExit: true });
const module = await WebAssembly.compile(await readFile(argv[2]));
const instance = await WebAssembly.instantiate(module, wasi.getImportObject());
exit(wasi.start(instance));
