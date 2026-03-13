import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { backendRoot, localPotraceBin, projectRoot, pythonEntry } from '../src/utils/paths.js';

test('paths resolve from repo root correctly', () => {
  assert.equal(path.basename(backendRoot), 'backend');
  assert.equal(path.basename(projectRoot), 'gambar');
  assert.equal(path.basename(path.dirname(pythonEntry)), 'ai-engine');
  assert.equal(path.basename(pythonEntry), 'pipeline.py');
  assert.equal(path.basename(localPotraceBin), 'potrace.exe');
});
