import fs from 'node:fs';
import path from 'node:path';
import { localPotraceBin, modelsDir, pythonEntry } from './utils/paths.js';

function toInteger(value, fallback) {
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

export const config = {
  port: toInteger(process.env.PORT, 3001),
  maxFileSizeBytes: 20 * 1024 * 1024,
  pythonExecutable: process.env.PYTHON_EXECUTABLE || 'python',
  pythonScript: process.env.PYTHON_SCRIPT || pythonEntry,
  modelDir: process.env.MODEL_DIR || modelsDir,
  potraceBin: process.env.POTRACE_BIN || (fs.existsSync(localPotraceBin) ? localPotraceBin : 'potrace'),
  outputRetentionHours: toInteger(process.env.OUTPUT_RETENTION_HOURS, 24),
  workerJobTimeoutMs: toInteger(process.env.WORKER_JOB_TIMEOUT_MS, 15 * 60 * 1000),
  allowedMimeTypes: new Set(['image/jpeg', 'image/png', 'image/webp']),
};

export function resolveModelPath(filename) {
  return path.join(config.modelDir, filename);
}
