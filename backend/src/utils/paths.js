import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export const backendRoot = path.resolve(__dirname, '..');
export const projectRoot = path.resolve(backendRoot, '..');
export const uploadsDir = path.join(projectRoot, 'uploads');
export const outputsDir = path.join(projectRoot, 'outputs');
export const modelsDir = path.join(projectRoot, 'models');
export const pythonEntry = path.join(projectRoot, 'ai-engine', 'pipeline.py');

export function resolveOutputFile(jobId, name) {
  return path.join(outputsDir, jobId, name);
}

