import fs from 'node:fs/promises';
import { config } from './config.js';
import { createApp } from './app.js';
import { createFileRetentionService } from './services/fileRetentionService.js';
import { PythonWorkerClient } from './services/pythonWorker.js';
import { outputsDir, uploadsDir } from './utils/paths.js';

async function main() {
  await Promise.all([
    fs.mkdir(uploadsDir, { recursive: true }),
    fs.mkdir(outputsDir, { recursive: true }),
  ]);

  const worker = new PythonWorkerClient({
    pythonExecutable: config.pythonExecutable,
    scriptPath: config.pythonScript,
    modelDir: config.modelDir,
    potraceBin: config.potraceBin,
    timeoutMs: config.workerJobTimeoutMs,
  });

  worker.start().catch((error) => {
    console.error('[startup] Python worker failed:', error.message);
  });

  const retention = createFileRetentionService({
    uploadsDir,
    outputsDir,
    retentionHours: config.outputRetentionHours,
  });
  await retention.start();

  const app = createApp({ config, uploadsDir, outputsDir, worker });
  const server = app.listen(config.port, () => {
    console.info(`Backend listening on http://localhost:${config.port}`);
  });

  async function shutdown(signal) {
    console.info(`[shutdown] ${signal}`);
    await retention.stop();
    await worker.shutdown();
    await new Promise((resolve) => server.close(resolve));
    process.exit(0);
  }

  process.on('SIGINT', () => void shutdown('SIGINT'));
  process.on('SIGTERM', () => void shutdown('SIGTERM'));
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
