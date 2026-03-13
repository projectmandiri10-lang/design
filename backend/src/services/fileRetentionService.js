import fs from 'node:fs/promises';
import path from 'node:path';

async function removeOlderThan(rootDir, cutoffTimestamp) {
  const entries = await fs.readdir(rootDir, { withFileTypes: true }).catch(() => []);

  await Promise.all(
    entries.map(async (entry) => {
      const targetPath = path.join(rootDir, entry.name);
      const stats = await fs.stat(targetPath).catch(() => null);
      if (!stats || stats.mtimeMs >= cutoffTimestamp) {
        return;
      }
      await fs.rm(targetPath, { recursive: true, force: true });
    }),
  );
}

export function createFileRetentionService({ uploadsDir, outputsDir, retentionHours }) {
  let intervalId = null;

  async function cleanup() {
    const cutoffTimestamp = Date.now() - retentionHours * 60 * 60 * 1000;
    await Promise.all([
      removeOlderThan(uploadsDir, cutoffTimestamp),
      removeOlderThan(outputsDir, cutoffTimestamp),
    ]);
  }

  return {
    async start() {
      await cleanup();
      intervalId = setInterval(() => {
        cleanup().catch((error) => {
          console.error('[cleanup] failed:', error.message);
        });
      }, 60 * 60 * 1000);
    },
    async stop() {
      if (intervalId) {
        clearInterval(intervalId);
        intervalId = null;
      }
    },
  };
}

