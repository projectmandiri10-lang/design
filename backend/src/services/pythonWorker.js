import { randomUUID } from 'node:crypto';
import { spawn } from 'node:child_process';
import readline from 'node:readline';

export class PythonWorkerClient {
  constructor({
    pythonExecutable,
    scriptPath,
    modelDir,
    potraceBin,
    timeoutMs,
  }) {
    this.pythonExecutable = pythonExecutable;
    this.scriptPath = scriptPath;
    this.modelDir = modelDir;
    this.potraceBin = potraceBin;
    this.timeoutMs = timeoutMs;

    this.child = null;
    this.pending = new Map();
    this.queue = Promise.resolve();
    this.startPromise = null;
    this.status = {
      state: 'stopped',
      warnings: [],
      health: {},
      startupError: null,
    };
  }

  async start() {
    if (this.startPromise) {
      return this.startPromise;
    }

    this.status.state = 'starting';
    this.status.startupError = null;

    this.startPromise = new Promise((resolve, reject) => {
      const child = spawn(
        this.pythonExecutable,
        [
          this.scriptPath,
          '--worker',
          '--models-dir',
          this.modelDir,
          '--potrace-bin',
          this.potraceBin,
        ],
        {
          stdio: ['pipe', 'pipe', 'pipe'],
          env: {
            ...process.env,
            PYTHONUNBUFFERED: '1',
          },
        },
      );

      this.child = child;

      const readyTimer = setTimeout(() => {
        const error = new Error('Python worker did not become ready in time.');
        this.status.state = 'error';
        this.status.startupError = error.message;
        reject(error);
      }, 15000);

      const stdout = readline.createInterface({ input: child.stdout });
      const stderr = readline.createInterface({ input: child.stderr });

      stdout.on('line', (line) => {
        this.#handleStdout(line, resolve, reject, readyTimer);
      });

      stderr.on('line', (line) => {
        console.error(`[python] ${line}`);
      });

      child.on('exit', (code) => {
        clearTimeout(readyTimer);
        const errorMessage = `Python worker exited with code ${code}.`;
        if (this.status.state !== 'ready') {
          this.status.state = 'error';
          this.status.startupError = errorMessage;
          reject(new Error(errorMessage));
        } else {
          this.status.state = 'stopped';
          this.status.startupError = errorMessage;
        }

        for (const { reject: pendingReject, timeout } of this.pending.values()) {
          clearTimeout(timeout);
          pendingReject(new Error(errorMessage));
        }
        this.pending.clear();
        this.child = null;
        this.startPromise = null;
      });

      child.on('error', (error) => {
        clearTimeout(readyTimer);
        this.status.state = 'error';
        this.status.startupError = error.message;
        reject(error);
        this.startPromise = null;
      });
    });

    return this.startPromise;
  }

  async ensureStarted() {
    if (this.status.state === 'ready' && this.child) {
      return;
    }
    await this.start();
  }

  async submitJob(payload) {
    this.queue = this.queue.catch(() => undefined).then(async () => {
      await this.ensureStarted();
      return this.#dispatch(payload);
    });

    return this.queue;
  }

  getStatus() {
    return { ...this.status };
  }

  async shutdown() {
    if (!this.child) {
      return;
    }

    this.child.kill();
    this.child = null;
    this.status.state = 'stopped';
  }

  #dispatch(payload) {
    const requestId = randomUUID();

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.pending.delete(requestId);
        reject(new Error('The Python pipeline timed out.'));
      }, this.timeoutMs);

      this.pending.set(requestId, { resolve, reject, timeout });
      this.child.stdin.write(`${JSON.stringify({ ...payload, requestId })}\n`);
    });
  }

  #handleStdout(line, resolveStartup, rejectStartup, readyTimer) {
    if (!line.trim()) {
      return;
    }

    let message;
    try {
      message = JSON.parse(line);
    } catch (error) {
      console.error('[python] invalid JSON:', line);
      return;
    }

    if (message.type === 'ready') {
      clearTimeout(readyTimer);
      this.status.state = 'ready';
      this.status.warnings = message.warnings || [];
      this.status.health = message.health || {};
      this.status.startupError = null;
      resolveStartup(this.status);
      return;
    }

    if (message.type === 'result' && message.requestId) {
      const pending = this.pending.get(message.requestId);
      if (!pending) {
        return;
      }

      clearTimeout(pending.timeout);
      this.pending.delete(message.requestId);

      if (message.success) {
        pending.resolve(message.payload);
      } else {
        pending.reject(new Error(message.error || 'Unknown pipeline error.'));
      }
      return;
    }

    if (message.type === 'log') {
      console.info(`[python] ${message.message}`);
      return;
    }

    if (message.type === 'fatal') {
      clearTimeout(readyTimer);
      this.status.state = 'error';
      this.status.startupError = message.error || 'Unknown fatal pipeline error.';
      rejectStartup(new Error(this.status.startupError));
    }
  }
}

