import fs from 'node:fs/promises';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import express from 'express';
import multer from 'multer';
import {
  assertSafeAssetReference,
  buildPublicFileUrl,
  parseColorCount,
  validateUploadedFile,
} from '../utils/vectorizeValidation.js';

function fileExtensionForMime(mimeType) {
  if (mimeType === 'image/png') return '.png';
  if (mimeType === 'image/webp') return '.webp';
  return '.jpg';
}

export function createVectorizeRouter({ config, uploadsDir, outputsDir, worker }) {
  const router = express.Router();

  const storage = multer.diskStorage({
    destination: uploadsDir,
    filename: (_req, file, callback) => {
      callback(null, `${Date.now()}-${randomUUID()}${fileExtensionForMime(file.mimetype)}`);
    },
  });

  const upload = multer({
    storage,
    limits: { fileSize: config.maxFileSizeBytes },
    fileFilter: (_req, file, callback) => {
      if (!config.allowedMimeTypes.has(file.mimetype)) {
        return callback(new Error('Only JPG, PNG, and WEBP files are supported.'));
      }
      callback(null, true);
    },
  });

  router.get('/health', (_req, res) => {
    res.json({
      ok: true,
      worker: worker.getStatus(),
    });
  });

  router.post('/vectorize', upload.single('image'), async (req, res, next) => {
    const tempUploadPath = req.file?.path;

    try {
      validateUploadedFile(req.file, config.allowedMimeTypes);
      const colorCount = parseColorCount(req.body.colorCount);
      const jobId = randomUUID();
      const outputDir = path.join(outputsDir, jobId);

      await fs.mkdir(outputDir, { recursive: true });

      const startedAt = Date.now();
      const payload = await worker.submitJob({
        jobId,
        inputPath: req.file.path,
        colorCount,
        outputDir,
      });
      const totalTimeMs = Date.now() - startedAt;

      const response = {
        jobId,
        svgContent: payload.svgContent,
        svgFile: buildPublicFileUrl(jobId, payload.artifacts.svg),
        processedPreviewFile: buildPublicFileUrl(jobId, payload.artifacts.processedPreview),
        originalFile: buildPublicFileUrl(jobId, payload.artifacts.original),
        palette: payload.palette,
        timings: {
          ...payload.timings,
          total: totalTimeMs,
        },
        warnings: payload.warnings,
        fallbacks: payload.fallbacks,
        metadata: payload.metadata,
      };

      console.info(`[vectorize] job=${jobId} totalMs=${totalTimeMs}`);
      res.json(response);
    } catch (error) {
      next(error);
    } finally {
      if (tempUploadPath) {
        await fs.rm(tempUploadPath, { force: true }).catch(() => undefined);
      }
    }
  });

  router.get('/files/:jobId/:name', async (req, res, next) => {
    try {
      const { jobId, name } = req.params;
      assertSafeAssetReference(jobId, name);
      const filePath = path.join(outputsDir, jobId, name);
      await fs.access(filePath);
      res.sendFile(filePath);
    } catch (error) {
      next(error);
    }
  });

  router.use((error, _req, res, _next) => {
    const status =
      error?.code === 'LIMIT_FILE_SIZE'
        ? 413
        : error.message?.includes('required') || error.message?.includes('supported') || error.message?.includes('colorCount')
          ? 400
          : 500;

    res.status(status).json({
      error: error.message || 'Unexpected server error.',
    });
  });

  return router;
}

