const VALID_COLOR_COUNTS = new Set([2, 3, 4, 5, 6]);
const SAFE_FILE_PATTERN = /^[a-zA-Z0-9._-]+$/;
const SAFE_JOB_PATTERN = /^[a-zA-Z0-9-]+$/;

export function parseColorCount(value) {
  const parsed = Number.parseInt(value, 10);
  if (!VALID_COLOR_COUNTS.has(parsed)) {
    throw new Error('colorCount must be an integer between 2 and 6.');
  }
  return parsed;
}

export function validateUploadedFile(file, allowedMimeTypes) {
  if (!file) {
    throw new Error('An image file is required.');
  }

  if (!allowedMimeTypes.has(file.mimetype)) {
    throw new Error('Only JPG, PNG, and WEBP files are supported.');
  }
}

export function assertSafeAssetReference(jobId, name) {
  if (!SAFE_JOB_PATTERN.test(jobId)) {
    throw new Error('Invalid job id.');
  }

  if (!SAFE_FILE_PATTERN.test(name)) {
    throw new Error('Invalid asset name.');
  }
}

export function buildPublicFileUrl(jobId, name) {
  return `/api/files/${encodeURIComponent(jobId)}/${encodeURIComponent(name)}`;
}

