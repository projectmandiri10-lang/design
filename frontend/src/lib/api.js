const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:3001';

export async function vectorizeImage(file, colorCount) {
  const formData = new FormData();
  formData.append('image', file);
  formData.append('colorCount', String(colorCount));

  const response = await fetch(`${API_BASE_URL}/api/vectorize`, {
    method: 'POST',
    body: formData,
  });

  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || 'Vectorization failed.');
  }

  return payload;
}

export function toAbsoluteFileUrl(relativePath) {
  if (!relativePath) {
    return null;
  }
  if (relativePath.startsWith('http://') || relativePath.startsWith('https://')) {
    return relativePath;
  }
  return `${API_BASE_URL}${relativePath}`;
}

