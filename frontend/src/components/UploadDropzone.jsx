import { useRef } from 'react';

const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];

export default function UploadDropzone({ disabled, onError, onFileSelected }) {
  const inputRef = useRef(null);

  function handleFiles(files) {
    const file = files?.[0];
    if (!file) {
      return;
    }
    if (!ACCEPTED_TYPES.includes(file.type)) {
      throw new Error('Only JPG, PNG, and WEBP files are supported.');
    }
    if (file.size > 20 * 1024 * 1024) {
      throw new Error('The maximum file size is 20 MB.');
    }
    onFileSelected(file);
  }

  function handleDrop(event) {
    event.preventDefault();
    if (disabled) {
      return;
    }
    try {
      handleFiles(event.dataTransfer.files);
    } catch (error) {
      onError?.(error.message);
    }
  }

  function handleChange(event) {
    try {
      handleFiles(event.target.files);
    } catch (error) {
      onError?.(error.message);
    }
  }

  return (
    <button
      type="button"
      className="flex min-h-64 w-full flex-col items-center justify-center rounded-2xl border-2 border-dashed border-slate-700 bg-slate-950/60 px-6 py-10 text-center transition hover:border-cyan-400 hover:bg-slate-950 disabled:cursor-not-allowed disabled:opacity-60"
      onDragOver={(event) => event.preventDefault()}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
      disabled={disabled}
    >
      <input
        ref={inputRef}
        type="file"
        className="hidden"
        accept=".jpg,.jpeg,.png,.webp"
        onChange={handleChange}
      />
      <span className="text-lg font-semibold text-white">Drag and drop a t-shirt photo</span>
      <span className="mt-3 max-w-md text-sm text-slate-400">
        Upload a JPG, PNG, or WEBP file up to 20 MB. The pipeline runs fully on your local machine.
      </span>
      <span className="mt-5 rounded-full border border-slate-700 px-4 py-2 text-sm text-cyan-300">
        Click to choose a file
      </span>
    </button>
  );
}
