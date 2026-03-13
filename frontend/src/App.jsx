import { useEffect, useMemo, useState } from 'react';
import PreviewCard from './components/PreviewCard.jsx';
import UploadDropzone from './components/UploadDropzone.jsx';
import { toAbsoluteFileUrl, vectorizeImage } from './lib/api.js';

const COLOR_OPTIONS = [2, 3, 4, 5, 6];

function ResultMetric({ label, value }) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-950/70 px-4 py-3">
      <div className="text-xs uppercase tracking-[0.2em] text-slate-500">{label}</div>
      <div className="mt-2 text-lg font-semibold text-white">{value}</div>
    </div>
  );
}

export default function App() {
  const [file, setFile] = useState(null);
  const [colorCount, setColorCount] = useState(4);
  const [error, setError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [result, setResult] = useState(null);

  const localPreviewUrl = useMemo(() => (file ? URL.createObjectURL(file) : null), [file]);
  const processedPreviewUrl = result ? toAbsoluteFileUrl(result.processedPreviewFile) : null;
  const svgPreviewUrl = result?.svgContent
    ? `data:image/svg+xml;charset=utf-8,${encodeURIComponent(result.svgContent)}`
    : null;

  useEffect(() => {
    return () => {
      if (localPreviewUrl) {
        URL.revokeObjectURL(localPreviewUrl);
      }
    };
  }, [localPreviewUrl]);

  function handleFileSelected(nextFile) {
    setError('');
    setResult(null);
    setFile(nextFile);
  }

  async function handleSubmit(event) {
    event.preventDefault();
    if (!file || isSubmitting) {
      return;
    }

    try {
      setIsSubmitting(true);
      setError('');
      const payload = await vectorizeImage(file, colorCount);
      setResult(payload);
    } catch (submitError) {
      setError(submitError.message);
    } finally {
      setIsSubmitting(false);
    }
  }

  function handleDownload() {
    if (!result?.svgContent) {
      return;
    }

    const blob = new Blob([result.svgContent], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${result.jobId || 'vectorized-artwork'}.svg`;
    link.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="mx-auto flex min-h-screen w-full max-w-7xl flex-col px-4 py-8 md:px-8">
      <header className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight text-white md:text-4xl">
          AI Screen Printing Vectorizer
        </h1>
        <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-400 md:text-base">
          Convert photos of printed t-shirt designs into clean SVG artwork with a fully local pipeline:
          perspective correction, texture removal, color reduction, tracing, and curve cleanup.
        </p>
      </header>

      <main className="grid gap-6 lg:grid-cols-[1.2fr_0.8fr]">
        <section className="space-y-6">
          <form className="panel space-y-6 p-6" onSubmit={handleSubmit}>
            <UploadDropzone
              disabled={isSubmitting}
              onError={(message) => setError(message)}
              onFileSelected={handleFileSelected}
            />


            <div className="grid gap-4 md:grid-cols-[1fr_auto] md:items-end">
              <label className="block">
                <span className="text-sm font-semibold text-slate-300">Color reduction</span>
                <select
                  className="mt-2 w-full rounded-xl border border-slate-700 bg-slate-950 px-4 py-3 text-white outline-none transition focus:border-cyan-400"
                  value={colorCount}
                  onChange={(event) => setColorCount(Number(event.target.value))}
                  disabled={isSubmitting}
                >
                  {COLOR_OPTIONS.map((option) => (
                    <option key={option} value={option}>
                      {option} colors
                    </option>
                  ))}
                </select>
              </label>

              <button
                type="submit"
                className="inline-flex h-12 items-center justify-center rounded-xl bg-cyan-500 px-6 font-semibold text-slate-950 transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
                disabled={!file || isSubmitting}
              >
                {isSubmitting ? 'Processing...' : 'Generate SVG'}
              </button>
            </div>

            {file ? (
              <div className="rounded-xl border border-slate-800 bg-slate-950/70 px-4 py-3 text-sm text-slate-300">
                <span className="font-semibold text-white">{file.name}</span>
                <span className="ml-2 text-slate-500">({(file.size / 1024 / 1024).toFixed(2)} MB)</span>
              </div>
            ) : null}

            {error ? (
              <div className="rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-200">
                {error}
              </div>
            ) : null}
          </form>

          <div className="grid gap-6 xl:grid-cols-3">
            <PreviewCard title="Original Photo">
              {localPreviewUrl ? (
                <img src={localPreviewUrl} alt="Original upload preview" className="max-h-80 rounded-xl object-contain" />
              ) : (
                <p className="text-center text-sm text-slate-500">Upload an image to preview the source photo.</p>
              )}
            </PreviewCard>

            <PreviewCard title="Processed Bitmap">
              {processedPreviewUrl ? (
                <img src={processedPreviewUrl} alt="Processed preview" className="max-h-80 rounded-xl object-contain" />
              ) : (
                <p className="text-center text-sm text-slate-500">The cleaned, color-reduced preview appears here.</p>
              )}
            </PreviewCard>

            <PreviewCard title="SVG Preview">
              {svgPreviewUrl ? (
                <img src={svgPreviewUrl} alt="Generated SVG preview" className="max-h-80 rounded-xl bg-white object-contain p-4" />
              ) : (
                <p className="text-center text-sm text-slate-500">The generated SVG appears here after processing.</p>
              )}
            </PreviewCard>
          </div>
        </section>

        <aside className="space-y-6">
          <section className="panel p-6">
            <h2 className="text-lg font-semibold text-white">Result</h2>
            <p className="mt-2 text-sm leading-6 text-slate-400">
              The backend returns one flat SVG file with all artwork colors preserved in a single document.
            </p>

            <div className="mt-6 grid gap-3 md:grid-cols-2">
              <ResultMetric label="Requested colors" value={`${colorCount}`} />
              <ResultMetric
                label="Total time"
                value={result?.timings?.total ? `${Math.round(result.timings.total)} ms` : '-'}
              />
            </div>

            <div className="mt-6">
              <div className="text-sm font-semibold text-slate-300">Palette</div>
              <div className="mt-3 flex flex-wrap gap-3">
                {result?.palette?.length ? (
                  result.palette.map((color) => (
                    <div
                      key={`${color.index}-${color.hex}`}
                      className="flex items-center gap-2 rounded-full border border-slate-800 bg-slate-950 px-3 py-2 text-xs text-slate-200"
                    >
                      <span
                        className="h-4 w-4 rounded-full border border-slate-700"
                        style={{ backgroundColor: color.hex }}
                      />
                      <span>{color.hex}</span>
                    </div>
                  ))
                ) : (
                  <span className="text-sm text-slate-500">Palette data appears after a successful run.</span>
                )}
              </div>
            </div>

            <div className="mt-6">
              <div className="text-sm font-semibold text-slate-300">Warnings</div>
              <ul className="mt-3 space-y-2 text-sm text-slate-400">
                {result?.warnings?.length ? (
                  result.warnings.map((warning) => <li key={warning}>- {warning}</li>)
                ) : (
                  <li>- No warnings reported.</li>
                )}
              </ul>
            </div>

            <button
              type="button"
              onClick={handleDownload}
              className="mt-6 inline-flex w-full items-center justify-center rounded-xl border border-cyan-400 px-4 py-3 font-semibold text-cyan-300 transition hover:bg-cyan-400 hover:text-slate-950 disabled:cursor-not-allowed disabled:border-slate-700 disabled:text-slate-500"
              disabled={!result?.svgContent}
            >
              Download SVG
            </button>
          </section>

          <section className="panel p-6">
            <h2 className="text-lg font-semibold text-white">Pipeline</h2>
            <ol className="mt-4 space-y-3 text-sm text-slate-400">
              <li>1. Perspective correction</li>
              <li>2. Cloth texture removal</li>
              <li>3. AI or classical upscale</li>
              <li>4. Cleanup and foreground isolation</li>
              <li>5. Screen-print color reduction</li>
              <li>6. Potrace vectorization</li>
              <li>7. SVG curve optimization</li>
            </ol>
          </section>
        </aside>
      </main>
    </div>
  );
}
