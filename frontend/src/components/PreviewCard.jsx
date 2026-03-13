export default function PreviewCard({ title, children }) {
  return (
    <section className="panel flex min-h-[22rem] flex-col overflow-hidden">
      <header className="border-b border-slate-800 px-5 py-4">
        <h2 className="text-sm font-semibold uppercase tracking-[0.2em] text-slate-400">{title}</h2>
      </header>
      <div className="flex flex-1 items-center justify-center bg-slate-950/70 p-4">{children}</div>
    </section>
  );
}

