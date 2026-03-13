import express from 'express';
import cors from 'cors';
import morgan from 'morgan';
import { createVectorizeRouter } from './routes/vectorize.js';

export function createApp({ config, uploadsDir, outputsDir, worker }) {
  const app = express();

  app.use(cors());
  app.use(express.json({ limit: '1mb' }));
  app.use(morgan('dev'));
  app.use('/api', createVectorizeRouter({ config, uploadsDir, outputsDir, worker }));

  app.get('/', (_req, res) => {
    res.json({
      name: 'AI Screen Printing Vectorizer API',
      ok: true,
    });
  });

  return app;
}

