import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import '@/shared/ui/index.css';
import { RouterProvider } from 'react-router';
import { router } from './app/lib/router';
import { ThemeProvider } from './shared/lib/theme-provider';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ThemeProvider>
      <RouterProvider router={router} />
    </ThemeProvider>
  </StrictMode>,
);
