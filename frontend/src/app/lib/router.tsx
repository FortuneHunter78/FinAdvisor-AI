import MainPage from '@/pages/main';
import { createBrowserRouter } from 'react-router';
import RegisterPage from '@/pages/register';
import LoginLayout from '@/shared/layouts/login';
import LoginPage from '@/pages/login/ui';
import MainLayout from '@/shared/layouts/main';

export const router = createBrowserRouter([
  {
    path: '/',
    element: (
      <MainLayout>
        <MainPage />
      </MainLayout>
    ),
  },
  {
    path: '/login',
    element: (
      <LoginLayout>
        <LoginPage />
      </LoginLayout>
    ),
  },
  {
    path: '/register',
    element: (
      <LoginLayout>
        <RegisterPage />
      </LoginLayout>
    ),
  },
]);
